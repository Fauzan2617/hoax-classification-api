import os
import re
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from huggingface_hub import hf_hub_download
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ==========================================
# 1. KONFIGURASI KEAMANAN & LINGKUNGAN
# ==========================================
HF_REPO_ID = "FauzanDwi/stacking-hoax-detector" # Ganti dengan ID repo kamu
MODEL_FILENAME = "stacking_final_tuning.pkl"
ALLOWED_ORIGINS = [
    "http://localhost:8501", # Untuk testing Streamlit lokal
    "https://domain-frontend-kamu.com" # Ganti dengan domain production nanti
]

# Inisialisasi Rate Limiter (Berbasis IP Address)
limiter = Limiter(key_func=get_remote_address)

# Variabel global untuk menyimpan model
model = None

# ==========================================
# 2. LIFESPAN (STARTUP & SHUTDOWN)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("[INFO] Mengunduh model dari Hugging Face Hub...")
    try:
        # Mengunduh secara aman ke temporary cache
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        model = joblib.load(model_path)
        print("[INFO] Model berhasil dimuat ke dalam memori.")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
        raise RuntimeError("Model tidak dapat dimuat, server dihentikan.")
    
    yield # Aplikasi berjalan di sini
    
    # Clean up (jika aplikasi dimatikan)
    model = None
    print("[INFO] Model dihapus dari memori. Server dimatikan.")

# Inisialisasi FastAPI
app = FastAPI(
    title="Hoax Classification API",
    description="API Aman berstandar industri untuk klasifikasi berita hoaks.",
    version="1.0.0",
    lifespan=lifespan
)

# Menempelkan Rate Limiter ke Aplikasi
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Konfigurasi CORS Super Ketat
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"], # Hanya izinkan metode POST untuk keamanan
    allow_headers=["*"],
)

# ==========================================
# 3. PYDANTIC SCHEMAS (SECURITY VALIDATION)
# ==========================================
class NewsInput(BaseModel):
    # Celah OOM ditutup: Panjang minimal 10, maksimal 3000 karakter
    text: str = Field(..., min_length=10, max_length=3000, description="Teks berita yang akan diuji")

    @field_validator('text')
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        # Celah XSS ditutup: Hapus semua tag HTML seperti <script> atau <div>
        clean_text = re.sub(r'<[^>]+>', '', v)
        
        # Mencegah karakter aneh, hanya izinkan huruf, angka, dan tanda baca umum
        clean_text = re.sub(r'[^\w\s.,!?\'"-]', '', clean_text)
        
        if len(clean_text.strip()) < 10:
            raise ValueError("Teks tidak valid atau terlalu pendek setelah dibersihkan.")
        return clean_text.strip()

# ==========================================
# 4. ENDPOINT PREDIKSI
# ==========================================
@app.post("/predict")
@limiter.limit("5/minute") # Celah DDoS ditutup: Maks 5 request per menit per IP
async def predict_hoax(request: Request, payload: NewsInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap dilayani.")
    
    try:
        # Prediksi menggunakan model pipeline
        prediction = model.predict([payload.text])[0]
        # Jika model support probabilitas, kita bisa ambil *confidence score*-nya
        probabilities = model.predict_proba([payload.text])[0]
        confidence = float(max(probabilities)) * 100

        # Mapping hasil (0 = Fakta, 1 = Hoaks) - sesuaikan dengan label trainingmu
        label_map = {0: "Fakta", 1: "Hoaks"}
        result_label = label_map.get(prediction, "Tidak Diketahui")

        # Nanti di Fase 3, kita akan tambahkan kode logging MLflow di sini
        
        return {
            "status": "success",
            "teks_bersih": payload.text[:100] + "...", # Kembalikan sedikit teks untuk log
            "prediksi": result_label,
            "confidence": f"{confidence:.2f}%"
        }
        
    except Exception as e:
        # Mencegah Stack Trace leak: Jangan mengembalikan error asli Python ke user!
        print(f"[ERROR PREDICTION] {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal saat memproses prediksi.")

# ==========================================
# 5. HEALTH CHECK
# ==========================================
@app.get("/")
@limiter.limit("10/minute")
async def root(request: Request):
    return {"message": "Hoax Classification API sedang berjalan dengan aman."}