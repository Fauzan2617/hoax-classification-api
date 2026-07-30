import os
import re
import joblib
import mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from huggingface_hub import hf_hub_download
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv  

load_dotenv()  

# ==========================================
# 1. KONFIGURASI
# ==========================================
HF_REPO_ID = "FauzanDwi/stacking-hoax-detector"  
MODEL_FILENAME = "stacking_final_tuning.pkl"               
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"  

ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "https://domain-frontend-kamu.com"
]

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")

limiter = Limiter(key_func=get_remote_address)

# Deklarasi variabel global
model = None
vectorizer = None 


# ==========================================
# 2. BACKGROUND TASK — MLflow Logging
# ==========================================
def log_to_mlflow(input_text: str, prediction: str, confidence: float):
    try:
        mlflow.set_experiment("Production_Hoax_Monitoring")  
        with mlflow.start_run():
            mlflow.log_param("input_length", len(input_text))   
            mlflow.log_param("prediction", prediction)          
            mlflow.log_metric("confidence", confidence)          
            mlflow.log_text(input_text, "input_text.txt")        
    except Exception as e:
        print(f"[WARNING] Gagal mengirim log ke MLflow: {e}")


# ==========================================
# 3. LIFESPAN — Startup & Shutdown
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer
    print("[INFO] Mengunduh model dan vectorizer dari Hugging Face Hub...")
    try:
        # 1. Load Model Stacking
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        model = joblib.load(model_path)
        
        # 2. Load Vectorizer TF-IDF (force_download untuk bypass cache lama)
        vec_path = hf_hub_download(repo_id=HF_REPO_ID, filename=VECTORIZER_FILENAME, force_download=True)
        # Langsung di-load ke variabel vectorizer
        vectorizer = joblib.load(vec_path)
        
        print("[INFO] Model dan Vectorizer berhasil dimuat ke dalam memori.")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model/vectorizer: {e}")
        raise RuntimeError("Model atau Vectorizer tidak dapat dimuat.")

    yield  

    # Bersihkan memori saat shutdown
    model = None  
    vectorizer = None


# ==========================================
# INISIALISASI APP
# ==========================================
app = FastAPI(title="Hoax Classification API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"]
)


# ==========================================
# 4. SCHEMA INPUT (Pydantic)
# ==========================================
class NewsInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=3000)  

    @field_validator('text')
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        clean_text = re.sub(r'<[^>]+>', '', v)                    
        clean_text = re.sub(r'[^\w\s.,!?\'"-]', '', clean_text)   
        if len(clean_text.strip()) < 10:
            raise ValueError("Teks tidak valid setelah dibersihkan.")
        return clean_text.strip()


# ==========================================
# 5. ENDPOINT PREDIKSI
# ==========================================
@app.post("/predict")
@limiter.limit("5/minute")  
async def predict_hoax(request: Request, payload: NewsInput, background_tasks: BackgroundTasks):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model atau Vectorizer belum siap.")

    try:
        # 1. Transformasi teks menjadi vektor matriks
        teks_vektor = vectorizer.transform([payload.text])

        # 2. Masukkan vektor yang sudah berbentuk angka ke dalam model
        prediction    = model.predict(teks_vektor)[0]           # Prediksi label (0 atau 1)
        probabilities = model.predict_proba(teks_vektor)[0]     # Probabilitas tiap kelas
        confidence    = float(max(probabilities)) * 100         # Ambil confidence tertinggi

        label_map    = {0: "Fakta", 1: "Hoaks"}
        result_label = label_map.get(prediction, "Tidak Diketahui")

        background_tasks.add_task(log_to_mlflow, payload.text, result_label, confidence)

        return {
            "status": "success",
            "teks_bersih": payload.text[:100] + "...",  
            "prediksi": result_label,
            "confidence": f"{confidence:.2f}%"
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail="Kesalahan internal server.")


@app.get("/")
@limiter.limit("10/minute")
async def root(request: Request):
    return {"message": "Hoax Classification API beroperasi dengan MLOps aktif."}