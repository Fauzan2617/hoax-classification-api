import streamlit as st
import requests

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Berita Hoaks", page_icon="🕵️‍♂️", layout="centered")

API_URL = "http://127.0.0.1:8000/predict"

st.title("🕵️‍♂️ Sistem Klasifikasi Berita Hoaks")
st.markdown("Masukkan teks berita di bawah ini, dan sistem AI kami akan memverifikasi apakah itu **Fakta**, **Hoaks**, atau **Perlu Verifikasi** berdasarkan pola teks.")

# Area input teks
news_text = st.text_area("Teks Berita:", height=200, placeholder="Salin dan tempel isi berita di sini... (Minimal 10 karakter)")

# Tombol prediksi
if st.button("Analisis Berita", type="primary"):
    if len(news_text.strip()) < 10:
        st.warning("⚠️ Teks terlalu pendek. Masukkan minimal 10 karakter.")
    else:
        with st.spinner("Menganalisis pola teks dan memvalidasi ke API..."):
            try:
                # Mengirim request POST ke FastAPI
                response = requests.post(API_URL, json={"text": news_text})
                
                if response.status_code == 200:
                    data = response.json()
                    hasil = data["prediksi"]
                    akurasi_str = data["confidence"]
                    
                    # Ekstrak angka dari string (misal: "58.50%" menjadi 58.50)
                    akurasi_float = float(akurasi_str.replace('%', ''))
                    
                    st.markdown("---")
                    st.subheader("Hasil Analisis:")
                    
                    # ==========================================
                    # LOGIKA CONFIDENCE THRESHOLD
                    # ==========================================
                    # Jika model tidak yakin (confidence di bawah 70%)
                    if akurasi_float < 70.0:
                        st.warning(f"⚠️ **Perlu Verifikasi** (Tingkat Keyakinan: {akurasi_str})")
                        st.markdown("Model AI kami mendeteksi sinyal yang bercampur atau struktur bahasa yang ambigu pada berita ini. Sebaiknya lakukan pengecekan silang melalui sumber berita terpercaya atau portal *fact-checking* resmi.")
                    
                    # Jika model yakin di atas 70% dan hasilnya Hoaks
                    elif hasil == "Hoaks":
                        st.error(f"🚨 **{hasil}** (Tingkat Keyakinan: {akurasi_str})")
                        st.markdown("Berita ini memiliki pola kalimat dan struktur yang sangat kuat terindikasi sebagai informasi palsu atau manipulatif.")
                    
                    # Jika model yakin di atas 70% dan hasilnya Fakta
                    else:
                        st.success(f"✅ **{hasil}** (Tingkat Keyakinan: {akurasi_str})")
                        st.markdown("Berita ini memiliki pola yang konsisten dengan informasi faktual yang divalidasi oleh sistem.")
                        
                elif response.status_code == 429:
                    st.error("🛑 Terlalu banyak permintaan. Sistem Rate Limiter aktif. Tunggu sebentar lalu coba lagi.")
                else:
                    st.error(f"❌ Terjadi kesalahan dari server API (Code: {response.status_code})")
                    
            except requests.exceptions.ConnectionError:
                st.error("🔌 Gagal terhubung ke API. Pastikan server FastAPI sudah menyala di terminal terpisah.")