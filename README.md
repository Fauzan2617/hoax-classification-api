Markdown
# 🕵️‍♂️ Sistem Klasifikasi Berita Hoaks (End-to-End ML Pipeline)

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)

Repositori ini berisi kode sumber untuk **Sistem Deteksi Berita Hoaks**, sebuah proyek tugas akhir (skripsi) yang mengimplementasikan arsitektur *Machine Learning* secara *end-to-end*. Proyek ini mendeteksi pola teks manipulatif atau informasi palsu berbahasa Indonesia menggunakan pendekatan *Stacking Ensemble Learning*.

## 🚀 Fitur Utama
* **Klasifikasi Real-Time:** Memproses dan memverifikasi teks berita dalam hitungan detik.
* **Arsitektur Microservices:** Pemisahan *frontend* (Streamlit) dan *backend* (FastAPI) untuk skalabilitas.
* **MLOps Tracking:** Terintegrasi dengan DagsHub dan MLflow untuk pelacakan eksperimen dan siklus hidup model.
* **Model Registry:** Penyimpanan model terpusat menggunakan Hugging Face Hub.
* **Containerized Deployment:** Dibungkus menggunakan Docker untuk memastikan konsistensi lingkungan dari tahap pengembangan hingga produksi di Back4App.

## 🛠️ Teknologi yang Digunakan
* **Machine Learning:** Python 3.10, Scikit-Learn `v1.7.2` (Strict), TF-IDF Vectorizer.
* **Algoritma:** Stacking Ensemble (Kombinasi Naive Bayes & Logistic Regression).
* **MLOps & Tracking:** DagsHub, MLflow, Hugging Face Hub.
* **Backend API:** FastAPI, Uvicorn.
* **Frontend UI:** Streamlit.
* **Infrastruktur/Deployment:** Docker, Back4App Containers.

---

## ⚙️ Panduan Instalasi (Lokal)

### Prasyarat
Pastikan sistem Anda sudah terinstal:
* Python 3.10+
* Git
* Docker (Opsional, jika ingin menjalankan via *container*)

### 1. Clone Repositori
```bash
git clone [https://github.com/username_kamu/repo_name.git](https://github.com/username_kamu/repo_name.git)
cd repo_name
2. Setup Virtual Environment (Direkomendasikan)
Bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
3. Instalasi Dependensi
Penting: Pastikan versi Scikit-Learn tepat berada di 1.7.2 untuk menghindari error Unpickling.

Bash
pip install -r requirements.txt
4. Menjalankan Aplikasi
Menjalankan Backend (FastAPI):

Bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
API akan berjalan di http://localhost:8000. Dokumentasi interaktif (Swagger UI) dapat diakses di http://localhost:8000/docs.

Menjalankan Frontend (Streamlit):
Buka terminal baru, pastikan virtual environment aktif, lalu jalankan:

Bash
streamlit run app.py
Antarmuka pengguna akan terbuka di browser pada http://localhost:8501.

🐳 Deployment dengan Docker
Proyek ini sudah dilengkapi dengan Dockerfile untuk kemudahan deployment.

Bash
# Build Docker Image
docker build -t hoax-classifier-api .

# Run Docker Container
docker run -p 8000:8000 hoax-classifier-api
📡 API Endpoint (Referensi Singkat)
POST /predict
Mengirim teks berita untuk dianalisis oleh model.

Request Body (JSON):

JSON
{
  "teks_berita": "AWAS VIRALKAN!! Tahukah ibu-ibu keluarga tercinta, jangan pernah lagi makan seblak pedas..."
}
Response (JSON):

JSON
{
  "status": "success",
  "prediksi": "Hoaks",
  "confidence_score": 91.89,
  "pesan": "Berita ini memiliki pola kalimat yang terindikasi manipulatif."
}
👨‍💻 Penulis
Muhammad Fauzan Dwi Putera

Mahasiswa Teknik Informatika, Universitas Pasundan

LinkedIn

GitHub

Proyek ini dikembangkan sebagai bagian dari penelitian sarjana (Skripsi).


***

**Beberapa tips tambahan sebelum di-push:**
1. Pastikan file `requirements.txt`, `Dockerfile`, `main.py` (FastAPI), dan `app.py` (Streamlit) sudah berada di *root* folder repositori kamu agar sesuai dengan instruksi di atas.
2. Ubah `username_kamu` dan `repo_name` pada bagian *Clone Repositori* dengan URL asli milikmu.
3. Tambahkan tautan profil LinkedIn kamu di bagian **Penulis** agar perekrut atau penguji bisa
