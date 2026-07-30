# Gunakan image Python versi ringan
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Buat user non-root untuk keamanan (Standar Industri)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Salin file requirements dan install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke dalam container
COPY --chown=user . .

# Expose port 7860 (Wajib untuk HF Spaces)
EXPOSE 7860

# Jalankan Uvicorn server
CMD ["uvicorn", "src/main:app", "--host", "0.0.0.0", "--port", "7860"]