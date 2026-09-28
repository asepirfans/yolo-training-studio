# YOLO Training Studio 🚀

Aplikasi web lokal berbasis Flask yang dirancang untuk mengelola seluruh *end-to-end lifecycle* Computer Vision dengan **Ultralytics YOLO** (YOLOv8 & YOLO11).

---

## ✨ Fitur Utama

- 📁 **Project & Dataset Management**: Mengelompokkan dataset dan runs training ke dalam project terpisah. Mendukung upload ZIP dataset berukuran besar.
- 🏷️ **In-App Labeling & Data Synthesis**: Bounding box annotation langsung di browser, auto-labeling (bootstrap), serta fitur *crop object with feathering*.
- 🎨 **Data Augmentation Studio**: Augmentasi gambar offline (HSV, rotation, blur, noise, flip, mosaic) dengan live preview sebelum diekspor.
- 🚀 **Background Training Engine**: Training YOLO non-blocking menggunakan background worker terpisah, realtime log output via Server-Sent Events (SSE), dan safety queue untuk VRAM GPU.
- ⚡ **Auto-Fallback Hardware**: Mendukung GPU NVIDIA (CUDA) dan otomatis beralih ke CPU jika GPU tidak tersedia.
- 📊 **Metrics & Analytics**: Visualisasi mAP50, mAP50-95, loss curves, confusion matrix, serta download bobot model (`best.pt`, `last.pt`).
- 🔍 **Model Testing & Live Webcam**: Pengujian langsung menggunakan upload gambar/video atau live webcam stream.

---

## 🛠️ Instalasi & Menjalankan

### 1. Prasyarat
- Python 3.8 - 3.11
- PyTorch & torchvision (dengan dukungan CUDA jika memiliki GPU NVIDIA)

### 2. Setup Lingkungan
```bash
# Clone repository
git clone <URL_REPO_ANDA>
cd yolo_training_studio

# Buat virtual environment (opsional tapi disarankan)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependensi
pip install -r requirements.txt
```

### 3. Menjalankan Server
```bash
python run_studio.py
```
Atau di Windows cukup klik dua kali **`start_studio.bat`**.

Akses aplikasi melalui browser:
- Lokal: `http://localhost:5055`
- Jaringan LAN: `http://<IP_PC_ANDA>:5055`
