import os
import sys

# Pastikan direktori yolo_training_studio masuk ke sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import socket
from config import Config
from app import create_app
from app.services.gpu_service import get_gpu_info

def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

app = create_app()

if __name__ == '__main__':
    gpu = get_gpu_info()
    protocol = "https" if Config.USE_SSL else "http"
    lan_ip = get_lan_ip()
    
    print("=" * 70)
    print("🚀 YOLO TRAINING STUDIO (Local Web Application)")
    print(f"📡 Local URL       : {protocol}://localhost:{Config.PORT}")
    print(f"🌐 LAN URL         : {protocol}://{lan_ip}:{Config.PORT}")
    print(f"🔒 SSL Mode        : {'AKTIF (HTTPS) - Kamera USB Didukung' if Config.USE_SSL else 'NONAKTIF (HTTP)'}")
    print(f"📁 Storage Root    : {Config.STORAGE_ROOT}")
    if gpu["available"]:
        print(f"⚡ GPU Akselerasi  : {gpu['device_name']} ({gpu['vram_total_mb']} MB VRAM)")
    else:
        print("⚠️ GPU Akselerasi  : Tidak terdeteksi (Fallback ke CPU)")
    print("=" * 70)
    
    if Config.USE_SSL:
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=False,
            threaded=True,
            ssl_context=(Config.SSL_CERT_PATH, Config.SSL_KEY_PATH)
        )
    else:
        app.run(host=Config.HOST, port=Config.PORT, debug=False, threaded=True)

