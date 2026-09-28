import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_ROOT = os.environ.get('STORAGE_ROOT', os.path.join(BASE_DIR, 'storage'))
DATASETS_DIR = os.path.join(STORAGE_ROOT, 'datasets')
RUNS_DIR = os.path.join(STORAGE_ROOT, 'runs')
MODELS_DIR = os.path.join(STORAGE_ROOT, 'models')

# Buat direktori penyimpanan jika belum ada
for path in [STORAGE_ROOT, DATASETS_DIR, RUNS_DIR, MODELS_DIR]:
    os.makedirs(path, exist_ok=True)

class Config:
    BASE_DIR = BASE_DIR
    STORAGE_ROOT = STORAGE_ROOT
    DATASETS_DIR = DATASETS_DIR
    RUNS_DIR = RUNS_DIR
    MODELS_DIR = MODELS_DIR

    SECRET_KEY = os.environ.get('SECRET_KEY', 'yolo-training-studio-secret-key-2026')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', f"sqlite:///{os.path.join(STORAGE_ROOT, 'training.db')}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload Settings & Multipart Limits
    MAX_DATASET_UPLOAD_GB = int(os.environ.get('MAX_DATASET_UPLOAD_GB', 20))
    MAX_CONTENT_LENGTH = MAX_DATASET_UPLOAD_GB * 1024 * 1024 * 1024  # bytes
    MAX_FORM_PARTS = None  # Hilangkan limit jumlah part multipart (Werkzeug default 1.000)
    MAX_FORM_MEMORY_SIZE = None  # Hilangkan limit form memory buffer (Werkzeug default 500KB)
    
    # Training concurrency
    MAX_CONCURRENT_TRAINING = int(os.environ.get('MAX_CONCURRENT_TRAINING', 1))
    
    # Models
    AVAILABLE_MODELS = [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
        "yolov8s-obb.pt"
    ]
    DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'yolov8s.pt')
    
    # Server network
    PORT = int(os.environ.get('PORT', 5055))
    HOST = os.environ.get('HOST', '0.0.0.0')

    # SSL / HTTPS Configuration (Default: False / HTTP)
    CERTS_DIR = os.path.join(BASE_DIR, 'certs')
    SSL_CERT_PATH = os.path.join(CERTS_DIR, 'cert.pem')
    SSL_KEY_PATH = os.path.join(CERTS_DIR, 'key.pem')
    USE_SSL = os.environ.get('USE_SSL', 'false').lower() in ('true', '1', 'yes') and os.path.exists(SSL_CERT_PATH) and os.path.exists(SSL_KEY_PATH)

