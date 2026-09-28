import sys
from flask import Blueprint, render_template, jsonify
from ..services.gpu_service import get_gpu_info, get_system_resources

try:
    import torch
except ImportError:
    torch = None

try:
    import ultralytics
except ImportError:
    ultralytics = None

system_bp = Blueprint('system', __name__)

@system_bp.route('/')
def diagnostics():
    gpu = get_gpu_info()
    res = get_system_resources()

    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__ if torch else "Not Installed",
        "cuda_available": torch.cuda.is_available() if torch else False,
        "cuda_version": torch.version.cuda if (torch and hasattr(torch.version, 'cuda')) else "N/A",
        "cudnn_version": torch.backends.cudnn.version() if (torch and hasattr(torch.backends, 'cudnn')) else "N/A",
        "ultralytics_version": getattr(ultralytics, '__version__', 'Not Installed') if ultralytics else "Not Installed",
        "gpu": gpu,
        "resources": res
    }

    return render_template('system.html', env=env_info)

@system_bp.route('/api/gpu')
def api_gpu():
    return jsonify(get_gpu_info())

@system_bp.route('/api/resources')
def api_resources():
    return jsonify(get_system_resources())
