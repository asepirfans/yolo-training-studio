import os
from flask import Blueprint, render_template, request, jsonify, send_file, abort, session
from config import Config
from ..models.dataset import Dataset
from ..services.inference_service import (
    get_available_inference_models,
    run_batch_inference,
    create_session_zip,
    get_dataset_split_samples,
    predict_single_frame
)
from ..services.gpu_service import get_gpu_info

testing_bp = Blueprint('testing', __name__)

@testing_bp.route('/')
def index():
    active_pid = session.get('active_project_id')
    proj_filter = active_pid if (active_pid and active_pid != 'all') else None
    models = get_available_inference_models(project_id=proj_filter)

    ds_query = Dataset.query
    if proj_filter:
        ds_query = ds_query.filter_by(project_id=proj_filter)
    datasets = ds_query.order_by(Dataset.created_at.desc()).all()

    gpu = get_gpu_info()
    preselected_run_id = request.args.get('run_id', '')
    
    return render_template(
        'testing/index.html',
        models=models,
        datasets=datasets,
        gpu=gpu,
        preselected_run_id=preselected_run_id
    )

@testing_bp.route('/run', methods=['POST'])
def run_inference_api():
    model_path = request.form.get('model_path', '').strip()
    if not model_path:
        return jsonify({"success": False, "error": "Pilih model yang akan digunakan untuk pengujian."}), 400

    source_type = request.form.get('source_type', 'upload')
    manual_type = request.form.get('manual_type', 'direct')  # 'direct' atau 'verify'
    conf = float(request.form.get('conf', 0.25))
    iou = float(request.form.get('iou', 0.45))
    imgsz = int(request.form.get('imgsz', 640))
    device = request.form.get('device', '0')

    # Tentukan apakah mode evaluasi pembanding (OK vs NG) aktif
    eval_mode = True if source_type == 'dataset' else (manual_type == 'verify')

    try:
        if source_type == 'dataset':
            dataset_id = request.form.get('dataset_id', '').strip()
            split = request.form.get('split', 'val').strip()
            limit = int(request.form.get('limit', 50))
            if not dataset_id:
                return jsonify({"success": False, "error": "Pilih dataset yang akan diuji."}), 400
                
            data = run_batch_inference(
                model_path=model_path,
                files=None,
                label_files=None,
                dataset_id=dataset_id,
                split=split,
                limit=limit,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                eval_mode=True
            )
        else:
            if 'files' not in request.files:
                return jsonify({"success": False, "error": "Tidak ada file gambar yang diunggah."}), 400
                
            files = request.files.getlist('files')
            if not files or len(files) == 0:
                return jsonify({"success": False, "error": "Daftar file gambar kosong."}), 400

            label_files = request.files.getlist('label_files') if ('label_files' in request.files and eval_mode) else None

            data = run_batch_inference(
                model_path=model_path,
                files=files,
                label_files=label_files,
                dataset_id=None,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                eval_mode=eval_mode
            )

        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@testing_bp.route('/image/<session_id>/<filename>')
def serve_annotated_image(session_id, filename):
    # Proteksi path traversal
    session_dir = os.path.abspath(os.path.join(Config.STORAGE_ROOT, "temp_inference", session_id, "annotated"))
    target_file = os.path.abspath(os.path.join(session_dir, filename))
    
    if not target_file.startswith(session_dir):
        abort(403)
        
    if not os.path.exists(target_file):
        abort(404)
        
    return send_file(target_file)

@testing_bp.route('/download-zip/<session_id>')
def download_zip(session_id):
    zip_path = create_session_zip(session_id)
    if not zip_path or not os.path.exists(zip_path):
        abort(404)
        
    return send_file(zip_path, as_attachment=True, download_name=f"dataset_test_results_{session_id}.zip")


@testing_bp.route('/live_predict', methods=['POST'])
def live_predict_api():
    """
    Endpoint inferensi real-time untuk 1 frame webcam dari client.
    Menerima frame gambar (multipart form 'frame' atau 'image_base64').
    """
    model_path = request.form.get('model_path', '').strip()
    if not model_path:
        json_data = request.get_json(silent=True) or {}
        model_path = json_data.get('model_path', '').strip()
        
    if not model_path:
        return jsonify({"success": False, "error": "Pilih model yang akan digunakan untuk inferensi."}), 400

    try:
        conf = float(request.form.get('conf', 0.25))
    except Exception:
        conf = 0.25

    try:
        iou = float(request.form.get('iou', 0.45))
    except Exception:
        iou = 0.45

    try:
        imgsz = int(request.form.get('imgsz', 640))
    except Exception:
        imgsz = 640

    device = request.form.get('device', '0')

    # Dapatkan data frame gambar
    frame_data = None
    if 'frame' in request.files:
        file_obj = request.files['frame']
        frame_data = file_obj.read()
    elif 'image_base64' in request.form:
        frame_data = request.form['image_base64']
    elif request.is_json:
        json_body = request.get_json(silent=True) or {}
        frame_data = json_body.get('image_base64')

    if not frame_data:
        return jsonify({"success": False, "error": "Frame gambar kosong."}), 400

    try:
        res = predict_single_frame(
            model_path=model_path,
            frame_data=frame_data,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device
        )
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

