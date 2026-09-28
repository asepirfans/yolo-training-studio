import os
from flask import Blueprint, render_template, request, jsonify, session, current_app, redirect, url_for
from app.models import Dataset, Project
from app.services.augmentation_service import (
    generate_live_preview_samples,
    start_augmentation_job,
    get_job_status
)

augmentation_bp = Blueprint('augmentation', __name__)


@augmentation_bp.route('/', methods=['GET'])
def index():
    """
    Halaman utama modul Augmentasi Dataset.
    Memuat daftar dataset dari project yang sedang aktif.
    """
    active_pid = session.get('active_project_id')
    
    # Query dataset
    query = Dataset.query
    if active_pid and active_pid != 'all':
        query = query.filter_by(project_id=active_pid)
    
    datasets = query.order_by(Dataset.created_at.desc()).all()
    
    # Dataset terpilih dari query param atau dataset pertama
    selected_ds_id = request.args.get('dataset_id')
    selected_dataset = None
    if selected_ds_id:
        selected_dataset = Dataset.query.get(selected_ds_id)
    if not selected_dataset and datasets:
        selected_dataset = datasets[0]

    return render_template(
        'augmentation/index.html',
        datasets=datasets,
        selected_dataset=selected_dataset
    )


@augmentation_bp.route('/samples/<dataset_id>', methods=['GET'])
def get_dataset_samples(dataset_id):
    """
    Mengambil daftar nama file gambar pada subset dataset untuk dipilih dalam live preview.
    """
    dataset = Dataset.query.get_or_404(dataset_id)
    if not dataset.yaml_path or not os.path.exists(dataset.yaml_path):
        return jsonify({"success": False, "samples": [], "error": "data.yaml tidak ditemukan"})

    from app.services.dataset_service import parse_yaml_safe, IMAGE_EXTENSIONS
    data = parse_yaml_safe(dataset.yaml_path)
    yaml_dir = os.path.dirname(dataset.yaml_path)
    base_p = data.get('path', '')
    base_dir = os.path.normpath(os.path.join(yaml_dir, base_p)) if base_p else yaml_dir

    train_val = data.get('train') or data.get('val')
    train_dir = os.path.normpath(os.path.join(base_dir, train_val)) if train_val else ''
    if not os.path.exists(train_dir):
        train_dir = os.path.normpath(os.path.join(yaml_dir, train_val)) if train_val else ''

    samples = []
    if os.path.exists(train_dir):
        for root, _, files in os.walk(train_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    samples.append(f)
                    if len(samples) >= 30: # Cukup 30 sampel pilihan
                        break
            if len(samples) >= 30:
                break

    return jsonify({"success": True, "samples": samples})


@augmentation_bp.route('/preview', methods=['POST'])
def preview_sample():
    """
    API untuk menghasilkan pratinjau langsung (Live Preview) beberapa variasi augmentasi
    pada 1 gambar sampel beserta kalkulasi bounding box-nya.
    """
    data = request.get_json() or {}
    dataset_id = data.get('dataset_id')
    sample_filename = data.get('sample_filename')
    config = data.get('config') or {}
    num_variations = int(data.get('num_variations', 3))

    dataset = Dataset.query.get_or_404(dataset_id)
    if not dataset.yaml_path or not os.path.exists(dataset.yaml_path):
        return jsonify({"success": False, "error": "data.yaml dataset tidak ditemukan"}), 400

    try:
        res = generate_live_preview_samples(
            dataset.yaml_path,
            sample_filename=sample_filename,
            config=config,
            num_variations=num_variations
        )
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@augmentation_bp.route('/start', methods=['POST'])
def start_augmentation():
    """
    Memulai pekerjaan augmentasi batch di background thread.
    """
    data = request.get_json() or {}
    source_dataset_id = data.get('source_dataset_id')
    new_dataset_name = (data.get('new_dataset_name') or '').strip()
    config = data.get('config') or {}
    target_subset = data.get('target_subset', 'train') # 'train', 'train_val', 'all'
    multiplier = int(data.get('multiplier', 2))
    include_original = bool(data.get('include_original', True))
    replace_source = bool(data.get('replace_source', False))

    if not source_dataset_id:
        return jsonify({"success": False, "error": "Dataset sumber harus dipilih."}), 400

    dataset = Dataset.query.get_or_404(source_dataset_id)
    if not new_dataset_name:
        new_dataset_name = f"{dataset.name}_aug_{multiplier}x"

    app_instance = current_app._get_current_object()
    job_id = start_augmentation_job(
        app=app_instance,
        source_dataset_id=source_dataset_id,
        new_dataset_name=new_dataset_name,
        config=config,
        target_subset=target_subset,
        multiplier=multiplier,
        include_original=include_original,
        replace_source=replace_source
    )

    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": "Proses augmentasi dataset telah dimulai di background."
    })


@augmentation_bp.route('/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    """
    Mengecek progres pekerjaan augmentasi secara real-time.
    """
    job = get_job_status(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job ID tidak ditemukan"}), 404
    return jsonify({"success": True, "job": job})
