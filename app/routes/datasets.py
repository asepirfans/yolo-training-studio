import os
import uuid
import json
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, send_file, session
from werkzeug.utils import secure_filename
from config import DATASETS_DIR
from ..models import db
from ..models.dataset import Dataset
from ..models.project import Project
from ..services.dataset_service import (
    extract_zip_securely, find_yaml_files, validate_dataset_structure,
    get_dataset_preview_samples
)

datasets_bp = Blueprint('datasets', __name__)

@datasets_bp.route('/')
def list_datasets():
    active_pid = session.get('active_project_id')
    query = Dataset.query
    if active_pid and active_pid != 'all':
        query = query.filter((Dataset.project_id == active_pid) | (Dataset.project_id == None))
    datasets = query.order_by(Dataset.created_at.desc()).all()
    return render_template('datasets/list.html', datasets=datasets)

@datasets_bp.route('/upload', methods=['GET'])
def upload_page():
    projects = Project.query.order_by(Project.name.asc()).all()
    return render_template('datasets/upload.html', projects=projects)

@datasets_bp.route('/upload', methods=['POST'])
def upload_dataset():
    if 'dataset_zip' not in request.files:
        flash("Tidak ada file ZIP yang diunggah.", "error")
        return redirect(url_for('datasets.upload_page'))

    file = request.files['dataset_zip']
    if file.filename == '':
        flash("Nama file kosong.", "error")
        return redirect(url_for('datasets.upload_page'))

    if not file.filename.lower().endswith('.zip'):
        flash("Format file harus berupa .zip.", "error")
        return redirect(url_for('datasets.upload_page'))

    custom_name = request.form.get('dataset_name', '').strip()
    safe_filename = secure_filename(file.filename)
    if not custom_name:
        custom_name = os.path.splitext(safe_filename)[0]

    # Buat direktori unik dataset
    dataset_id = f"dataset_{uuid.uuid4().hex[:8]}"
    storage_path = os.path.join(DATASETS_DIR, dataset_id)
    os.makedirs(storage_path, exist_ok=True)

    zip_dest = os.path.join(storage_path, "original.zip")
    file.save(zip_dest)

    # Ekstraksi ZIP
    extract_folder = os.path.join(storage_path, "extracted")
    try:
        extract_zip_securely(zip_dest, extract_folder)
    except Exception as e:
        flash(f"Gagal mengekstrak ZIP: {str(e)}", "error")
        return redirect(url_for('datasets.upload_page'))

    # Deteksi file YAML
    yaml_files = find_yaml_files(extract_folder)
    yaml_path = yaml_files[0] if yaml_files else None

    # Validasi dataset awal
    train_c, val_c, test_c, num_cls = 0, 0, 0, 0
    classes_dict = {}

    if yaml_path:
        val_rep = validate_dataset_structure(extract_folder, yaml_path)
        train_c = val_rep.get("train_count", 0)
        val_c = val_rep.get("val_count", 0)
        test_c = val_rep.get("test_count", 0)
        num_cls = val_rep.get("num_classes", 0)
        classes_dict = val_rep.get("class_names", {})

    # Tentukan project tujuan
    target_project_id = request.form.get('project_id') or session.get('active_project_id')
    if not target_project_id or target_project_id == 'all':
        first_proj = Project.query.first()
        target_project_id = first_proj.id if first_proj else None

    # Simpan ke Database
    new_dataset = Dataset(
        id=dataset_id,
        name=custom_name,
        upload_filename=safe_filename,
        storage_path=storage_path,
        yaml_path=yaml_path,
        num_classes=num_cls,
        class_names_json=json.dumps(classes_dict),
        train_count=train_c,
        val_count=val_c,
        test_count=test_c,
        project_id=target_project_id
    )
    db.session.add(new_dataset)
    db.session.commit()

    flash(f"Dataset '{custom_name}' berhasil diunggah dan diekstrak!", "success")
    return redirect(url_for('datasets.detail', dataset_id=dataset_id))

@datasets_bp.route('/<dataset_id>')
def detail(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    validation_report = None
    if dataset.yaml_path and os.path.exists(dataset.yaml_path):
        validation_report = validate_dataset_structure(dataset.storage_path, dataset.yaml_path)

    return render_template(
        'datasets/detail.html',
        dataset=dataset,
        validation=validation_report
    )

@datasets_bp.route('/<dataset_id>/validate', methods=['POST'])
def validate_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if not dataset.yaml_path or not os.path.exists(dataset.yaml_path):
        return jsonify({"success": False, "error": "File YAML tidak ditemukan"}), 400

    report = validate_dataset_structure(dataset.storage_path, dataset.yaml_path)
    
    # Update data di database
    dataset.train_count = report.get("train_count", 0)
    dataset.val_count = report.get("val_count", 0)
    dataset.test_count = report.get("test_count", 0)
    dataset.num_classes = report.get("num_classes", 0)
    dataset.class_names_json = json.dumps(report.get("class_names", {}))
    db.session.commit()

    return jsonify({"success": True, "report": report})

@datasets_bp.route('/<dataset_id>/preview', methods=['GET'])
def preview_samples(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if not dataset.yaml_path or not os.path.exists(dataset.yaml_path):
        return jsonify({"success": False, "samples": []})

    samples = get_dataset_preview_samples(dataset.yaml_path, max_samples=12)
    # Format URL gambar
    for s in samples:
        s["image_url"] = url_for('datasets.serve_dataset_file', dataset_id=dataset_id, file_path=os.path.relpath(s["filepath"], dataset.storage_path).replace('\\', '/'))

    return jsonify({"success": True, "samples": samples})

@datasets_bp.route('/<dataset_id>/files/<path:file_path>')
def serve_dataset_file(dataset_id, file_path):
    dataset = Dataset.query.get_or_404(dataset_id)
    abs_path = os.path.abspath(os.path.join(dataset.storage_path, file_path))
    # Proteksi path traversal
    if not abs_path.startswith(os.path.abspath(dataset.storage_path)):
        return "Forbidden", 403

    if not os.path.exists(abs_path):
        return "File Not Found", 404

    return send_file(abs_path)

@datasets_bp.route('/<dataset_id>/delete', methods=['POST'])
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    name = dataset.name
    
    # Hapus file fisik di storage
    if os.path.exists(dataset.storage_path):
        try:
            import shutil
            shutil.rmtree(dataset.storage_path, ignore_errors=True)
        except Exception:
            pass

    db.session.delete(dataset)
    db.session.commit()

    flash(f"Dataset '{name}' berhasil dihapus.", "info")
    return redirect(url_for('datasets.list_datasets'))

@datasets_bp.route('/<dataset_id>/download')
def download_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    zip_path = None
    # 1. Cek upload_filename
    if dataset.upload_filename and dataset.storage_path:
        candidate = os.path.join(dataset.storage_path, dataset.upload_filename)
        if os.path.exists(candidate) and os.path.isfile(candidate):
            zip_path = candidate

    # 2. Cek original.zip
    if not zip_path and dataset.storage_path:
        candidate = os.path.join(dataset.storage_path, "original.zip")
        if os.path.exists(candidate) and os.path.isfile(candidate):
            zip_path = candidate

    # 3. Cari file *.zip di folder storage_path
    if not zip_path and dataset.storage_path and os.path.exists(dataset.storage_path):
        for fname in os.listdir(dataset.storage_path):
            if fname.lower().endswith('.zip'):
                candidate = os.path.join(dataset.storage_path, fname)
                if os.path.isfile(candidate):
                    zip_path = candidate
                    break

    # 4. Jika belum ada file ZIP fisik, bungkus on-the-fly dari extracted folder
    if not zip_path or not os.path.exists(zip_path):
        import zipfile
        source_dir = None
        if dataset.yaml_path and os.path.exists(dataset.yaml_path):
            source_dir = os.path.dirname(dataset.yaml_path)
        else:
            cand = os.path.join(dataset.storage_path, 'extracted', 'dataset')
            if os.path.exists(cand):
                source_dir = cand
            else:
                cand = os.path.join(dataset.storage_path, 'extracted')
                source_dir = cand if os.path.exists(cand) else dataset.storage_path

        clean_name = secure_filename(dataset.name).strip() or f"dataset_{dataset.id[-6:]}"
        zip_filename = f"{clean_name}.zip"
        os.makedirs(dataset.storage_path, exist_ok=True)
        zip_path = os.path.join(dataset.storage_path, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if file.lower().endswith('.zip'):
                        continue
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, source_dir)
                    zipf.write(full_p, rel_p)

        dataset.upload_filename = zip_filename
        db.session.commit()

    download_name = f"{secure_filename(dataset.name) or 'dataset'}.zip"
    return send_file(zip_path, as_attachment=True, download_name=download_name)
