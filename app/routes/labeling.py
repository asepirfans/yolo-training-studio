import os
import json
import shutil
from flask import (
    Blueprint, render_template, request, jsonify, redirect, 
    url_for, send_from_directory, flash, session, abort
)
from ..models import db
from ..models.labeling import LabelingSession
from ..models.project import Project
from ..services.labeling_service import (
    create_labeling_session, get_session_images, load_annotations,
    save_annotations, crop_object_with_feathering, get_session_crops,
    synthesize_blend_data, split_and_export_dataset, get_session_dir,
    update_crop_layer, delete_crop_layer, delete_session_image,
    bootstrap_auto_label_from_session
)

labeling_bp = Blueprint('labeling', __name__, url_prefix='/labeling')

@labeling_bp.route('/')
def index():
    active_pid = session.get('active_project_id')
    query = LabelingSession.query
    if active_pid and active_pid != 'all':
        query = query.filter_by(project_id=active_pid)
    sessions = query.order_by(LabelingSession.created_at.desc()).all()
    return render_template('labeling/index.html', sessions=sessions)

@labeling_bp.route('/new', methods=['GET'])
def new_session():
    active_pid = session.get('active_project_id')
    projects = Project.query.order_by(Project.name.asc()).all()
    return render_template('labeling/new.html', projects=projects, active_pid=active_pid)

@labeling_bp.route('/create', methods=['POST'])
def create():
    name = request.form.get('name', '').strip()
    project_id = request.form.get('project_id')
    active_pid = session.get('active_project_id')

    if not project_id or project_id == 'all':
        project_id = active_pid if (active_pid and active_pid != 'all') else None

    # Tangkap file: bisa dari multi-file input ('files') atau folder input ('folder_files')
    files = request.files.getlist('files')
    if not files or len(files) == 0 or not files[0].filename:
        files = request.files.getlist('folder_files')

    if not files or len(files) == 0 or not files[0].filename:
        flash("Pilih minimal satu gambar atau satu folder gambar untuk dilabeli.", "error")
        return redirect(url_for('labeling.new_session'))

    try:
        session_rec = create_labeling_session(name, project_id, files)
        flash(f"Sesi pelabelan '{session_rec.name}' berhasil dibuat dengan {session_rec.image_count} gambar.", "success")
        return redirect(url_for('labeling.editor', session_id=session_rec.id))
    except Exception as e:
        flash(f"Gagal membuat sesi pelabelan: {str(e)}", "error")
        return redirect(url_for('labeling.new_session'))

@labeling_bp.route('/<session_id>')
def editor(session_id):
    session_rec = LabelingSession.query.get_or_404(session_id)
    return render_template('labeling/editor.html', session=session_rec)

@labeling_bp.route('/<session_id>/images')
def list_images(session_id):
    LabelingSession.query.get_or_404(session_id)
    images = get_session_images(session_id)
    return jsonify({"success": True, "images": images})

@labeling_bp.route('/<session_id>/image/<filename>', methods=['GET'])
def serve_image(session_id, filename):
    session_dir = get_session_dir(session_id)
    images_dir = os.path.join(session_dir, 'images')
    return send_from_directory(images_dir, filename)

@labeling_bp.route('/<session_id>/image/<filename>', methods=['DELETE'])
def delete_image(session_id, filename):
    LabelingSession.query.get_or_404(session_id)
    try:
        res = delete_session_image(session_id, filename)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@labeling_bp.route('/<session_id>/annotations/<filename>', methods=['GET'])
def get_annotations(session_id, filename):
    LabelingSession.query.get_or_404(session_id)
    bboxes = load_annotations(session_id, filename)
    return jsonify({"success": True, "bboxes": bboxes})

@labeling_bp.route('/<session_id>/annotations/<filename>', methods=['POST'])
def update_annotations(session_id, filename):
    LabelingSession.query.get_or_404(session_id)
    data = request.get_json() or {}
    bboxes = data.get('bboxes', [])
    save_annotations(session_id, filename, bboxes)
    return jsonify({"success": True})

@labeling_bp.route('/<session_id>/classes', methods=['GET', 'POST'])
def manage_classes(session_id):
    LabelingSession.query.get_or_404(session_id)
    from ..services.labeling_service import get_session_classes, save_session_classes, add_class_to_session

    if request.method == 'POST':
        data = request.get_json() or {}
        # Single class addition
        if data.get('name'):
            new_cls, all_classes = add_class_to_session(session_id, data.get('name'), data.get('color'))
            return jsonify({"success": True, "class": new_cls, "classes": all_classes})
        # Bulk update
        if 'classes' in data:
            classes = data.get('classes', [])
            save_session_classes(session_id, classes)
            return jsonify({"success": True, "classes": classes})
    return jsonify({"success": True, "classes": get_session_classes(session_id)})

@labeling_bp.route('/<session_id>/classes/<int:class_id>', methods=['DELETE'])
def delete_class(session_id, class_id):
    LabelingSession.query.get_or_404(session_id)
    from ..services.labeling_service import delete_class_from_session
    try:
        updated_classes = delete_class_from_session(session_id, class_id)
        return jsonify({"success": True, "classes": updated_classes})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@labeling_bp.route('/<session_id>/crop_object', methods=['POST'])
def crop_object(session_id):
    LabelingSession.query.get_or_404(session_id)
    data = request.get_json() or {}
    filename = data.get('filename')
    bbox = data.get('bbox')
    blur_radius = int(data.get('blur_radius', 12))

    if not filename or not bbox:
        return jsonify({"success": False, "error": "Parameter filename dan bbox wajib disertakan."}), 400

    try:
        crop_data = crop_object_with_feathering(session_id, filename, bbox, blur_radius=blur_radius)
        return jsonify({"success": True, "data": crop_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@labeling_bp.route('/<session_id>/crops')
def list_crops(session_id):
    LabelingSession.query.get_or_404(session_id)
    crops = get_session_crops(session_id)
    return jsonify({"success": True, "crops": crops})

@labeling_bp.route('/<session_id>/crops/<filename>')
def serve_crop(session_id, filename):
    session_dir = get_session_dir(session_id)
    crops_dir = os.path.join(session_dir, 'crops')
    return send_from_directory(crops_dir, filename)

@labeling_bp.route('/<session_id>/place_crop', methods=['POST'])
def place_crop(session_id):
    LabelingSession.query.get_or_404(session_id)
    from ..services.labeling_service import place_crop_on_image
    data = request.get_json() or {}
    bg_filename = data.get('bg_filename')
    crop_filename = data.get('crop_filename')
    target_x = float(data.get('target_x_pct', 0.5))
    target_y = float(data.get('target_y_pct', 0.5))
    scale = float(data.get('scale', 1.0))
    class_id = int(data.get('class_id', 0))
    create_new = bool(data.get('create_new', False))

    if not bg_filename or not crop_filename:
        return jsonify({"success": False, "error": "Background image dan Crop file wajib dipilih."}), 400

    try:
        res = place_crop_on_image(session_id, bg_filename, crop_filename, target_x, target_y, scale, class_id, create_new)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@labeling_bp.route('/<session_id>/update_crop_layer', methods=['POST'])
def update_crop_layer_route(session_id):
    LabelingSession.query.get_or_404(session_id)
    data = request.get_json() or {}
    bg_filename = data.get('bg_filename')
    bbox_id = data.get('bbox_id')
    target_x = float(data.get('target_x_pct', 0.5))
    target_y = float(data.get('target_y_pct', 0.5))
    new_w = float(data.get('w')) if data.get('w') is not None else None
    new_h = float(data.get('h')) if data.get('h') is not None else None
    scale = float(data.get('scale')) if data.get('scale') is not None else None
    class_id = int(data.get('class_id')) if data.get('class_id') is not None else None

    if not bg_filename or not bbox_id:
        return jsonify({"success": False, "error": "bg_filename dan bbox_id wajib disertakan."}), 400

    try:
        res = update_crop_layer(session_id, bg_filename, bbox_id, target_x, target_y, new_w, new_h, scale, class_id)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@labeling_bp.route('/<session_id>/delete_crop_layer/<filename>/<bbox_id>', methods=['DELETE'])
def delete_crop_layer_route(session_id, filename, bbox_id):
    LabelingSession.query.get_or_404(session_id)
    try:
        res = delete_crop_layer(session_id, filename, bbox_id)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@labeling_bp.route('/<session_id>/available_models', methods=['GET'])
def get_available_models(session_id):
    session_rec = LabelingSession.query.get_or_404(session_id)
    from ..services.inference_service import get_available_inference_models
    try:
        models = get_available_inference_models(project_id=session_rec.project_id)
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@labeling_bp.route('/<session_id>/auto_label', methods=['POST'])
def auto_label(session_id):
    session_rec = LabelingSession.query.get_or_404(session_id)
    from ..services.labeling_service import run_auto_labeling
    from config import RUNS_DIR
    import uuid
    from werkzeug.utils import secure_filename

    conf_threshold = float(request.form.get('conf_threshold', 0.35))
    target_mode = request.form.get('target_mode', 'current')
    current_filename = request.form.get('current_filename')
    overwrite = (request.form.get('overwrite', 'false').lower() == 'true')
    model_path = request.form.get('model_path')
    model_run_id = request.form.get('model_run_id')

    # Cek jika ada file model .pt yang di-upload
    uploaded_model = request.files.get('model_file')
    if uploaded_model and uploaded_model.filename:
        ext = os.path.splitext(uploaded_model.filename)[1].lower()
        if ext not in ['.pt', '.onnx']:
            return jsonify({"success": False, "error": "Format file model harus berekstensi .pt"}), 400
        
        session_dir = get_session_dir(session_id)
        models_dir = os.path.join(session_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        safe_mname = secure_filename(uploaded_model.filename)
        dest_path = os.path.join(models_dir, f"{uuid.uuid4().hex[:6]}_{safe_mname}")
        uploaded_model.save(dest_path)
        model_path = dest_path

    elif model_run_id:
        cand_path = os.path.join(RUNS_DIR, model_run_id, "outputs", "train", "weights", "best.pt")
        if os.path.exists(cand_path):
            model_path = cand_path
        else:
            return jsonify({"success": False, "error": f"Bobot best.pt untuk run {model_run_id} tidak ditemukan."}), 404

    if not model_path or not os.path.exists(model_path):
        return jsonify({"success": False, "error": "Model YOLO (.pt) belum dipilih atau tidak ditemukan."}), 400

    try:
        res = run_auto_labeling(
            session_id=session_id,
            model_path=model_path,
            target_mode=target_mode,
            current_filename=current_filename,
            conf_threshold=conf_threshold,
            overwrite=overwrite
        )
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@labeling_bp.route('/<session_id>/bootstrap_autolabel', methods=['POST'])
def bootstrap_autolabel(session_id):
    LabelingSession.query.get_or_404(session_id)
    data = request.get_json() or {}
    epochs = int(data.get('epochs', 10))
    conf_threshold = float(data.get('conf_threshold', 0.35))
    overwrite = bool(data.get('overwrite', False))

    try:
        res = bootstrap_auto_label_from_session(session_id, epochs=epochs, conf_threshold=conf_threshold, overwrite=overwrite)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@labeling_bp.route('/<session_id>/synthesize', methods=['POST'])
def synthesize(session_id):
    LabelingSession.query.get_or_404(session_id)
    data = request.get_json() or {}
    bg_filename = data.get('bg_filename')
    crop_filename = data.get('crop_filename')
    target_x = float(data.get('target_x_pct', 0.5))
    target_y = float(data.get('target_y_pct', 0.5))
    scale = float(data.get('scale', 1.0))
    class_id = int(data.get('class_id', 0))

    if not bg_filename or not crop_filename:
        return jsonify({"success": False, "error": "Background image dan Crop file wajib dipilih."}), 400

    try:
        res = synthesize_blend_data(session_id, bg_filename, crop_filename, target_x, target_y, scale, class_id)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@labeling_bp.route('/<session_id>/export_split', methods=['POST'])
def export_split(session_id):
    LabelingSession.query.get_or_404(session_id)
    data = request.get_json() or {}
    train_pct = int(data.get('train_pct', 70))
    val_pct = int(data.get('val_pct', 20))
    test_pct = int(data.get('test_pct', 10))
    custom_name = data.get('dataset_name', '').strip()
    register_to_studio = bool(data.get('register_to_studio', True))

    if (train_pct + val_pct + test_pct) != 100:
        return jsonify({"success": False, "error": "Total rasio Train + Val + Test harus berjumlah 100%."}), 400

    try:
        res = split_and_export_dataset(
            session_id, 
            train_pct=train_pct, 
            val_pct=val_pct, 
            test_pct=test_pct,
            custom_dataset_name=custom_name,
            register_to_studio=register_to_studio
        )
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@labeling_bp.route('/<session_id>/delete', methods=['POST'])
def delete_session(session_id):
    session_rec = LabelingSession.query.get_or_404(session_id)
    session_dir = get_session_dir(session_id)
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass

    db.session.delete(session_rec)
    db.session.commit()
    flash(f"Sesi pelabelan '{session_rec.name}' berhasil dihapus.", "info")
    return redirect(url_for('labeling.index'))
