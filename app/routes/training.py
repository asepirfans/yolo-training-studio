import os
import re
import json
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, Response, flash, session
from config import Config
from ..models import db
from ..models.dataset import Dataset
from ..models.training_run import TrainingRun
from ..services.gpu_service import get_available_devices, get_gpu_info, get_system_resources
from ..services.training_service import (
    create_training_run, stop_training_run, sync_run_state,
    generate_sse_logs, get_run_dir
)
from ..services.result_service import parse_results_csv, find_best_training_run

training_bp = Blueprint('training', __name__)

@training_bp.route('/new', methods=['GET'])
def new_training():
    active_pid = session.get('active_project_id')
    query = Dataset.query
    if active_pid and active_pid != 'all':
        query = query.filter_by(project_id=active_pid)
    datasets = query.order_by(Dataset.created_at.desc()).all()
    selected_dataset_id = request.args.get('dataset_id')
    selected_dataset = None
    if selected_dataset_id:
        selected_dataset = Dataset.query.get(selected_dataset_id)
        if active_pid and active_pid != 'all' and selected_dataset and selected_dataset.project_id != active_pid:
            selected_dataset = None

    devices = get_available_devices()
    gpu_info = get_gpu_info()

    return render_template(
        'training/config.html',
        datasets=datasets,
        selected_dataset=selected_dataset,
        available_models=Config.AVAILABLE_MODELS,
        devices=devices,
        gpu=gpu_info
    )

@training_bp.route('/start', methods=['POST'])
def start_training():
    data = request.get_json(silent=True) or request.form.to_dict()
    
    # Parsing payload jika datang dari form biasa atau JSON
    if request.is_json:
        payload = request.json
    else:
        dataset_id = request.form.get('dataset_id')
        model_name = request.form.get('model_name', Config.DEFAULT_MODEL)
        
        # Hyperparameters
        training_cfg = {
            "epochs": int(request.form.get('epochs', 100)),
            "batch": int(request.form.get('batch', 16)) if request.form.get('batch', '').isdigit() else request.form.get('batch', 16),
            "imgsz": int(request.form.get('imgsz', 640)),
            "workers": int(request.form.get('workers', 0)),
            "patience": int(request.form.get('patience', 50)),
            "device": request.form.get('device', '0'),
            "seed": int(request.form.get('seed', 0)),
            "cache": request.form.get('cache') == 'true' or request.form.get('cache') == 'on',
            "pretrained": True
        }

        # Optimization
        optim_cfg = {
            "optimizer": request.form.get('optimizer', 'AdamW'),
            "lr0": float(request.form.get('lr0', 0.001)),
            "lrf": float(request.form.get('lrf', 0.01)),
            "momentum": float(request.form.get('momentum', 0.937)),
            "weight_decay": float(request.form.get('weight_decay', 0.0005)),
            "warmup_epochs": float(request.form.get('warmup_epochs', 3.0)),
            "close_mosaic": int(request.form.get('close_mosaic', 10))
        }

        # Augmentation
        aug_enabled = request.form.get('aug_enabled') in ('true', 'on', '1')
        aug_cfg = {
            "enabled": aug_enabled,
            "hsv_h": float(request.form.get('hsv_h', 0.015)),
            "hsv_s": float(request.form.get('hsv_s', 0.7)),
            "hsv_v": float(request.form.get('hsv_v', 0.4)),
            "degrees": float(request.form.get('degrees', 0.0)),
            "translate": float(request.form.get('translate', 0.1)),
            "scale": float(request.form.get('scale', 0.5)),
            "shear": float(request.form.get('shear', 0.0)),
            "flipud": float(request.form.get('flipud', 0.0)),
            "fliplr": float(request.form.get('fliplr', 0.5)),
            "mosaic": float(request.form.get('mosaic', 1.0) if aug_enabled else 0.0)
        }

        payload = {
            "dataset_id": dataset_id,
            "model": model_name,
            "training": training_cfg,
            "optimization": optim_cfg,
            "augmentation": aug_cfg
        }

    dataset_id = payload.get('dataset_id')
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        if request.is_json:
            return jsonify({"success": False, "error": "Dataset tidak ditemukan"}), 404
        flash("Dataset tidak ditemukan.", "error")
        return redirect(url_for('training.new_training'))

    # Strict project isolation for training run
    active_pid = session.get('active_project_id')
    if active_pid and active_pid != 'all':
        payload['project_id'] = active_pid
    else:
        payload['project_id'] = dataset.project_id or (Project.query.first().id if Project.query.first() else None)

    # Buat training run dan jalankan
    run_record = create_training_run(dataset, payload)

    if request.is_json:
        return jsonify({"success": True, "run_id": run_record.id, "status": run_record.status})

    return redirect(url_for('training.monitor', run_id=run_record.id))

@training_bp.route('/<run_id>')
def monitor(run_id):
    run_record = TrainingRun.query.get_or_404(run_id)
    sync_run_state(run_record)
    gpu_info = get_gpu_info()
    sys_res = get_system_resources()

    return render_template(
        'training/monitor.html',
        run=run_record,
        gpu=gpu_info,
        resources=sys_res
    )

@training_bp.route('/history')
def history():
    status_filter = request.args.get('status')
    sort_by = request.args.get('sort', 'latest')

    query = TrainingRun.query
    active_pid = session.get('active_project_id')
    if active_pid and active_pid != 'all':
        query = query.filter_by(project_id=active_pid)

    if status_filter:
        query = query.filter_by(status=status_filter)

    if sort_by == 'highest_map50':
        query = query.order_by(TrainingRun.best_map50.desc().nullslast())
    elif sort_by == 'highest_map5095':
        query = query.order_by(TrainingRun.best_map5095.desc().nullslast())
    else:
        query = query.order_by(TrainingRun.created_at.desc())

    runs = query.all()
    for r in runs:
        if r.status in ['RUNNING', 'STARTING']:
            sync_run_state(r)

    best_run = find_best_training_run(project_id=active_pid, runs_list=runs)
    best_run_id = best_run.id if best_run else None

    return render_template(
        'training/history.html', 
        runs=runs, 
        current_status=status_filter, 
        current_sort=sort_by,
        best_run=best_run,
        best_run_id=best_run_id
    )

@training_bp.route('/<run_id>/stop', methods=['POST'])
def stop(run_id):
    success, msg = stop_training_run(run_id)
    return jsonify({"success": success, "message": msg})

@training_bp.route('/<run_id>/status', methods=['GET'])
def get_status(run_id):
    run_record = TrainingRun.query.get_or_404(run_id)
    sync_run_state(run_record)
    return jsonify(run_record.to_dict())

@training_bp.route('/<run_id>/metrics', methods=['GET'])
def get_metrics(run_id):
    run_record = TrainingRun.query.get_or_404(run_id)
    sync_run_state(run_record)

    output_dir = run_record.output_path or os.path.join(get_run_dir(run_id), "outputs", "train")
    metrics_data = parse_results_csv(output_dir)

    # Sinkronisasi best metrics ke DB jika ada pembaruan
    best_info = metrics_data.get("best", {})
    if best_info and best_info.get("map50") is not None:
        new_m50 = best_info.get("map50")
        new_m5095 = best_info.get("map5095")
        if run_record.best_map50 != new_m50 or run_record.best_map5095 != new_m5095:
            run_record.best_map50 = new_m50
            run_record.best_map5095 = new_m5095
            db.session.commit()

    return jsonify({
        "status": run_record.status,
        "current_epoch": run_record.current_epoch,
        "total_epochs": run_record.total_epochs,
        "metrics": metrics_data
    })

@training_bp.route('/<run_id>/logs/stream')
def stream_logs(run_id):
    from flask import current_app
    app = current_app._get_current_object()
    return Response(generate_sse_logs(app, run_id), mimetype='text/event-stream')

@training_bp.route('/<run_id>/delete', methods=['POST'])
def delete_run(run_id):
    import os
    import shutil
    run_record = TrainingRun.query.get_or_404(run_id)

    # Jika proses masih berjalan, hentikan dulu
    if run_record.status in ['RUNNING', 'STARTING']:
        stop_training_run(run_id)

    run_dir = get_run_dir(run_id)
    if os.path.exists(run_dir):
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            pass

    db.session.delete(run_record)
    db.session.commit()

    flash(f"Riwayat training '{run_id}' berhasil dihapus.", "info")
    next_url = request.referrer or url_for('training.history')
    return redirect(next_url)

@training_bp.route('/<run_id>/toggle_best', methods=['POST'])
def toggle_best_run(run_id):
    """
    Menandai atau mencabut tanda Hasil Training Terbaik secara MANUAL oleh user.
    """
    run_record = TrainingRun.query.get_or_404(run_id)
    new_state = not bool(run_record.is_best)
    
    if new_state:
        # Jika dijadikan terbaik: unflag run lain pada project yang sama agar rapi
        other_runs = TrainingRun.query.filter_by(project_id=run_record.project_id).all()
        for r in other_runs:
            r.is_best = False
        run_record.is_best = True
        msg = f"Run '{run_id}' berhasil ditandai sebagai Hasil Training Terbaik."
    else:
        run_record.is_best = False
        msg = f"Tanda Hasil Training Terbaik untuk '{run_id}' telah dicabut."

    db.session.commit()

    if request.is_json:
        return jsonify({"success": True, "is_best": run_record.is_best, "message": msg})

    flash(msg, "success")
    next_url = request.referrer or url_for('training.history')
    return redirect(next_url)

@training_bp.route('/<run_id>/rename_run_id', methods=['POST'])
@training_bp.route('/<run_id>/rename', methods=['POST'])
def rename_training_run_id(run_id):
    """
    Mengubah / mengedit Run ID hasil training (bukan nama model pretrained).
    Mengganti primary key di database dan memindahkan folder output di penyimpanan.
    """
    run_record = TrainingRun.query.get_or_404(run_id)
    data = request.get_json(silent=True) or request.form.to_dict()
    new_id = (data.get('new_run_id') or data.get('run_id') or data.get('model_name') or '').strip()

    if not new_id:
        if request.is_json:
            return jsonify({"success": False, "error": "Run ID tidak boleh kosong"}), 400
        flash("Run ID tidak boleh kosong.", "error")
        return redirect(request.referrer or url_for('training.history'))

    # Validasi karakter: huruf, angka, underscore, strip
    if not re.match(r'^[a-zA-Z0-9_\-]+$', new_id):
        err = "Run ID hanya boleh berisi huruf, angka, garis bawah (_), dan tanda hubung (-)."
        if request.is_json:
            return jsonify({"success": False, "error": err}), 400
        flash(err, "error")
        return redirect(request.referrer or url_for('training.history'))

    if len(new_id) < 3 or len(new_id) > 64:
        err = "Panjang Run ID harus antara 3 hingga 64 karakter."
        if request.is_json:
            return jsonify({"success": False, "error": err}), 400
        flash(err, "error")
        return redirect(request.referrer or url_for('training.history'))

    if new_id == run_id:
        if request.is_json:
            return jsonify({"success": True, "old_id": run_id, "new_id": new_id, "message": "Run ID sama, tidak ada perubahan."})
        return redirect(request.referrer or url_for('training.history'))

    # Cek apakah ID baru sudah digunakan di database
    existing = TrainingRun.query.get(new_id)
    if existing:
        err = f"Run ID '{new_id}' sudah digunakan oleh riwayat training lain."
        if request.is_json:
            return jsonify({"success": False, "error": err}), 400
        flash(err, "error")
        return redirect(request.referrer or url_for('training.history'))

    # Rename direktori pada penyimpanan jika folder fisik ada
    old_dir = get_run_dir(run_id)
    new_dir = get_run_dir(new_id)
    if os.path.exists(old_dir):
        if os.path.exists(new_dir):
            err = f"Folder tujuan '{new_dir}' sudah ada di penyimpanan."
            if request.is_json:
                return jsonify({"success": False, "error": err}), 400
            flash(err, "error")
            return redirect(request.referrer or url_for('training.history'))
        try:
            os.rename(old_dir, new_dir)
        except Exception as e:
            err = f"Gagal mengubah nama folder run di disk: {str(e)}"
            if request.is_json:
                return jsonify({"success": False, "error": err}), 500
            flash(err, "error")
            return redirect(request.referrer or url_for('training.history'))

    # Update output_path string jika mengandung old run_id
    new_output_path = run_record.output_path.replace(run_id, new_id) if run_record.output_path else None

    # Update database via direct SQL (karena primary key id di-update)
    db.session.execute(
        db.text("UPDATE training_runs SET id = :new_id, output_path = :new_path WHERE id = :old_id"),
        {"new_id": new_id, "new_path": new_output_path, "old_id": run_id}
    )
    db.session.commit()

    msg = f"Run ID '{run_id}' berhasil diubah menjadi '{new_id}'."
    if request.is_json:
        return jsonify({
            "success": True,
            "old_id": run_id,
            "new_id": new_id,
            "redirect_url": url_for('results.detail', run_id=new_id),
            "message": msg
        })

    flash(msg, "success")
    return redirect(request.referrer or url_for('training.history'))
