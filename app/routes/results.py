import os
from flask import Blueprint, render_template, send_file, abort, jsonify
from ..models.training_run import TrainingRun
from ..services.training_service import get_run_dir, sync_run_state
from ..services.result_service import (
    parse_results_csv, discover_result_artifacts, get_weight_paths,
    find_best_training_run
)

results_bp = Blueprint('results', __name__)

@results_bp.route('/<run_id>')
def detail(run_id):
    run_record = TrainingRun.query.get_or_404(run_id)
    sync_run_state(run_record)

    run_dir = get_run_dir(run_id)
    output_dir = run_record.output_path or os.path.join(run_dir, "outputs", "train")

    metrics_data = parse_results_csv(output_dir)
    best_info = metrics_data.get("best", {})
    if best_info and best_info.get("map50") is not None:
        new_m50 = best_info.get("map50")
        new_m5095 = best_info.get("map5095")
        if run_record.best_map50 != new_m50 or run_record.best_map5095 != new_m5095:
            from ..models import db
            run_record.best_map50 = new_m50
            run_record.best_map5095 = new_m5095
            db.session.commit()

    artifacts = discover_result_artifacts(output_dir)
    weights = get_weight_paths(output_dir)
    import json
    epochs_json = json.dumps(metrics_data.get("epochs", []))

    # Cek apakah run ini adalah best run
    best_run = find_best_training_run(project_id=run_record.project_id)
    is_best_run = bool(best_run and best_run.id == run_record.id)

    return render_template(
        'results/detail.html',
        run=run_record,
        metrics=metrics_data,
        epochs_json=epochs_json,
        artifacts=artifacts,
        weights=weights,
        is_best_run=is_best_run,
        best_run=best_run
    )

@results_bp.route('/<run_id>/file/<path:filename>')
def serve_output_file(run_id, filename):
    run_record = TrainingRun.query.get_or_404(run_id)
    output_dir = os.path.abspath(run_record.output_path or os.path.join(get_run_dir(run_id), "outputs", "train"))

    target_file = os.path.abspath(os.path.join(output_dir, filename))
    # Proteksi path traversal
    if not target_file.startswith(output_dir):
        abort(403)

    if not os.path.exists(target_file):
        abort(404)

    return send_file(target_file)

@results_bp.route('/<run_id>/weights/<weight_type>')
def download_weight(run_id, weight_type):
    run_record = TrainingRun.query.get_or_404(run_id)
    output_dir = run_record.output_path or os.path.join(get_run_dir(run_id), "outputs", "train")

    if weight_type not in ['best', 'last']:
        abort(400)

    weight_file = os.path.join(output_dir, "weights", f"{weight_type}.pt")
    if not os.path.exists(weight_file):
        abort(404)

    download_name = f"{run_record.model_name}_{weight_type}_{run_id}.pt"
    return send_file(weight_file, as_attachment=True, download_name=download_name)
