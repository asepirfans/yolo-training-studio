from flask import Blueprint, render_template, jsonify, session, redirect, url_for
from ..models.training_run import TrainingRun
from ..models.dataset import Dataset
from ..services.gpu_service import get_gpu_info, get_system_resources
from ..services.training_service import sync_run_state
from ..services.result_service import find_best_training_run

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
def index():
    active_pid = session.get('active_project_id')
    if not active_pid:
        return redirect(url_for('projects.index'))

    runs_query = TrainingRun.query
    ds_query = Dataset.query
    
    if active_pid and active_pid != 'all':
        runs_query = runs_query.filter_by(project_id=active_pid)
        ds_query = ds_query.filter_by(project_id=active_pid)

    recent_runs = runs_query.order_by(TrainingRun.created_at.desc()).limit(10).all()
    for r in recent_runs:
        if r.status in ['RUNNING', 'STARTING']:
            sync_run_state(r)

    best_run = find_best_training_run(project_id=active_pid)
    best_run_id = best_run.id if best_run else None

    active_run = TrainingRun.query.filter(TrainingRun.status.in_(['RUNNING', 'STARTING'])).first()
    total_datasets = ds_query.count()
    total_runs = runs_query.count()
    gpu_info = get_gpu_info()
    sys_resources = get_system_resources()

    return render_template(
        'dashboard.html',
        recent_runs=recent_runs,
        best_run=best_run,
        best_run_id=best_run_id,
        active_run=active_run,
        total_datasets=total_datasets,
        total_runs=total_runs,
        gpu=gpu_info,
        resources=sys_resources
    )
