import os
import shutil
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from ..models import db
from ..models.project import Project
from ..models.dataset import Dataset
from ..models.training_run import TrainingRun

projects_bp = Blueprint('projects', __name__)

@projects_bp.route('/')
def index():
    projects = Project.query.order_by(Project.created_at.desc()).all()
    project_cards = []
    
    for p in projects:
        datasets = Dataset.query.filter_by(project_id=p.id).all()
        runs = TrainingRun.query.filter_by(project_id=p.id).all()
        completed_runs = [r for r in runs if r.status == 'COMPLETED']
        best_run = None
        if completed_runs:
            # Cari run dengan mAP50 tertinggi
            valid_map_runs = [r for r in completed_runs if r.best_map50 is not None]
            if valid_map_runs:
                best_run = max(valid_map_runs, key=lambda x: x.best_map50)
        
        project_cards.append({
            "project": p,
            "dataset_count": len(datasets),
            "run_count": len(runs),
            "completed_run_count": len(completed_runs),
            "best_map50": best_run.best_map50 if best_run else None,
            "best_run_id": best_run.id if best_run else None
        })

    return render_template('projects/index.html', project_cards=project_cards)

@projects_bp.route('/create', methods=['POST'])
def create_project():
    name = request.form.get('name', '').strip()
    description = request.form.get('description', '').strip()

    if not name:
        flash("Nama project wajib diisi.", "danger")
        return redirect(url_for('projects.index'))

    proj_id = f"proj_{uuid.uuid4().hex[:8]}"
    new_proj = Project(
        id=proj_id,
        name=name,
        description=description
    )

    db.session.add(new_proj)
    db.session.commit()

    # Otomatis aktifkan project yang baru dibuat
    session['active_project_id'] = proj_id
    flash(f"Project '{name}' berhasil dibuat dan diaktifkan.", "success")
    return redirect(url_for('projects.index'))

@projects_bp.route('/set-active/<project_id>', methods=['POST', 'GET'])
def set_active(project_id):
    if project_id == 'all':
        session['active_project_id'] = 'all'
        flash("Menampilkan semua project (mode global).", "info")
    else:
        proj = Project.query.get_or_404(project_id)
        session['active_project_id'] = proj.id
        flash(f"Project aktif beralih ke '{proj.name}'.", "info")

    next_url = request.args.get('next') or request.referrer or url_for('dashboard.index')
    return redirect(next_url)

@projects_bp.route('/<project_id>/edit', methods=['POST'])
def edit_project(project_id):
    proj = Project.query.get_or_404(project_id)
    name = request.form.get('name', '').strip()
    description = request.form.get('description', '').strip()

    if not name:
        flash("Nama project tidak boleh kosong.", "danger")
        return redirect(url_for('projects.index'))

    proj.name = name
    proj.description = description
    db.session.commit()

    flash(f"Project '{name}' berhasil diperbarui.", "success")
    return redirect(url_for('projects.index'))

@projects_bp.route('/<project_id>/delete', methods=['POST'])
def delete_project(project_id):
    proj = Project.query.get_or_404(project_id)
    name = proj.name

    # Hapus file fisik dataset milik project
    for ds in proj.datasets:
        if ds.storage_path and os.path.exists(ds.storage_path):
            shutil.rmtree(ds.storage_path, ignore_errors=True)

    # Hapus file fisik runs milik project
    for run in proj.runs:
        run_folder = os.path.dirname(run.output_path) if run.output_path else None
        if run_folder and os.path.exists(run_folder):
            shutil.rmtree(run_folder, ignore_errors=True)

    # Jika project yang dihapus adalah project aktif, reset ke project lain atau 'all'
    if session.get('active_project_id') == proj.id:
        remaining = Project.query.filter(Project.id != proj.id).first()
        session['active_project_id'] = remaining.id if remaining else 'all'

    db.session.delete(proj)
    db.session.commit()

    flash(f"Project '{name}' dan data di dalamnya berhasil dihapus.", "success")
    return redirect(url_for('projects.index'))
