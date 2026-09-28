import os
import sqlite3
import psutil
from flask import Flask, session
from .models import db
from .models.project import Project
from .models.dataset import Dataset
from .models.training_run import TrainingRun
from config import Config

def reconcile_stale_runs(app):
    """
    PRD Section 58: Training Recovery saat Flask start.
    Cek run berstatus RUNNING atau STARTING. Jika PID tidak aktif, ubah ke INTERRUPTED.
    """
    with app.app_context():
        try:
            stale_runs = TrainingRun.query.filter(TrainingRun.status.in_(['RUNNING', 'STARTING', 'QUEUED'])).all()
            for run in stale_runs:
                is_alive = False
                if run.pid:
                    try:
                        p = psutil.Process(run.pid)
                        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                            is_alive = True
                    except Exception:
                        is_alive = False
                
                if not is_alive:
                    run.status = 'INTERRUPTED'
                    run.error_message = 'Training terhenti karena server di-restart atau proses terputus.'
            db.session.commit()
        except Exception as e:
            print(f"[Reconcile Notice] {e}")

def migrate_and_init_projects(app):
    """
    Memastikan tabel projects ada, kolom project_id pada datasets dan training_runs
    dimigrasikan secara aman pada SQLite, serta menginisialisasi default project.
    """
    with app.app_context():
        try:
            # 1. Pastikan tabel projects dibuat
            db.create_all()

            # 2. Periksa dan tambahkan kolom project_id pada SQLite jika belum ada
            db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
            if 'sqlite:///' in db_uri:
                db_path = db_uri.replace('sqlite:///', '')
                if os.path.exists(db_path):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()

                    cursor.execute("PRAGMA table_info(datasets);")
                    ds_cols = [c[1] for c in cursor.fetchall()]
                    if 'project_id' not in ds_cols:
                        cursor.execute("ALTER TABLE datasets ADD COLUMN project_id VARCHAR(64);")

                    cursor.execute("PRAGMA table_info(training_runs);")
                    run_cols = [c[1] for c in cursor.fetchall()]
                    if 'project_id' not in run_cols:
                        cursor.execute("ALTER TABLE training_runs ADD COLUMN project_id VARCHAR(64);")

                    conn.commit()
                    conn.close()

            # 3. Migrasikan dataset dan run lama yang ber-project_id NULL ke project pertama jika ada
            first_proj = Project.query.first()
            if first_proj:
                Dataset.query.filter((Dataset.project_id == None) | (Dataset.project_id == '')).update({'project_id': first_proj.id})
                TrainingRun.query.filter((TrainingRun.project_id == None) | (TrainingRun.project_id == '')).update({'project_id': first_proj.id})
                db.session.commit()
        except Exception as e:
            print(f"[Project Init Error] {e}")

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Pastikan konfigurasi upload multipart Werkzeug/Flask tanpa batasan parts / memori
    app.config['MAX_FORM_PARTS'] = getattr(config_class, 'MAX_FORM_PARTS', None)
    app.config['MAX_FORM_MEMORY_SIZE'] = getattr(config_class, 'MAX_FORM_MEMORY_SIZE', None)

    # Inisialisasi Database
    db.init_app(app)

    # Global 413 Handler (Request Entity Too Large)
    @app.errorhandler(413)
    def request_entity_too_large(error):
        from flask import request, flash, redirect, jsonify
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({
                "success": False,
                "error": "Ukuran data atau jumlah file melebihi kapasitas yang diizinkan."
            }), 413
        flash("Ukuran data atau jumlah file yang diunggah melebihi batas sistem.", "error")
        return redirect(request.referrer or '/')

    # Daftarkan Blueprints
    from .routes.dashboard import dashboard_bp
    from .routes.datasets import datasets_bp
    from .routes.training import training_bp
    from .routes.results import results_bp
    from .routes.system import system_bp
    from .routes.testing import testing_bp
    from .routes.projects import projects_bp
    from .routes.labeling import labeling_bp
    from .routes.augmentation import augmentation_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(datasets_bp, url_prefix='/datasets')
    app.register_blueprint(labeling_bp)
    app.register_blueprint(augmentation_bp, url_prefix='/augmentation')
    app.register_blueprint(training_bp, url_prefix='/training')
    app.register_blueprint(results_bp, url_prefix='/results')
    app.register_blueprint(system_bp, url_prefix='/system')
    app.register_blueprint(testing_bp, url_prefix='/testing')
    app.register_blueprint(projects_bp, url_prefix='/projects')

    # Context Processor untuk Project Switcher Global
    @app.context_processor
    def inject_project_context():
        try:
            all_projs = Project.query.order_by(Project.created_at.asc()).all()
            active_pid = session.get('active_project_id')
            active_proj = None
            if active_pid and active_pid != 'all':
                active_proj = Project.query.get(active_pid)
            return {
                'all_projects': all_projs,
                'active_project': active_proj,
                'active_project_id': active_pid or 'all'
            }
        except Exception:
            return {
                'all_projects': [],
                'active_project': None,
                'active_project_id': 'all'
            }

    # Inisialisasi DB Tables, Migrasi Project & Startup Reconcile
    migrate_and_init_projects(app)
    reconcile_stale_runs(app)

    return app
