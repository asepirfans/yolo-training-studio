from datetime import datetime
import json
from . import db

class TrainingRun(db.Model):
    __tablename__ = 'training_runs'

    id = db.Column(db.String(64), primary_key=True)
    dataset_id = db.Column(db.String(64), db.ForeignKey('datasets.id'), nullable=False)
    model_name = db.Column(db.String(128), nullable=False)
    status = db.Column(db.String(32), default='QUEUED')  # QUEUED, STARTING, RUNNING, COMPLETED, FAILED, STOPPED, INTERRUPTED
    config_json = db.Column(db.Text, nullable=False, default='{}')
    pid = db.Column(db.Integer, nullable=True)
    current_epoch = db.Column(db.Integer, default=0)
    total_epochs = db.Column(db.Integer, default=100)
    best_map50 = db.Column(db.Float, nullable=True)
    best_map5095 = db.Column(db.Float, nullable=True)
    started_at = db.Column(db.DateTime, nullable=True)
    finished_at = db.Column(db.DateTime, nullable=True)
    duration_seconds = db.Column(db.Integer, default=0)
    output_path = db.Column(db.String(512), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    project_id = db.Column(db.String(64), db.ForeignKey('projects.id'), nullable=True)
    is_best = db.Column(db.Boolean, default=False, nullable=True)

    def get_config(self):
        try:
            return json.loads(self.config_json or '{}')
        except Exception:
            return {}

    def get_formatted_duration(self):
        if not self.duration_seconds:
            return "-"
        m, s = divmod(self.duration_seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset.name if self.dataset else "Unknown",
            "model_name": self.model_name,
            "status": self.status,
            "config": self.get_config(),
            "pid": self.pid,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "best_map50": self.best_map50,
            "best_map5095": self.best_map5095,
            "started_at": self.started_at.strftime('%Y-%m-%d %H:%M:%S') if self.started_at else None,
            "finished_at": self.finished_at.strftime('%Y-%m-%d %H:%M:%S') if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "formatted_duration": self.get_formatted_duration(),
            "output_path": self.output_path,
            "error_message": self.error_message,
            "project_id": self.project_id,
            "project_name": self.project.name if self.project else "Tanpa Project",
            "created_at": self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            "is_best": bool(self.is_best)
        }
