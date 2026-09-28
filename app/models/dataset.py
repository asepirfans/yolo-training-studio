from datetime import datetime
import json
from . import db

class Dataset(db.Model):
    __tablename__ = 'datasets'

    id = db.Column(db.String(64), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    upload_filename = db.Column(db.String(255), nullable=False)
    storage_path = db.Column(db.String(512), nullable=False)
    yaml_path = db.Column(db.String(512), nullable=True)
    num_classes = db.Column(db.Integer, default=0)
    class_names_json = db.Column(db.Text, default='{}')
    train_count = db.Column(db.Integer, default=0)
    val_count = db.Column(db.Integer, default=0)
    test_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    project_id = db.Column(db.String(64), db.ForeignKey('projects.id'), nullable=True)

    # Relationship
    runs = db.relationship('TrainingRun', backref='dataset', lazy=True, cascade="all, delete-orphan")

    def get_class_names(self):
        try:
            return json.loads(self.class_names_json or '{}')
        except Exception:
            return {}

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "upload_filename": self.upload_filename,
            "storage_path": self.storage_path,
            "yaml_path": self.yaml_path,
            "num_classes": self.num_classes,
            "class_names": self.get_class_names(),
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "project_id": self.project_id,
            "project_name": self.project.name if self.project else "Tanpa Project",
            "created_at": self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }
