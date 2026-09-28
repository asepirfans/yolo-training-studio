import uuid
from datetime import datetime
import json
from . import db

class LabelingSession(db.Model):
    __tablename__ = 'labeling_sessions'

    id = db.Column(db.String(64), primary_key=True, default=lambda: f"lbl_{uuid.uuid4().hex[:8]}")
    name = db.Column(db.String(255), nullable=False)
    project_id = db.Column(db.String(64), db.ForeignKey('projects.id'), nullable=True)
    classes_json = db.Column(db.Text, default='[]')
    storage_path = db.Column(db.String(512), nullable=False)
    crop_path = db.Column(db.String(512), nullable=True)
    image_count = db.Column(db.Integer, default=0)
    labeled_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship ke Project
    project = db.relationship('Project', backref=db.backref('labeling_sessions', lazy=True))

    def get_classes(self):
        try:
            return json.loads(self.classes_json or '[]')
        except Exception:
            return []

    def set_classes(self, classes_list):
        self.classes_json = json.dumps(classes_list)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "project_name": self.project.name if self.project else "Tanpa Project",
            "classes": self.get_classes(),
            "storage_path": self.storage_path,
            "crop_path": self.crop_path,
            "image_count": self.image_count,
            "labeled_count": self.labeled_count,
            "created_at": self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            "updated_at": self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }
