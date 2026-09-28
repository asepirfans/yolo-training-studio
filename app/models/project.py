from datetime import datetime
from . import db

class Project(db.Model):
    __tablename__ = 'projects'

    id = db.Column(db.String(64), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships: One Project has many Datasets and many TrainingRuns
    datasets = db.relationship('Dataset', backref='project', lazy=True, cascade="all, delete-orphan")
    runs = db.relationship('TrainingRun', backref='project', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description or "",
            "dataset_count": len(self.datasets),
            "run_count": len(self.runs),
            "created_at": self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }
