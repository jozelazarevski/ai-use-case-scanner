from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

db = SQLAlchemy()

class User(db.Model, UserMixin):
    """User model for storing user account information"""
    __tablename__ = 'users'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    first_name = db.Column(db.String(64), nullable=False)
    last_name = db.Column(db.String(64), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    organization = db.Column(db.String(120), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    projects = db.relationship('Project', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    sessions = db.relationship('UserSession', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        """Set user password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches stored hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_active_session(self):
        """Get the user's active session if any"""
        return UserSession.query.filter(
            UserSession.user_id == self.id, 
            UserSession.is_active == True,
            UserSession.expires_at > datetime.utcnow()
        ).first()
    
    def __repr__(self):
        return f'<User {self.email}>'


class UserSession(db.Model):
    """User session model for tracking login sessions"""
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    session_token = db.Column(db.String(36), nullable=False, unique=True)
    ip_address = db.Column(db.String(45), nullable=True)  # IPv6 can be up to 45 chars
    user_agent = db.Column(db.String(255), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    
    def is_expired(self):
        """Check if the session is expired"""
        return datetime.utcnow() > self.expires_at
    
    def __repr__(self):
        return f'<UserSession {self.id} for user {self.user_id}>'


class Project(db.Model):
    """Project model for storing user projects"""
    __tablename__ = 'projects'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    datasets = db.relationship('Dataset', backref='project', lazy='dynamic', cascade='all, delete-orphan')
    use_cases = db.relationship('UseCase', backref='project', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Project {self.title}>'


class Dataset(db.Model):
    """Dataset model for storing uploaded files"""
    __tablename__ = 'datasets'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = db.Column(db.String(36), db.ForeignKey('projects.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_size = db.Column(db.Integer, nullable=True)  # Size in bytes
    file_type = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Dataset {self.filename}>'


class UseCase(db.Model):
    """Use case model for storing AI use case proposals"""
    __tablename__ = 'use_cases'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = db.Column(db.String(36), db.ForeignKey('projects.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    target_variable = db.Column(db.String(128), nullable=True)
    model_type = db.Column(db.String(64), nullable=True)
    business_value = db.Column(db.Text, nullable=True)
    implementation_complexity = db.Column(db.String(32), nullable=True)
    prediction_interpretation = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    kpis = db.relationship('KPI', backref='use_case', lazy='dynamic', cascade='all, delete-orphan')
    models = db.relationship('Model', backref='use_case', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<UseCase {self.title}>'


class KPI(db.Model):
    """KPI model for storing key performance indicators for use cases"""
    __tablename__ = 'kpis'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    use_case_id = db.Column(db.String(36), db.ForeignKey('use_cases.id'), nullable=False)
    description = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<KPI {self.id} for use case {self.use_case_id}>'


class Model(db.Model):
    """Model for storing trained machine learning models"""
    __tablename__ = 'models'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    use_case_id = db.Column(db.String(36), db.ForeignKey('use_cases.id'), nullable=False)
    model_type = db.Column(db.String(64), nullable=False)
    model_file_path = db.Column(db.String(512), nullable=True)
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    training_time = db.Column(db.Float, nullable=True)  # in seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    feature_importances = db.relationship('FeatureImportance', backref='model', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Model {self.id} for use case {self.use_case_id}>'


class FeatureImportance(db.Model):
    """Feature importance for trained models"""
    __tablename__ = 'feature_importances'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = db.Column(db.String(36), db.ForeignKey('models.id'), nullable=False)
    feature_name = db.Column(db.String(128), nullable=False)
    importance_value = db.Column(db.Float, nullable=False)
    
    def __repr__(self):
        return f'<FeatureImportance {self.feature_name}: {self.importance_value}>'
