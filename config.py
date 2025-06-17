# -*- coding: utf-8 -*-
"""
Configuration module for the AI Use Case Generator application.
Handles environment variables, application settings, and encodings.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Base configuration class with common settings"""
    
    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24).hex())
    
    # AI Model configuration
    CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
    
    # AI Model Selection - Choose "claude" or "gemini"
    ACTIVE_MODEL = os.environ.get('ACTIVE_MODEL', "gemini")
    
    # Claude API settings
    CLAUDE_API_URL = os.environ.get('CLAUDE_API_URL', "https://api.anthropic.com/v1/messages")
    CLAUDE_MODEL = os.environ.get('CLAUDE_MODEL', "claude-3-opus-20240229")
    
    # Google Gemini API settings
    GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
    
    # GORQ API settings (if used)
    GORQ_API_URL = os.environ.get('GORQ_API_URL', "https://api.gorq.com/v1/ask")
    GORQ_MODEL = os.environ.get('GORQ_MODEL', "gorq-2.0-advanced")

    # Session Configuration
    SESSION_TYPE = os.environ.get('SESSION_TYPE', "filesystem")
    SESSION_FILE_DIR = os.environ.get('SESSION_FILE_DIR', "/tmp/flask_session")
    SESSION_PERMANENT = os.environ.get('SESSION_PERMANENT', 'False') == 'True'
    SESSION_USE_SIGNER = True
    SESSION_COOKIE_MAX_SIZE = 4093
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False') == 'True'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', "uploads")
    ALLOWED_EXTENSIONS = {"txt", "pdf", "csv", "docx", "xlsx", "xls", "data"}
    
    # Database Configuration
    DATABASE_DIR = os.environ.get('DATABASE_DIR', "databases")
    DB_NAME = os.environ.get('DB_NAME', 'AIUseCase')
    DB_USER = os.environ.get('DB_USER', 'postgres')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', 'root')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', '5432')
    
    # User account settings
    PASSWORD_MIN_LENGTH = int(os.environ.get('PASSWORD_MIN_LENGTH', '8'))
    SESSION_EXPIRY_DAYS = int(os.environ.get('SESSION_EXPIRY_DAYS', '30'))
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', '')  # Empty string for console logging
    
    @classmethod
    def setup_logging(cls):
        """Configure logging based on settings"""
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        
        # Basic configuration
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Configure file handler if LOG_FILE is specified
        if cls.LOG_FILE:
            file_handler = logging.FileHandler(cls.LOG_FILE)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            
            # Add file handler to root logger
            logging.getLogger('').addHandler(file_handler)
        
        # Set lower log level for noisy libraries
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        logging.getLogger('flask_session').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        
        logger = logging.getLogger(__name__)
        logger.debug("Logging configured")


class DevelopmentConfig(Config):
    """Development environment specific configuration"""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False
    
    # Additional development settings
    EXPLAIN_TEMPLATE_LOADING = True  # For Flask debug


class TestingConfig(Config):
    """Testing environment specific configuration"""
    DEBUG = False
    TESTING = True
    
    # Use in-memory SQLite for testing
    DB_NAME = 'test_aiusecase'
    SESSION_COOKIE_SECURE = False
    
    # Override AI keys with test values
    CLAUDE_API_KEY = 'test_claude_key'
    GOOGLE_API_KEY = 'test_google_key'
    
    # Use temporary directories for uploads and sessions
    UPLOAD_FOLDER = "/tmp/test_uploads"
    SESSION_FILE_DIR = "/tmp/test_sessions"
    DATABASE_DIR = "/tmp/test_databases"


class ProductionConfig(Config):
    """Production environment specific configuration"""
    DEBUG = False
    TESTING = False
    
    # Enable secure cookies in production
    SESSION_COOKIE_SECURE = True
    
    # Force HTTPS
    PREFERRED_URL_SCHEME = 'https'


# Configuration mapping
config_by_name = {
    'dev': DevelopmentConfig,
    'test': TestingConfig,
    'prod': ProductionConfig
}

# Get active configuration based on environment variable or default to development
active_config = config_by_name.get(
    os.environ.get('FLASK_ENV', 'dev').lower(), 
    DevelopmentConfig
)

# File encoding options for flexibility in reading various files
encodings = [
    # Unicode encodings
    'utf-8',        # The most common Unicode encoding
    'utf-16',       # Unicode with 16-bit code units (with BOM)
    'utf-16-le',    # Little-endian UTF-16
    'utf-16-be',    # Big-endian UTF-16
    'utf-32',       # Unicode with 32-bit code units (with BOM)
    'utf-32-le',    # Little-endian UTF-32
    'utf-32-be',    # Big-endian UTF-32
    
    # Western European encodings
    'latin-1',      # ISO-8859-1, Western European
    'iso8859-1',    # Alias for latin-1
    'cp1252',       # Windows-1252, Western European (superset of latin-1)
    'iso8859-15',   # Latin-9, Western European with Euro symbol
    
    # Central/Eastern European encodings
    'iso8859-2',    # ISO-8859-2, Central European
    'cp1250',       # Windows-1250, Central European
    
    # Cyrillic encodings
    'iso8859-5',    # ISO-8859-5, Cyrillic
    'cp1251',       # Windows-1251, Cyrillic
    'koi8-r',       # KOI8-R, Russian Cyrillic
    'koi8-u',       # KOI8-U, Ukrainian Cyrillic
    
    # Greek encodings
    'iso8859-7',    # ISO-8859-7, Greek
    'cp1253',       # Windows-1253, Greek
    
    # Turkish encodings
    'iso8859-9',    # ISO-8859-9, Turkish
    'cp1254',       # Windows-1254, Turkish
    
    # Hebrew encodings
    'iso8859-8',    # ISO-8859-8, Hebrew
    'cp1255',       # Windows-1255, Hebrew
    
    # Arabic encodings
    'iso8859-6',    # ISO-8859-6, Arabic
    'cp1256',       # Windows-1256, Arabic
    
    # Baltic encodings
    'iso8859-4',    # ISO-8859-4, Baltic
    'cp1257',       # Windows-1257, Baltic
    
    # Vietnamese encoding
    'cp1258',       # Windows-1258, Vietnamese
    
    # Japanese encodings
    'shift-jis',    # Shift-JIS, Japanese
    'cp932',        # Windows Japanese
    'euc-jp',       # Extended Unix Code for Japanese
    
    # Chinese encodings
    'gb2312',       # GB2312, Simplified Chinese
    'gbk',          # GBK, Simplified Chinese extension
    'gb18030',      # GB18030, Chinese standard
    'big5',         # Big5, Traditional Chinese
    'big5hkscs',    # Big5-HKSCS, Traditional Chinese with Hong Kong extensions
    
    # Korean encodings
    'euc-kr',       # Extended Unix Code for Korean
    'cp949',        # Windows Korean
    
    # Thai encoding
    'cp874',        # Windows Thai
    
    # IBM and Mac legacy encodings
    'cp437',        # Original IBM PC encoding
    'cp850',        # DOS Western European
    'cp852',        # DOS Central European
    'cp855',        # DOS Cyrillic
    'cp866',        # DOS Russian
    'mac-roman',    # Apple MacOS Roman
    
    # Misc
    'ascii',        # 7-bit ASCII
    'idna',         # International Domain Names in Applications
    'palmos',       # PalmOS encoding
    'punycode'      # ASCII encoding of Unicode for network protocols
]