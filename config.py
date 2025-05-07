import os

class Config:
    CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
    SECRET_KEY=  os.environ.get('SECRET_KEY','')
    
    # AI Model Selection - Choose "claude" or "gemini"
    ACTIVE_MODEL = "gemini"
    
    # Claude API settings
    CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
    CLAUDE_MODEL = "claude-3-opus-20240229"  # Updated to a valid model identifier
    
    # Google Gemini API settings
    GEMINI_MODEL = 'gemini-1.5-flash'#"gemini-2.0-pro-exp"  # Or your preferred Gemini model version
    
    GORQ_API_URL="https://api.gorq.com/v1/ask"
    # GORQ Model Configuration
    GORQ_MODEL = "gorq-2.0-advanced"  # Specify the GORQ model version

    # Session Configuration
    SESSION_TYPE = "filesystem"  # or "redis" if you have Redis
    SESSION_FILE_DIR = "/tmp/flask_session"  # for filesystem storage
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_COOKIE_MAX_SIZE = 4093
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    
    # File Upload Configuration
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {"txt", "pdf", "csv", "docx", "xlsx", "xsls", "data"}
    
    # Database Configuration
    DATABASE_DIR = "databases"
    
    # User account settings
    PASSWORD_MIN_LENGTH = 8
    SESSION_EXPIRY_DAYS = 30  # For "remember me" option

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config_by_name = dict(
    dev=DevelopmentConfig,
    prod=ProductionConfig
)

 

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