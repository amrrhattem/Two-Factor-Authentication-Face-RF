import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'auth_system.db')
RF_MODEL_PATH = os.path.join(BASE_DIR, 'rf_model_15features.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder_15features.pkl')
ALL_FEATURES_PATH = os.path.join(BASE_DIR, 'Features', 'all_features.pkl')

# --- Security ---
ADMIN_TOKEN = os.environ.get('ADMIN_TOKEN', 'change-me-in-production')

# --- Face Recognition ---
FACE_SIMILARITY_THRESHOLD = 0.60
FACE_MODEL_NAME = 'buffalo_l'
FACE_DET_SIZE = (640, 640)

# --- Server ---
DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
PORT = int(os.environ.get('PORT', 5000))