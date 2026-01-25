"""
Trademify Configuration Settings
การตั้งค่าระบบ Pattern Recognition
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "faiss_indices"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Settings
class DataConfig:
    # Timeframes to analyze
    TIMEFRAMES = ["M5", "M15", "H1"]
    
    # Window size for pattern matching (จำนวนแท่งเทียนในแต่ละ pattern)
    WINDOW_SIZE = 60
    
    
    # Future candles to analyze for voting (แท่งเทียนถัดไปที่จะดูผลลัพธ์)
    FUTURE_CANDLES = 10
    
    # Years of historical data
    HISTORY_YEARS = 5
    
    # 🥇 GOLD ONLY - Best performance in backtest (88.7% win rate)
    # Forex pairs disabled due to poor backtest results
    DEFAULT_SYMBOLS = [
        "XAUUSDm"  # Gold - BEST PERFORMER!
        # "EURUSDm", # ❌ Disabled - 48.1% win rate
        # "GBPUSDm", # ❌ Disabled - 44.4% win rate
    ]

# Pattern Matching Settings
class PatternConfig:
    # Number of similar patterns to find
    TOP_K_PATTERNS = 10
    
    # Minimum correlation threshold (ต่ำกว่านี้ไม่เทรด)
    MIN_CORRELATION = 0.85
    
    # Confidence threshold for trading signal
    CONFIDENCE_THRESHOLD = 80  # percent
    
    # FAISS index type
    FAISS_INDEX_TYPE = "IVF"  # Options: "Flat", "IVF", "HNSW"
    
    # Number of clusters for IVF
    N_CLUSTERS = 100
    
    # Number of probes during search
    N_PROBES = 10

# Normalization Settings
class NormConfig:
    # Normalization method: "zscore", "minmax", "log_return"
    METHOD = "zscore"
    
    # Rolling window for Z-score calculation
    ZSCORE_WINDOW = 20

# Voting System Settings
class VotingConfig:
    # Minimum votes required for signal
    MIN_CONFIDENCE = 70  # percent
    
    # Strong signal threshold
    STRONG_SIGNAL = 80  # percent
    
    # Time filters (session-based analysis)
    SESSIONS = {
        "ASIA": {"start": "00:00", "end": "08:00"},
        "LONDON": {"start": "08:00", "end": "16:00"},
        "NEW_YORK": {"start": "13:00", "end": "22:00"},
    }

# Firebase Settings
class FirebaseConfig:
    CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "config/firebase_credentials.json")
    DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "")
    PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")

# API Settings
class APIConfig:
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    # Allow all origins for VPS deployment - add specific origins via CORS_ORIGINS env var
    # Use "*" to allow any origin (useful for development/VPS with varying hostnames)
    _default_origins = "*"
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", _default_origins).split(",")
    # If "*" is in the list, we'll use allow_origin_regex in FastAPI instead
    ALLOW_ALL_ORIGINS = "*" in _default_origins or os.getenv("CORS_ALLOW_ALL", "true").lower() == "true"

# Risk Management Settings
class RiskConfig:
    # Stop loss calculation method
    SL_METHOD = "historical_max"  # Max adverse movement from historical patterns
    
    # Take profit ratio (risk:reward)
    TP_RATIO = 2.0
    
    # Maximum allowed drawdown from patterns (%)
    MAX_DRAWDOWN_THRESHOLD = 3.0
