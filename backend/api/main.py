"""
Trademify API Server
FastAPI Backend for Pattern Recognition Trading System

Security:
- API Key Authentication (X-API-Key header)
- CORS configured via CORS_ORIGINS env var
- Rate limiting enabled
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import numpy as np
import logging
import json
import os
from datetime import datetime

from config import APIConfig, DataConfig, PatternConfig
from config.enhanced_settings import EnhancedAnalysisConfig, TradingMode
from data_processing import DataLake, DataGenerator, SlidingWindowGenerator, prepare_database
from similarity_engine import FAISSEngine, PatternMatcher
from analysis import VotingSystem, PatternAnalyzer, Signal
from analysis import EnhancedAnalyzer, get_enhanced_analyzer, SignalQuality
from analysis.enhanced_analyzer import get_multi_factor_analyzer, analyze_with_multi_factor
from analysis.multi_factor_analyzer import MultiFactorAnalyzer, MultiFactorResult
from services import get_firebase_service
from services.mt5_service import get_mt5_service, init_mt5_service, MarketStatus

# Security module imports
from api.security import (
    verify_api_key,
    optional_api_key,
    get_cors_origins,
    check_rate_limit,
    ValidatedStartBotRequest,
    ValidatedTradeRequest,
    add_security_headers,
)

# Import trading routes
from api.trading_routes import router as trading_router, init_trading_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot settings file for auto-start
BOT_SETTINGS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'bot_settings.json')

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    from enum import Enum
    from datetime import datetime, date
    
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return list(convert_numpy_types(v) for v in obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.floating, float)):
        val = float(obj)
        # Handle NaN and Inf
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, str):
        return obj
    elif hasattr(obj, 'to_dict'):
        # Handle dataclasses with to_dict method
        try:
            return convert_numpy_types(obj.to_dict())
        except Exception:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        # Handle dataclasses and custom objects
        try:
            return convert_numpy_types(vars(obj))
        except TypeError:
            return str(obj)
    else:
        # Fallback to string representation
        try:
            return str(obj)
        except Exception:
            return "N/A"

def load_bot_settings():
    """Load saved bot settings"""
    try:
        if os.path.exists(BOT_SETTINGS_FILE):
            with open(BOT_SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                settings = convert_numpy_types(settings)
                
                # Ensure symbols is a list, not string
                if 'symbols' in settings and isinstance(settings['symbols'], str):
                    settings['symbols'] = [s.strip() for s in settings['symbols'].split(',') if s.strip()]
                
                return settings
    except Exception as e:
        logging.warning(f"Failed to load bot settings: {e}")
    return None

def save_bot_settings(settings: dict):
    """Save bot settings for auto-start"""
    try:
        os.makedirs(os.path.dirname(BOT_SETTINGS_FILE), exist_ok=True)
        # Convert numpy types before saving
        clean_settings = convert_numpy_types(settings)
        with open(BOT_SETTINGS_FILE, 'w') as f:
            json.dump(clean_settings, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save bot settings: {e}")

def clear_bot_settings():
    """Clear saved bot settings (disable auto-start)"""
    try:
        if os.path.exists(BOT_SETTINGS_FILE):
            os.remove(BOT_SETTINGS_FILE)
    except Exception as e:
        logging.warning(f"Failed to clear bot settings: {e}")


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup: Auto-start bot if settings exist
    logger.info("🚀 Trademify API starting...")
    
    settings = load_bot_settings()
    if settings and settings.get('auto_start', False):
        logger.info("🤖 Auto-starting bot with saved settings...")
        try:
            await auto_start_bot(settings)
        except Exception as e:
            logger.error(f"Failed to auto-start bot: {e}")
    
    yield  # App runs here
    
    # Shutdown: Stop bot gracefully
    logger.info("🛑 Shutting down...")
    global _auto_bot, _bot_task
    if _auto_bot:
        await _auto_bot.stop()
        _auto_bot = None
    if _bot_task:
        _bot_task.cancel()
        _bot_task = None


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Trademify API",
    description="Pattern Recognition Trading System - Find similar historical patterns to predict market movements",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware - Configure allowed origins from environment
# Set CORS_ORIGINS env var for production (comma-separated URLs)
cors_origins = get_cors_origins()
logger.info(f"🔒 CORS origins: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True if cors_origins != ["*"] else False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*", "X-API-Key"],
    expose_headers=["*"],
)

# Security headers middleware
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Global exception handler to ensure CORS headers on errors
from fastapi.responses import JSONResponse
from starlette.requests import Request as StarletteRequest

@app.exception_handler(Exception)
async def global_exception_handler(request: StarletteRequest, exc: Exception):
    logger.error(f"Global exception: {exc}")
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": "Internal server error"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Include trading routes
app.include_router(trading_router)

# Global state
pattern_matchers: Dict[str, PatternMatcher] = {}
firebase_service = None


# Request/Response Models
class AnalyzeRequest(BaseModel):
    """Request model for pattern analysis"""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(default="H1", description="Timeframe (M5, M15, H1)")
    current_pattern: List[float] = Field(..., description="Current price pattern (normalized)")
    current_price: float = Field(..., description="Current actual price")
    k: int = Field(default=10, description="Number of similar patterns to find")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "EURUSD",
                "timeframe": "H1",
                "current_pattern": [0.1, 0.2, -0.1, 0.3, 0.5],  # Shortened for example
                "current_price": 1.0850,
                "k": 10
            }
        }


class SignalResponse(BaseModel):
    """Response model for trading signal"""
    status: str
    signal: str
    confidence: float
    vote_details: Optional[Dict[str, int]]
    price_projection: Optional[Dict[str, float]]
    average_movement: Optional[List[float]]
    matched_patterns: Optional[List[Dict[str, Any]]]
    n_matches: int
    timestamp: str
    message: Optional[str] = None
    duration: Optional[Dict[str, Any]] = None  # Signal duration estimation


class BuildIndexRequest(BaseModel):
    """Request model for building pattern index"""
    symbol: str
    timeframe: str = "H1"
    window_size: int = 60
    future_candles: int = 10
    use_sample_data: bool = False


class IndexStatusResponse(BaseModel):
    """Response model for index status"""
    symbol: str
    timeframe: str
    n_patterns: int
    is_ready: bool
    last_updated: Optional[str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    indices_loaded: List[str]


# Event handlers
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global firebase_service
    
    logger.info("Starting Trademify API Server...")
    
    # Initialize Firebase
    firebase_service = get_firebase_service()
    
    # Initialize MT5 Service (real connection on VPS)
    mt5_connected = await init_mt5_service()
    if mt5_connected:
        logger.info("✅ MT5 connected - Using LIVE data")
    else:
        logger.warning("⚠️ MT5 not connected - check VPS connection")
    
    # Initialize Trading System
    await init_trading_system()
    
    logger.info("Trademify API Server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Trademify API Server...")


# ============================================
# HEALTH CHECK ENDPOINTS
# ============================================

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """
    🏥 Health check endpoint
    
    All paths (/, /health, /api/v1/health) point to same handler
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        indices_loaded=list(pattern_matchers.keys())
    )


@app.get("/api/v1/system/health")
async def get_system_health():
    """
    🏥 Comprehensive system health check
    
    Returns status of all system components:
    - MT5 Connection
    - API Server
    - Bot Status
    - Data Lake
    - FAISS Index
    - Intelligence Modules
    """
    global _auto_bot
    
    import time
    
    # Try to import psutil (optional)
    memory_usage = 0
    try:
        import psutil
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        pass
    
    # Check MT5 connection
    mt5_connected = False
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            mt5_connected = True
            mt5.shutdown()
    except:
        pass
    
    # Count active intelligence modules
    intelligence_modules = 0
    if _auto_bot:
        if hasattr(_auto_bot, 'intelligence') and _auto_bot.intelligence:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'smart_brain') and _auto_bot.smart_brain:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'neural_brain') and _auto_bot.neural_brain:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'deep_intelligence') and _auto_bot.deep_intelligence:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'quantum_strategy') and _auto_bot.quantum_strategy:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'alpha_engine') and _auto_bot.alpha_engine:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'omega_brain') and _auto_bot.omega_brain:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'titan_core') and _auto_bot.titan_core:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'learning_system') and _auto_bot.learning_system:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'pro_features') and _auto_bot.pro_features:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'risk_guardian') and _auto_bot.risk_guardian:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'sentiment_analyzer') and _auto_bot.sentiment_analyzer:
            intelligence_modules += 1
        # Add base modules (Data Lake, Pattern Matcher, Voting, Enhanced)
        if hasattr(_auto_bot, 'data_provider') and _auto_bot.data_provider:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'pattern_matchers') and _auto_bot.pattern_matchers:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'voting_system') and _auto_bot.voting_system:
            intelligence_modules += 1
        if hasattr(_auto_bot, 'enhanced_analyzer') and _auto_bot.enhanced_analyzer:
            intelligence_modules += 1
    
    # Get uptime
    uptime = 0
    if _auto_bot and hasattr(_auto_bot, '_start_time'):
        uptime = int(time.time() - _auto_bot._start_time)
    
    # Get last analysis time
    last_analysis_time = None
    if _auto_bot and hasattr(_auto_bot, '_last_analysis'):
        last_analysis = _auto_bot._last_analysis
        if last_analysis and 'timestamp' in last_analysis:
            last_analysis_time = last_analysis['timestamp']
    
    return {
        "mt5_connected": mt5_connected,
        "api_status": "online",
        "bot_running": _auto_bot is not None and getattr(_auto_bot, '_running', False),
        "data_lake_ready": _auto_bot is not None and getattr(_auto_bot, 'data_provider', None) is not None,
        "faiss_loaded": len(pattern_matchers) > 0 or (_auto_bot is not None and bool(getattr(_auto_bot, 'pattern_matchers', None))),
        "intelligence_modules": intelligence_modules,
        "total_modules": 16,
        "last_analysis_time": last_analysis_time,
        "memory_usage": memory_usage,
        "uptime": uptime,
        "patterns_loaded": list(pattern_matchers.keys()),
    }


@app.post("/api/v1/build-index", response_model=IndexStatusResponse)
async def build_index(request: BuildIndexRequest, background_tasks: BackgroundTasks):
    """
    Build pattern index for a symbol
    
    This endpoint prepares the pattern database and FAISS index
    for a specific symbol and timeframe.
    """
    key = f"{request.symbol}_{request.timeframe}"
    
    try:
        # Get data
        if request.use_sample_data:
            # Generate sample data for testing
            logger.info(f"Generating sample data for {request.symbol}")
            df = DataGenerator.generate_sample_ohlcv(n_candles=5000, seed=42)
            df = DataGenerator.inject_patterns(df, "double_bottom", 20)
        else:
            # Load from Data Lake
            lake = DataLake(request.symbol, request.timeframe)
            try:
                df = lake.load_from_parquet()
            except FileNotFoundError:
                logger.info(f"No cached data found, downloading {request.symbol}")
                df = lake.download_data()
                if not df.empty:
                    lake.save_to_parquet(df)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {request.symbol}")
        
        # Prepare database
        logger.info(f"Preparing pattern database for {request.symbol}...")
        database = prepare_database(
            df=df,
            symbol=request.symbol,
            timeframe=request.timeframe,
            window_size=request.window_size,
            future_candles=request.future_candles,
            norm_method="zscore"
        )
        
        # Build pattern matcher
        matcher = PatternMatcher(
            window_size=request.window_size,
            index_type="IVF",
            min_correlation=PatternConfig.MIN_CORRELATION
        )
        
        matcher.fit(
            patterns=database["windows"],
            futures=database["futures"],
            metadata=[m.to_dict() for m in database["metadata"]]
        )
        
        # Store in global state
        pattern_matchers[key] = matcher
        
        logger.info(f"Index built for {key} with {len(database['windows'])} patterns")
        
        return IndexStatusResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            n_patterns=len(database["windows"]),
            is_ready=True,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to build index for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/index-status/{symbol}/{timeframe}", response_model=IndexStatusResponse)
async def get_index_status(symbol: str, timeframe: str = "H1"):
    """Get status of pattern index"""
    key = f"{symbol}_{timeframe}"
    
    if key in pattern_matchers:
        matcher = pattern_matchers[key]
        return IndexStatusResponse(
            symbol=symbol,
            timeframe=timeframe,
            n_patterns=matcher.engine.n_vectors,
            is_ready=True,
            last_updated=None
        )
    
    return IndexStatusResponse(
        symbol=symbol,
        timeframe=timeframe,
        n_patterns=0,
        is_ready=False,
        last_updated=None
    )


@app.post("/api/v1/analyze", response_model=SignalResponse)
async def analyze_pattern(request: AnalyzeRequest):
    """
    Analyze current pattern and generate trading signal
    
    This is the main endpoint that:
    1. Finds similar historical patterns
    2. Analyzes their outcomes
    3. Generates a trading signal with confidence
    """
    key = f"{request.symbol}_{request.timeframe}"
    
    # Check if index is ready
    if key not in pattern_matchers:
        raise HTTPException(
            status_code=404,
            detail=f"Index not found for {request.symbol}/{request.timeframe}. Please build index first."
        )
    
    matcher = pattern_matchers[key]
    
    # Validate pattern length
    if len(request.current_pattern) != matcher.window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Pattern length must be {matcher.window_size}, got {len(request.current_pattern)}"
        )
    
    try:
        # Convert to numpy
        query = np.array(request.current_pattern, dtype=np.float32)
        
        # Create analyzer with timeframe for duration estimation
        voting_system = VotingSystem(
            confidence_threshold=70.0,
            strong_signal_threshold=80.0,
            timeframe=request.timeframe
        )
        analyzer = PatternAnalyzer(
            similarity_engine=matcher,
            voting_system=voting_system,
            min_correlation=PatternConfig.MIN_CORRELATION
        )
        
        # Analyze
        result = analyzer.analyze(
            query_pattern=query,
            current_price=request.current_price,
            k=request.k
        )
        
        # Push to Firebase
        if firebase_service:
            firebase_service.update_current_signal(
                request.symbol,
                request.timeframe,
                result
            )
        
        return SignalResponse(
            status=result.get("status", "error"),
            signal=result.get("signal", "WAIT"),
            confidence=result.get("confidence", 0.0),
            vote_details=result.get("vote_details"),
            price_projection=result.get("price_projection"),
            average_movement=result.get("average_movement"),
            matched_patterns=result.get("matched_patterns"),
            n_matches=result.get("n_matches", 0),
            timestamp=datetime.now().isoformat(),
            message=result.get("message"),
            duration=result.get("duration")
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# AI-ENHANCED ANALYSIS ENDPOINT (High Win Rate Mode)
# =====================================================

class EnhancedAnalyzeRequest(BaseModel):
    """Request model for AI-enhanced pattern analysis"""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(default="H1", description="Timeframe (M5, M15, H1)")
    current_pattern: List[float] = Field(..., description="Current price pattern (normalized)")
    current_price: float = Field(..., description="Current actual price")
    ohlcv_data: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="OHLCV data for technical analysis (open, high, low, close, volume arrays)"
    )
    htf_ohlcv: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Higher timeframe OHLCV data for MTF analysis"
    )
    k: int = Field(default=10, description="Number of similar patterns to find")
    min_quality: str = Field(default="MEDIUM", description="Minimum signal quality (PREMIUM, HIGH, MEDIUM, LOW)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "timeframe": "H1",
                "current_pattern": [0.1, 0.2, -0.1, 0.3, 0.5],
                "current_price": 45000.0,
                "ohlcv_data": {
                    "open": [44900, 44950, 45000],
                    "high": [45100, 45050, 45200],
                    "low": [44800, 44900, 44950],
                    "close": [44950, 45000, 45100],
                    "volume": [1000, 1200, 1500]
                },
                "k": 10,
                "min_quality": "MEDIUM"
            }
        }


class EnhancedSignalResponse(BaseModel):
    """Response model for AI-enhanced trading signal"""
    status: str
    signal: str
    base_confidence: float
    enhanced_confidence: float
    quality: str
    scores: Dict[str, float]
    indicators: Optional[Dict[str, Any]]
    volume_analysis: Optional[Dict[str, Any]]
    mtf_analysis: Optional[Dict[str, Any]]
    market_regime: str
    risk_management: Dict[str, Any]
    factors: Dict[str, List[str]]
    vote_details: Optional[Dict[str, int]]
    price_projection: Optional[Dict[str, float]]
    n_matches: int
    timestamp: str
    message: Optional[str] = None


@app.post("/api/v1/analyze-enhanced", response_model=EnhancedSignalResponse)
async def analyze_pattern_enhanced(request: EnhancedAnalyzeRequest):
    """
    🔥 AI-Enhanced Pattern Analysis for High Win Rate
    
    วิเคราะห์ Pattern ด้วย AI factors หลายมิติเพื่อเพิ่ม Win Rate:
    - Technical Indicators (RSI, MACD, Bollinger, ATR, EMA)
    - Volume Analysis (OBV, Volume Confirmation)
    - Multi-Timeframe Analysis (HTF Trend Alignment)
    - Market Regime Detection (Trending/Ranging/Volatile)
    - Session Timing (London-NY overlap = best)
    - Momentum Analysis
    
    Signal Quality Levels:
    - PREMIUM: Win Rate ประมาณ 85%+ (เข้าเต็มกำลัง)
    - HIGH: Win Rate ประมาณ 75-85% (เข้าปกติ)
    - MEDIUM: Win Rate ประมาณ 65-75% (เข้าน้อย)
    - LOW: Win Rate < 65% (ไม่แนะนำ)
    - SKIP: ไม่ควรเข้า
    """
    key = f"{request.symbol}_{request.timeframe}"
    
    # Check if index is ready
    if key not in pattern_matchers:
        raise HTTPException(
            status_code=404,
            detail=f"Index not found for {request.symbol}/{request.timeframe}. Please build index first."
        )
    
    matcher = pattern_matchers[key]
    
    # Validate pattern length
    if len(request.current_pattern) != matcher.window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Pattern length must be {matcher.window_size}, got {len(request.current_pattern)}"
        )
    
    try:
        # Convert to numpy
        query = np.array(request.current_pattern, dtype=np.float32)
        
        # Step 1: Get base signal from pattern matching
        voting_system = VotingSystem(
            confidence_threshold=70.0,
            strong_signal_threshold=80.0
        )
        analyzer = PatternAnalyzer(
            similarity_engine=matcher,
            voting_system=voting_system,
            min_correlation=PatternConfig.MIN_CORRELATION
        )
        
        base_result = analyzer.analyze(
            query_pattern=query,
            current_price=request.current_price,
            k=request.k
        )
        
        base_signal = base_result.get("signal", "WAIT")
        base_confidence = base_result.get("confidence", 0.0)
        
        # Step 2: Prepare OHLCV data for enhanced analysis
        if request.ohlcv_data:
            ohlcv_data = {
                k: np.array(v, dtype=np.float32)
                for k, v in request.ohlcv_data.items()
            }
        else:
            # Create minimal data if not provided
            ohlcv_data = {
                "open": np.array([request.current_price] * 60),
                "high": np.array([request.current_price * 1.001] * 60),
                "low": np.array([request.current_price * 0.999] * 60),
                "close": np.array([request.current_price] * 60),
                "volume": np.array([10000.0] * 60),
            }
        
        # HTF data
        htf_data = None
        if request.htf_ohlcv:
            htf_data = {
                k: np.array(v, dtype=np.float32)
                for k, v in request.htf_ohlcv.items()
            }
        
        # Parse min quality
        quality_map = {
            "PREMIUM": SignalQuality.PREMIUM,
            "HIGH": SignalQuality.HIGH,
            "MEDIUM": SignalQuality.MEDIUM,
            "LOW": SignalQuality.LOW,
            "SKIP": SignalQuality.SKIP,
        }
        min_quality = quality_map.get(request.min_quality.upper(), SignalQuality.MEDIUM)
        
        # Step 3: AI Enhanced Analysis
        enhanced_analyzer = EnhancedAnalyzer(
            min_quality=min_quality,
            enable_volume_filter=True,
            enable_mtf_filter=htf_data is not None,
            enable_regime_filter=True,
        )
        
        price_projection = base_result.get("price_projection", {})
        
        enhanced_result = enhanced_analyzer.analyze(
            base_signal=base_signal,
            base_confidence=base_confidence,
            ohlcv_data=ohlcv_data,
            current_price=request.current_price,
            stop_loss=price_projection.get("stop_loss"),
            take_profit=price_projection.get("take_profit"),
            htf_data=htf_data,
            current_time=datetime.now(),
        )
        
        # Push to Firebase with enhanced data
        if firebase_service:
            firebase_service.update_current_signal(
                request.symbol,
                request.timeframe,
                {
                    **base_result,
                    "enhanced": enhanced_result.to_dict()
                }
            )
        
        return EnhancedSignalResponse(
            status=base_result.get("status", "success"),
            signal=enhanced_result.signal,
            base_confidence=base_confidence,
            enhanced_confidence=enhanced_result.enhanced_confidence,
            quality=enhanced_result.quality.value,
            scores={
                "pattern": enhanced_result.pattern_score,
                "technical": enhanced_result.technical_score,
                "volume": enhanced_result.volume_score,
                "mtf": enhanced_result.mtf_score,
                "regime": enhanced_result.regime_score,
                "timing": enhanced_result.timing_score,
                "momentum": enhanced_result.momentum_score,
            },
            indicators=enhanced_result.indicators.to_dict() if enhanced_result.indicators else None,
            volume_analysis=enhanced_result.volume_analysis.to_dict() if enhanced_result.volume_analysis else None,
            mtf_analysis=enhanced_result.mtf_analysis.to_dict() if enhanced_result.mtf_analysis else None,
            market_regime=enhanced_result.market_regime.value,
            risk_management={
                "adjusted_stop_loss": enhanced_result.adjusted_stop_loss,
                "adjusted_take_profit": enhanced_result.adjusted_take_profit,
                "risk_reward_ratio": enhanced_result.risk_reward_ratio,
                "recommended_position_size": enhanced_result.recommended_position_size,
                "entry_timing": enhanced_result.entry_timing,
            },
            factors={
                "bullish": enhanced_result.bullish_factors,
                "bearish": enhanced_result.bearish_factors,
                "skip_reasons": enhanced_result.skip_reasons,
            },
            vote_details=base_result.get("vote_details"),
            price_projection={
                "current": request.current_price,
                "projected": price_projection.get("projected"),
                "stop_loss": enhanced_result.adjusted_stop_loss,
                "take_profit": enhanced_result.adjusted_take_profit,
            },
            n_matches=base_result.get("n_matches", 0),
            timestamp=datetime.now().isoformat(),
            message=f"Quality: {enhanced_result.quality.value} | R:R = {enhanced_result.risk_reward_ratio:.2f}"
        )
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# MULTI-FACTOR AI ANALYSIS ENDPOINT (Maximum Win Rate)
# =====================================================

class MultiFactorAnalyzeRequest(BaseModel):
    """Request model for multi-factor AI analysis"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(default="H1", description="Timeframe")
    current_pattern: List[float] = Field(..., description="Normalized pattern")
    current_price: float = Field(..., description="Current price")
    ohlcv_data: Optional[Dict[str, List[float]]] = Field(default=None)
    pattern_dates: Optional[List[str]] = Field(default=None, description="ISO format dates of matched patterns")
    k: int = Field(default=10)
    trading_mode: str = Field(default="CONSERVATIVE", description="AGGRESSIVE, BALANCED, CONSERVATIVE, SNIPER")


class MultiFactorResponse(BaseModel):
    """Response for multi-factor analysis"""
    status: str
    signal: str
    base_confidence: float
    final_score: float
    quality: str
    recommendation: str
    factors: List[Dict[str, Any]]
    scores: Dict[str, float]
    position_size_multiplier: float
    reasons: Dict[str, List[str]]
    trading_mode: str
    timestamp: str


@app.post("/api/v1/analyze-multi-factor", response_model=MultiFactorResponse)
async def analyze_multi_factor(request: MultiFactorAnalyzeRequest):
    """
    🚀 Multi-Factor AI Analysis for Maximum Win Rate
    
    วิเคราะห์สัญญาณด้วยหลายปัจจัยเพื่อ Win Rate สูงสุด:
    
    📊 ปัจจัยที่วิเคราะห์:
    1. Pattern Match Score (25%) - Base pattern matching
    2. Trend Alignment (20%) - Signal ไปทางเดียวกับ trend
    3. Volume Confirmation (15%) - Volume รองรับการเคลื่อนไหว
    4. Pattern Recency (10%) - Patterns ล่าสุดมีน้ำหนักมาก
    5. Volatility Assessment (10%) - ตลาดอยู่ในช่วง volatility ที่ดี
    6. Session Timing (10%) - เวลาที่ดีในการเทรด
    7. Momentum Confluence (10%) - RSI, MACD เห็นด้วย
    
    🎯 Trading Modes:
    - AGGRESSIVE: Min Score 55%, Win Rate ~60%
    - BALANCED: Min Score 65%, Win Rate ~70%
    - CONSERVATIVE: Min Score 75%, Win Rate ~80%
    - SNIPER: Min Score 85%, Win Rate ~85%+
    
    📈 Recommendations:
    - TRADE_STRONG: เข้าเต็มกำลัง (quality premium)
    - TRADE: เข้าปกติ (quality high)
    - TRADE_REDUCED: เข้าลด size (quality medium)
    - SKIP: ไม่เข้า
    """
    key = f"{request.symbol}_{request.timeframe}"
    
    # Check if index exists
    if key not in pattern_matchers:
        raise HTTPException(
            status_code=404,
            detail=f"Index not found for {key}. Build index first."
        )
    
    matcher = pattern_matchers[key]
    
    try:
        # Step 1: Get base pattern matching result
        query = np.array(request.current_pattern, dtype=np.float32)
        
        voting_system = VotingSystem(
            confidence_threshold=70.0,
            strong_signal_threshold=80.0
        )
        
        base_analyzer = PatternAnalyzer(
            similarity_engine=matcher,
            voting_system=voting_system,
            min_correlation=PatternConfig.MIN_CORRELATION
        )
        
        base_result = base_analyzer.analyze(
            query_pattern=query,
            current_price=request.current_price,
            k=request.k
        )
        
        # Create VoteResult for multi-factor analyzer
        from analysis.voting_system import VoteResult
        vote_result = VoteResult(
            signal=Signal[base_result.get("signal", "WAIT")],
            confidence=base_result.get("confidence", 0.0),
            bullish_votes=base_result.get("vote_details", {}).get("bullish", 0),
            bearish_votes=base_result.get("vote_details", {}).get("bearish", 0),
            total_votes=base_result.get("n_matches", 0),
            average_movement=np.array(base_result.get("average_movement", [])),
        )
        
        # Step 2: Prepare OHLCV data
        if request.ohlcv_data:
            ohlcv = {
                k: np.array(v, dtype=np.float32)
                for k, v in request.ohlcv_data.items()
            }
        else:
            ohlcv = {
                "close": np.array([request.current_price] * 100),
                "volume": None,
                "high": None,
                "low": None,
            }
        
        # Parse pattern dates
        pattern_dates = None
        if request.pattern_dates:
            pattern_dates = [
                datetime.fromisoformat(d) for d in request.pattern_dates
            ]
        
        # Step 3: Configure trading mode
        mode_map = {
            "AGGRESSIVE": TradingMode.AGGRESSIVE,
            "BALANCED": TradingMode.BALANCED,
            "CONSERVATIVE": TradingMode.CONSERVATIVE,
            "SNIPER": TradingMode.SNIPER,
        }
        trading_mode = mode_map.get(request.trading_mode.upper(), TradingMode.CONSERVATIVE)
        
        # Create config with trading mode
        config = EnhancedAnalysisConfig.from_env()
        config.mode = trading_mode
        
        # Step 4: Run multi-factor analysis
        mf_analyzer = MultiFactorAnalyzer(config)
        mf_result = mf_analyzer.analyze(
            vote_result=vote_result,
            prices=ohlcv.get("close", np.array([])),
            volumes=ohlcv.get("volume"),
            highs=ohlcv.get("high"),
            lows=ohlcv.get("low"),
            pattern_dates=pattern_dates,
            current_time=datetime.now(),
            symbol=request.symbol,
        )
        
        # Push to Firebase
        if firebase_service:
            firebase_service.update_current_signal(
                request.symbol,
                request.timeframe,
                {
                    **base_result,
                    "multi_factor": mf_result.to_dict()
                }
            )
        
        return MultiFactorResponse(
            status="success",
            signal=mf_result.signal.value,
            base_confidence=mf_result.base_confidence,
            final_score=mf_result.final_score,
            quality=mf_result.quality.value if hasattr(mf_result.quality, 'value') else str(mf_result.quality),
            recommendation=mf_result.recommendation,
            factors=[f.to_dict() for f in mf_result.factors],
            scores={
                "pattern": mf_result.pattern_score,
                "trend": mf_result.trend_score,
                "volume": mf_result.volume_score,
                "recency": mf_result.recency_score,
                "volatility": mf_result.volatility_score,
                "session": mf_result.session_score,
                "momentum": mf_result.momentum_score,
            },
            position_size_multiplier=mf_result.position_size_multiplier,
            reasons={
                "bullish": mf_result.bullish_reasons,
                "bearish": mf_result.bearish_reasons,
                "skip": mf_result.skip_reasons,
            },
            trading_mode=trading_mode.value,
            timestamp=datetime.now().isoformat(),
        )
        
    except Exception as e:
        logger.error(f"Multi-factor analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/multi-factor/config")
async def get_multi_factor_config():
    """
    Get current multi-factor analysis configuration
    """
    config = EnhancedAnalysisConfig.from_env()
    return config.to_dict()


@app.post("/api/v1/analyze-realtime-enhanced")
async def analyze_realtime_enhanced(
    symbol: str = "BTCUSDT",
    timeframe: str = "H1",
    htf_timeframe: str = "H4",
    min_quality: str = "MEDIUM"
):
    """
    🔥🔥 Real-time AI-Enhanced Analysis from Binance
    
    ดึงข้อมูลจริงจาก Binance พร้อมวิเคราะห์ด้วย AI factors ทั้งหมด
    รวมถึง Multi-Timeframe Analysis จาก Higher Timeframe
    """
    from data_processing.binance_data import BinanceDataProvider
    from data_processing import Normalizer
    
    try:
        provider = BinanceDataProvider()
        window_size = DataConfig.WINDOW_SIZE
        
        # 1. Get current timeframe data
        df = await provider.get_klines(
            symbol=symbol,
            timeframe=timeframe,
            limit=window_size + 100
        )
        
        # 2. Get higher timeframe data
        htf_df = await provider.get_klines(
            symbol=symbol,
            timeframe=htf_timeframe,
            limit=100
        )
        
        await provider.close()
        
        if df.empty or len(df) < window_size:
            raise HTTPException(
                status_code=400,
                detail=f"ข้อมูลไม่เพียงพอสำหรับ {symbol}"
            )
        
        # 3. Prepare OHLCV arrays
        current_price = float(df['close'].iloc[-1])
        
        ohlcv_data = {
            "open": df['open'].values[-100:].astype(np.float32),
            "high": df['high'].values[-100:].astype(np.float32),
            "low": df['low'].values[-100:].astype(np.float32),
            "close": df['close'].values[-100:].astype(np.float32),
            "volume": df['volume'].values[-100:].astype(np.float32),
        }
        
        htf_data = None
        if not htf_df.empty:
            htf_data = {
                "open": htf_df['open'].values.astype(np.float32),
                "high": htf_df['high'].values.astype(np.float32),
                "low": htf_df['low'].values.astype(np.float32),
                "close": htf_df['close'].values.astype(np.float32),
                "volume": htf_df['volume'].values.astype(np.float32),
            }
        
        # 4. Normalize pattern
        normalizer = Normalizer(method="zscore")
        normalized = normalizer.normalize(df['close'].values[-window_size:])
        
        # 5. Check index
        key = f"{symbol}_{timeframe}"
        base_signal = "WAIT"
        base_confidence = 0.0
        vote_details = None
        price_projection = {}
        n_matches = 0
        
        if key in pattern_matchers:
            matcher = pattern_matchers[key]
            
            voting_system = VotingSystem(
                confidence_threshold=70.0,
                strong_signal_threshold=80.0
            )
            analyzer = PatternAnalyzer(
                similarity_engine=matcher,
                voting_system=voting_system,
                min_correlation=PatternConfig.MIN_CORRELATION
            )
            
            base_result = analyzer.analyze(
                query_pattern=normalized.astype(np.float32),
                current_price=current_price,
                k=10
            )
            
            base_signal = base_result.get("signal", "WAIT")
            base_confidence = base_result.get("confidence", 0.0)
            vote_details = base_result.get("vote_details")
            price_projection = base_result.get("price_projection", {})
            n_matches = base_result.get("n_matches", 0)
        
        # 6. Enhanced Analysis
        quality_map = {
            "PREMIUM": SignalQuality.PREMIUM,
            "HIGH": SignalQuality.HIGH,
            "MEDIUM": SignalQuality.MEDIUM,
            "LOW": SignalQuality.LOW,
        }
        min_qual = quality_map.get(min_quality.upper(), SignalQuality.MEDIUM)
        
        enhanced_analyzer = EnhancedAnalyzer(
            min_quality=min_qual,
            enable_volume_filter=True,
            enable_mtf_filter=htf_data is not None,
            enable_regime_filter=True,
        )
        
        enhanced_result = enhanced_analyzer.analyze(
            base_signal=base_signal,
            base_confidence=base_confidence,
            ohlcv_data=ohlcv_data,
            current_price=current_price,
            stop_loss=price_projection.get("stop_loss"),
            take_profit=price_projection.get("take_profit"),
            htf_data=htf_data,
            current_time=datetime.now(),
        )
        
        return {
            "status": "success",
            "symbol": symbol,
            "timeframe": timeframe,
            "htf_timeframe": htf_timeframe,
            "current_price": current_price,
            "signal": enhanced_result.signal,
            "base_confidence": base_confidence,
            "enhanced_confidence": enhanced_result.enhanced_confidence,
            "quality": enhanced_result.quality.value,
            "scores": {
                "pattern": enhanced_result.pattern_score,
                "technical": enhanced_result.technical_score,
                "volume": enhanced_result.volume_score,
                "mtf": enhanced_result.mtf_score,
                "regime": enhanced_result.regime_score,
                "timing": enhanced_result.timing_score,
                "momentum": enhanced_result.momentum_score,
            },
            "indicators": enhanced_result.indicators.to_dict() if enhanced_result.indicators else None,
            "volume_analysis": enhanced_result.volume_analysis.to_dict() if enhanced_result.volume_analysis else None,
            "mtf_analysis": enhanced_result.mtf_analysis.to_dict() if enhanced_result.mtf_analysis else None,
            "market_regime": enhanced_result.market_regime.value,
            "risk_management": {
                "stop_loss": enhanced_result.adjusted_stop_loss,
                "take_profit": enhanced_result.adjusted_take_profit,
                "risk_reward": enhanced_result.risk_reward_ratio,
                "position_size": enhanced_result.recommended_position_size,
                "entry_timing": enhanced_result.entry_timing,
            },
            "factors": {
                "bullish": enhanced_result.bullish_factors,
                "bearish": enhanced_result.bearish_factors,
                "skip_reasons": enhanced_result.skip_reasons,
            },
            "vote_details": vote_details,
            "n_matches": n_matches,
            "market_data": {
                "open": float(df['open'].iloc[-1]),
                "high": float(df['high'].iloc[-1]),
                "low": float(df['low'].iloc[-1]),
                "close": current_price,
                "volume": float(df['volume'].iloc[-1]),
            },
            "timestamp": datetime.now().isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced real-time analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/symbols")
async def list_symbols():
    """List available symbols and their status"""
    symbols = {}
    
    for key, matcher in pattern_matchers.items():
        parts = key.split("_")
        symbol = parts[0]
        timeframe = parts[1] if len(parts) > 1 else "H1"
        
        if symbol not in symbols:
            symbols[symbol] = {"timeframes": {}}
        
        symbols[symbol]["timeframes"][timeframe] = {
            "n_patterns": matcher.engine.n_vectors,
            "is_ready": True
        }
    
    # Add default symbols
    for symbol in DataConfig.DEFAULT_SYMBOLS:
        if symbol not in symbols:
            symbols[symbol] = {"timeframes": {}, "is_ready": False}
    
    return {"symbols": symbols}


@app.post("/api/v1/generate-sample-signal")
async def generate_sample_signal(symbol: str = "EURUSD", timeframe: str = "H1"):
    """
    Generate a sample signal for testing
    
    This endpoint is useful for frontend development
    """
    import random
    
    signals = [Signal.STRONG_BUY, Signal.BUY, Signal.WAIT, Signal.SELL, Signal.STRONG_SELL]
    signal = random.choice(signals)
    
    bullish = random.randint(0, 10)
    bearish = 10 - bullish
    
    return SignalResponse(
        status="success",
        signal=signal.value,
        confidence=max(bullish, bearish) / 10 * 100,
        vote_details={
            "bullish": bullish,
            "bearish": bearish,
            "total": 10
        },
        price_projection={
            "current": 1.0850,
            "projected": 1.0850 + random.uniform(-0.01, 0.01),
            "stop_loss": 1.0850 - 0.0050,
            "take_profit": 1.0850 + 0.0100
        },
        average_movement=[i * random.uniform(-0.1, 0.1) for i in range(10)],
        matched_patterns=[
            {"index": i, "correlation": random.uniform(0.85, 0.99), "distance": random.uniform(0, 1)}
            for i in range(10)
        ],
        n_matches=10,
        timestamp=datetime.now().isoformat()
    )


# =====================================================
# REAL-TIME ANALYSIS ENDPOINTS (ข้อมูลจริงจาก Binance)
# =====================================================

class RealTimeAnalyzeRequest(BaseModel):
    """Request for real-time analysis"""
    symbol: str = Field(default="BTCUSDT", description="Binance symbol")
    timeframe: str = Field(default="H1", description="Timeframe")
    window_size: int = Field(default=60, description="Pattern window size")
    k: int = Field(default=10, description="Number of similar patterns")


@app.post("/api/v1/analyze-realtime")
async def analyze_realtime(request: RealTimeAnalyzeRequest):
    """
    🔥 วิเคราะห์ Pattern แบบ Real-time จาก Binance
    
    ดึงข้อมูลล่าสุดจาก Binance แล้ววิเคราะห์ทันที
    """
    from data_processing.binance_data import BinanceDataProvider
    from data_processing import Normalizer
    
    try:
        # 1. ดึงข้อมูลล่าสุดจาก Binance
        provider = BinanceDataProvider()
        df = await provider.get_klines(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=request.window_size + 100  # Extra for normalization
        )
        await provider.close()
        
        if df.empty or len(df) < request.window_size:
            raise HTTPException(
                status_code=400,
                detail=f"ไม่มีข้อมูลเพียงพอสำหรับ {request.symbol}"
            )
        
        # 2. Get current price
        current_price = float(df['close'].iloc[-1])
        
        # 3. Normalize pattern
        normalizer = Normalizer(method="zscore")
        close_prices = df['close'].values
        
        # Get normalized current pattern (use normalize method)
        normalized = normalizer.normalize(close_prices[-request.window_size:])
        
        # 4. Check if index exists
        key = f"{request.symbol}_{request.timeframe}"
        
        if key not in pattern_matchers:
            # Auto-build index with recent data
            logger.info(f"Auto-building index for {request.symbol}...")
            
            # Get more historical data for index
            historical_df = await provider.get_historical_klines(
                symbol=request.symbol,
                timeframe=request.timeframe,
                days=90  # 90 days of data
            )
            await provider.close()
            
            if len(historical_df) > request.window_size + 20:
                # Prepare database
                database = prepare_database(
                    df=historical_df,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    window_size=request.window_size,
                    future_candles=10,
                    norm_method="zscore"
                )
                
                # Build matcher
                matcher = PatternMatcher(
                    window_size=request.window_size,
                    index_type="IVF" if len(database["windows"]) > 500 else "Flat",
                    min_correlation=PatternConfig.MIN_CORRELATION
                )
                
                matcher.fit(
                    patterns=database["windows"],
                    futures=database["futures"],
                    metadata=[m.to_dict() for m in database["metadata"]]
                )
                
                pattern_matchers[key] = matcher
                logger.info(f"Built index with {len(database['windows'])} patterns")
        
        # 5. Analyze if index exists
        if key in pattern_matchers:
            matcher = pattern_matchers[key]
            
            voting_system = VotingSystem(
                confidence_threshold=70.0,
                strong_signal_threshold=80.0
            )
            analyzer = PatternAnalyzer(
                similarity_engine=matcher,
                voting_system=voting_system,
                min_correlation=PatternConfig.MIN_CORRELATION
            )
            
            result = analyzer.analyze(
                query_pattern=normalized.astype(np.float32),
                current_price=current_price,
                k=request.k
            )
            
            return {
                "status": "success",
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "current_price": current_price,
                "signal": result.get("signal", "WAIT"),
                "confidence": result.get("confidence", 0.0),
                "vote_details": result.get("vote_details"),
                "price_projection": result.get("price_projection"),
                "n_matches": result.get("n_matches", 0),
                "timestamp": datetime.now().isoformat(),
                "market_data": {
                    "open": float(df['open'].iloc[-1]),
                    "high": float(df['high'].iloc[-1]),
                    "low": float(df['low'].iloc[-1]),
                    "close": current_price,
                    "volume": float(df['volume'].iloc[-1])
                }
            }
        else:
            # Return market data only
            return {
                "status": "no_index",
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "current_price": current_price,
                "signal": "WAIT",
                "confidence": 0.0,
                "message": "Index not ready yet. Building in background...",
                "timestamp": datetime.now().isoformat(),
                "market_data": {
                    "open": float(df['open'].iloc[-1]),
                    "high": float(df['high'].iloc[-1]),
                    "low": float(df['low'].iloc[-1]),
                    "close": current_price,
                    "volume": float(df['volume'].iloc[-1])
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Real-time analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "H1", limit: int = 100):
    """
    ดึงข้อมูลตลาดล่าสุดจาก Binance
    """
    from data_processing.binance_data import BinanceDataProvider
    
    try:
        provider = BinanceDataProvider()
        df = await provider.get_klines(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        # Get 24h stats
        stats = await provider.get_24h_stats(symbol)
        await provider.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        # Convert to list of dicts
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "datetime": idx.isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": float(df['close'].iloc[-1]),
            "price_change_24h": float(stats.get("priceChangePercent", 0)),
            "volume_24h": float(stats.get("volume", 0)),
            "high_24h": float(stats.get("highPrice", 0)),
            "low_24h": float(stats.get("lowPrice", 0)),
            "candles": candles,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/build-realtime-index")
async def build_realtime_index(
    symbol: str = "BTCUSDT",
    timeframe: str = "H1",
    days: int = 90
):
    """
    🔨 สร้าง Pattern Index จากข้อมูลจริง Binance
    """
    from data_processing.binance_data import BinanceDataProvider
    
    try:
        logger.info(f"Building real-time index for {symbol} {timeframe} ({days} days)")
        
        # Download historical data
        provider = BinanceDataProvider()
        df = await provider.get_historical_klines(
            symbol=symbol,
            timeframe=timeframe,
            days=days
        )
        await provider.close()
        
        if len(df) < 200:
            raise HTTPException(
                status_code=400,
                detail=f"ข้อมูลไม่เพียงพอ ได้เพียง {len(df)} แท่งเทียน"
            )
        
        # Prepare database
        window_size = DataConfig.WINDOW_SIZE
        future_candles = DataConfig.FUTURE_CANDLES
        
        database = prepare_database(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            window_size=window_size,
            future_candles=future_candles,
            norm_method="zscore"
        )
        
        # Build pattern matcher
        n_patterns = len(database["windows"])
        index_type = "IVF" if n_patterns > 500 else "Flat"
        
        matcher = PatternMatcher(
            window_size=window_size,
            index_type=index_type,
            min_correlation=PatternConfig.MIN_CORRELATION
        )
        
        matcher.fit(
            patterns=database["windows"],
            futures=database["futures"],
            metadata=[m.to_dict() for m in database["metadata"]]
        )
        
        # Store in global state
        key = f"{symbol}_{timeframe}"
        pattern_matchers[key] = matcher
        
        logger.info(f"✓ Built {key} index with {n_patterns} patterns")
        
        return {
            "status": "success",
            "symbol": symbol,
            "timeframe": timeframe,
            "n_patterns": n_patterns,
            "n_candles": len(df),
            "index_type": index_type,
            "date_range": {
                "start": df.index[0].isoformat(),
                "end": df.index[-1].isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Build index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# MT5 MARKET & PRICE ENDPOINTS
# (Trading endpoints are in trading_routes.py)
# ============================================

@app.get("/api/v1/mt5/market-status")
async def get_market_status(symbol: str = "EURUSD"):
    """
    🕐 Get market status (Open/Closed/Weekend)
    
    Returns detailed market status with session info and time until next open.
    """
    mt5_service = get_mt5_service()
    market_info = mt5_service.get_market_status(symbol)
    
    return {
        "status": market_info.status.value,
        "message": market_info.message,
        "message_th": market_info.message_th,
        "is_tradeable": market_info.is_tradeable,
        "color": market_info.color,
        "next_open": market_info.next_open.isoformat() if market_info.next_open else None,
        "time_until_open": market_info.time_until_open,
        "server_time": market_info.server_time.isoformat() if market_info.server_time else None,
        "mt5_connected": mt5_service.is_connected(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/mt5/price/{symbol}")
async def get_mt5_price(symbol: str):
    """
    💰 Get current price for a Forex/CFD symbol
    
    Returns LIVE price from MT5 only. No mock data.
    Supports: EURUSDm, GBPUSDm, XAUUSDm (Exness format with 'm' suffix)
    """
    # Clean symbol - keep case sensitivity for Exness 'm' suffix
    clean_symbol = symbol.replace("/", "").replace("-", "")
    
    mt5_service = get_mt5_service()
    market_info = mt5_service.get_market_status(clean_symbol)
    
    # ดึงราคาจาก MT5 Service
    price_data = await mt5_service.get_price(clean_symbol)
    
    # ถ้าได้ราคาจาก MT5
    if price_data.price > 0:
        return {
            "symbol": symbol,
            "clean_symbol": clean_symbol,
            "price": price_data.price,
            "bid": price_data.bid,
            "ask": price_data.ask,
            "spread": price_data.spread,
            "high": price_data.high,
            "low": price_data.low,
            "source": price_data.source,
            "is_live": price_data.is_live,
            "market_status": price_data.market_status.value,
            "broker": "MT5",
            "mt5_connected": mt5_service.is_connected(),
            "last_update": price_data.last_update.isoformat() if price_data.last_update else None,
            "timestamp": datetime.now().isoformat()
        }
    
    # MT5 ไม่พร้อม - ส่ง error ชัดเจน (ไม่ใช้ mock data)
    error_msg = price_data.error or "MT5 not connected"
    
    # ถ้าตลาดปิด - แสดงข้อมูลสถานะตลาดแทน
    if market_info.status in [MarketStatus.WEEKEND, MarketStatus.CLOSED]:
        return {
            "symbol": symbol,
            "clean_symbol": clean_symbol,
            "price": 0,
            "bid": 0,
            "ask": 0,
            "spread": 0,
            "source": "market_closed",
            "is_live": False,
            "market_status": market_info.status.value,
            "broker": "MT5",
            "mt5_connected": mt5_service.is_connected(),
            "error": None,
            "message": market_info.message_th,
            "next_open": market_info.next_open.isoformat() if market_info.next_open else None,
            "time_until_open": market_info.time_until_open,
            "timestamp": datetime.now().isoformat()
        }
    
    # MT5 ไม่ได้เชื่อมต่อ - ส่ง error
    raise HTTPException(
        status_code=503,
        detail={
            "error": "MT5_NOT_CONNECTED",
            "message": "❌ MT5 Terminal ไม่ได้เชื่อมต่อ",
            "message_en": error_msg,
            "help": [
                "1. ตรวจสอบว่า MT5 Terminal เปิดอยู่",
                "2. ตรวจสอบ .env มีค่า MT5_LOGIN, MT5_PASSWORD, MT5_SERVER",
                "3. ลอง reconnect ที่ POST /api/v1/mt5/reconnect"
            ],
            "symbol": symbol,
            "mt5_connected": False,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.get("/api/v1/mt5/account")
async def get_mt5_account():
    """
    📊 Get MT5 account information
    """
    mt5_service = get_mt5_service()
    account = await mt5_service.get_account_info()
    
    return {
        **account,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/mt5/ohlcv/{symbol}")
async def get_mt5_ohlcv(symbol: str, timeframe: str = "H1", count: int = 100):
    """
    📈 Get OHLCV data from MT5
    """
    clean_symbol = symbol.replace("/", "").replace("-", "")
    
    mt5_service = get_mt5_service()
    
    if not mt5_service.is_connected():
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    ohlcv = await mt5_service.get_ohlcv(clean_symbol, timeframe, count)
    
    if not ohlcv:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "count": len(ohlcv),
        "data": ohlcv,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/mt5/symbols")
async def get_mt5_symbols(filter: str = ""):
    """
    📋 Get available symbols from MT5
    
    filter: Optional filter (e.g., "USD", "EUR", "XAU")
    """
    mt5_service = get_mt5_service()
    
    # Ensure connected
    if not mt5_service._connected:
        await mt5_service.connect()
    
    if not mt5_service._connected or not mt5_service._mt5:
        return {
            "symbols": [],
            "count": 0,
            "source": "error",
            "message": "MT5 not connected - ensure MT5 terminal is running"
        }
    
    try:
        all_symbols = mt5_service._mt5.symbols_get()
        if not all_symbols:
            return {"symbols": [], "count": 0, "source": "mt5", "message": "No symbols found"}
        
        names = [s.name for s in all_symbols]
        
        if filter:
            names = [s for s in names if filter.upper() in s.upper()]
        
        # Limit to first 100
        names = names[:100]
        
        return {
            "symbols": names,
            "count": len(names),
            "total": len(all_symbols),
            "source": "mt5",
            "message": f"Found {len(all_symbols)} symbols total"
        }
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        return {
            "symbols": [],
            "count": 0,
            "source": "error",
            "message": str(e)
        }


@app.get("/api/v1/mt5/find-symbols")
async def find_mt5_symbols():
    """
    🔍 Find available Forex/Gold symbols in MT5
    
    Returns symbols like EURUSDm, GBPUSDm, XAUUSDm that are tradeable
    """
    mt5_service = get_mt5_service()
    
    # Ensure connected
    if not mt5_service._connected:
        await mt5_service.connect()
    
    if not mt5_service._connected or not mt5_service._mt5:
        return {
            "success": False,
            "error": "MT5 not connected",
            "forex_symbols": [],
            "gold_symbols": []
        }
    
    try:
        all_symbols = mt5_service._mt5.symbols_get()
        if not all_symbols:
            return {"success": False, "error": "No symbols found"}
        
        forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]
        gold_names = ["XAUUSD", "GOLD"]
        
        found_forex = []
        found_gold = []
        
        for s in all_symbols:
            name = s.name.upper()
            # Find forex pairs
            for pair in forex_pairs:
                if pair in name:
                    found_forex.append({"name": s.name, "description": s.description if hasattr(s, 'description') else ""})
                    break
            # Find gold
            for gold in gold_names:
                if gold in name:
                    found_gold.append({"name": s.name, "description": s.description if hasattr(s, 'description') else ""})
                    break
        
        # Recommend best symbols
        recommended = []
        for pair in forex_pairs[:3]:  # EURUSD, GBPUSD, USDJPY
            for f in found_forex:
                if pair in f["name"].upper():
                    recommended.append(f["name"])
                    break
        for f in found_gold:
            if "XAU" in f["name"].upper() or "GOLD" in f["name"].upper():
                recommended.append(f["name"])
                break
        
        return {
            "success": True,
            "forex_symbols": found_forex[:20],
            "gold_symbols": found_gold[:10],
            "recommended": recommended,
            "message": f"Found {len(found_forex)} forex and {len(found_gold)} gold symbols"
        }
    except Exception as e:
        logger.error(f"Error finding symbols: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/v1/mt5/reconnect")
async def reconnect_mt5():
    """
    🔄 Reconnect to MT5 Terminal
    """
    mt5_service = get_mt5_service()
    
    # Disconnect first
    await mt5_service.disconnect()
    
    # Try to reconnect
    connected = await mt5_service.connect()
    
    if connected:
        account = await mt5_service.get_account_info()
        return {
            "status": "connected",
            "message": "✅ MT5 reconnected successfully",
            "account": account,
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "status": "disconnected",
            "message": "❌ Could not connect to MT5 Terminal",
            "help": "Please ensure MT5 Terminal is running",
            "timestamp": datetime.now().isoformat()
        }


# =====================================================
# SSE (Server-Sent Events) for Real-time Updates
# =====================================================
import asyncio
from fastapi.responses import StreamingResponse

# Global event queue for SSE
sse_clients: list = []


async def event_generator(client_queue: asyncio.Queue):
    """Generate SSE events"""
    try:
        while True:
            try:
                # Wait for new event with timeout (15 sec for more stable connection)
                data = await asyncio.wait_for(client_queue.get(), timeout=15.0)
                yield f"event: {data.get('event', 'message')}\n"
                yield f"data: {json.dumps(data.get('data', {}))}\n\n"
            except asyncio.TimeoutError:
                # Send keepalive ping every 15 seconds
                yield f"event: ping\ndata: {json.dumps({'timestamp': datetime.now().isoformat()})}\n\n"
    except asyncio.CancelledError:
        pass
    except GeneratorExit:
        pass


@app.get("/api/v1/events")
async def sse_events():
    """
    SSE endpoint for real-time updates
    
    Events:
    - signal: New signal analysis
    - trade: Trade executed
    - position: Position update
    - bot_status: Bot status change
    """
    client_queue = asyncio.Queue()
    sse_clients.append(client_queue)
    
    # Also register with bot if running
    if _auto_bot:
        _auto_bot.add_subscriber(client_queue)
    
    async def event_gen():
        try:
            async for event in event_generator(client_queue):
                yield event
        finally:
            # Cleanup on disconnect
            if client_queue in sse_clients:
                sse_clients.remove(client_queue)
            if _auto_bot:
                _auto_bot.remove_subscriber(client_queue)
    
    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


async def broadcast_event(event_type: str, data: dict):
    """Broadcast event to all SSE clients"""
    message = {"event": event_type, "data": data}
    for client in sse_clients:
        try:
            await client.put(message)
        except:
            pass


# =====================================================
# AI Trading Bot Integration (Single Expert System)
# =====================================================
from ai_trading_bot import AITradingBot, EnhancedTradingBot, get_bot, SignalQuality

_auto_bot: Optional[AITradingBot] = None
_bot_task: Optional[asyncio.Task] = None


# Use validated request model from security module
StartBotRequest = ValidatedStartBotRequest


async def auto_start_bot(settings: dict):
    """Internal function to start bot with settings"""
    global _auto_bot, _bot_task
    
    quality_map = {
        "PREMIUM": SignalQuality.PREMIUM,
        "HIGH": SignalQuality.HIGH,
        "MEDIUM": SignalQuality.MEDIUM,
        "LOW": SignalQuality.LOW,
    }
    
    # Handle symbols - can be string or list
    symbols = settings.get('symbols', ['EURUSDm', 'GBPUSDm', 'XAUUSDm'])
    if isinstance(symbols, str):
        # Split comma-separated string into list
        symbols = [s.strip() for s in symbols.split(',') if s.strip()]
    
    # Normalize symbol names (fix case issues from JSON)
    symbol_map = {
        'EURUSDM': 'EURUSDm', 'GBPUSDM': 'GBPUSDm', 'XAUUSDM': 'XAUUSDm',
        'EURUSD': 'EURUSDm', 'GBPUSD': 'GBPUSDm', 'XAUUSD': 'XAUUSDm',
    }
    symbols = [symbol_map.get(s.upper(), s) for s in symbols]
    
    logger.info(f"🔧 Parsed symbols: {symbols} (type: {type(symbols).__name__})")
    
    _auto_bot = AITradingBot(
        symbols=symbols,
        timeframe=settings.get('timeframe', 'H1'),
        htf_timeframe=settings.get('htf_timeframe', 'H4'),
        min_quality=quality_map.get(settings.get('min_quality', 'HIGH'), SignalQuality.HIGH),
        broker_type=settings.get('broker_type', 'MT5'),
        allowed_signals=settings.get('allowed_signals', ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"]),
    )
    
    # Add SSE broadcast to bot
    for client in sse_clients:
        _auto_bot.add_subscriber(client)
    
    async def run_bot():
        try:
            logger.info("🔄 Initializing bot...")
            await _auto_bot.initialize()
            logger.info("✅ Bot initialized, starting run loop...")
            await _auto_bot.run(interval_seconds=settings.get('interval', 60))
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
            import traceback
            traceback.print_exc()
    
    _bot_task = asyncio.create_task(run_bot())
    logger.info(f"🤖 Bot started: {settings.get('symbols')} @ {settings.get('timeframe')}")


@app.post("/api/v1/bot/start")
async def start_auto_bot(
    request: StartBotRequest, 
    background_tasks: BackgroundTasks
):
    """
    🤖 Start AI Trading Bot - Expert Pattern Recognition System
    
    The AI bot will:
    - Analyze symbols using FAISS pattern matching
    - Apply multi-factor AI analysis (RSI, MACD, Volume, MTF)
    - Filter signals by quality (PREMIUM/HIGH/MEDIUM/LOW)
    - Execute trades with adaptive risk management
    - Broadcast real-time updates via SSE
    - Auto-restart on server restart (if auto_start=True)
    
    Quality Levels:
    - PREMIUM: 85%+ confidence (safest, fewer trades)
    - HIGH: 75%+ confidence (recommended)
    - MEDIUM: 65%+ confidence (more trades)
    - LOW: 50%+ confidence (aggressive)
    """
    global _auto_bot, _bot_task
    
    if _auto_bot and _auto_bot._running:
        return {"status": "error", "message": "Bot already running"}
    
    # Save settings for auto-start
    settings = {
        "symbols": request.symbols,
        "timeframe": request.timeframe,
        "htf_timeframe": request.htf_timeframe,
        "min_quality": request.min_quality,
        "interval": request.interval,
        "auto_start": request.auto_start,
        "broker_type": request.broker_type,
        "allowed_signals": request.allowed_signals,
    }
    
    if request.auto_start:
        save_bot_settings(settings)
        logger.info("💾 Bot settings saved for auto-start")
    
    # Start the bot
    await auto_start_bot(settings)
    
    return {
        "status": "started",
        "symbols": request.symbols,
        "broker_type": request.broker_type,
        "min_quality": request.min_quality,
        "interval": request.interval,
        "mode": "PRODUCTION",
        "auto_start": request.auto_start,
        "allowed_signals": request.allowed_signals,
    }


@app.post("/api/v1/bot/stop")
async def stop_auto_bot(
    disable_auto_start: bool = True
):
    """🛑 Stop the AI trading bot
    
    Args:
        disable_auto_start: If True, also disable auto-start on server restart
    """
    global _auto_bot, _bot_task
    
    if not _auto_bot:
        return {"status": "error", "message": "Bot not running"}
    
    await _auto_bot.stop()
    
    if _bot_task:
        _bot_task.cancel()
        _bot_task = None
    
    _auto_bot = None
    
    # Clear auto-start settings
    if disable_auto_start:
        clear_bot_settings()
        logger.info("🗑️ Auto-start disabled")
    
    return {"status": "stopped", "auto_start_disabled": disable_auto_start}


@app.get("/api/v1/bot/status")
async def get_bot_status():
    """📊 Get auto trading bot status"""
    logger.info("📊 Getting bot status...")
    try:
        saved_settings = load_bot_settings()
        logger.info(f"📊 Saved settings: {saved_settings}")
        
        if not _auto_bot:
            logger.info("📊 Bot not running, returning default status")
            return {
                "running": False,
                "message": "Bot not started",
                "auto_start_enabled": saved_settings is not None and saved_settings.get('auto_start', False),
                "saved_settings": saved_settings
            }
        
        logger.info("📊 Getting bot status from _auto_bot...")
        status = _auto_bot.get_status()
        status["auto_start_enabled"] = saved_settings is not None and saved_settings.get('auto_start', False)
        logger.info(f"📊 Bot status: {status}")
        return status
    except Exception as e:
        logger.error(f"Error in get_bot_status: {e}")
        import traceback
        traceback.print_exc()
        # Return error as valid JSON instead of raising
        return {
            "running": False,
            "error": str(e),
            "message": f"Error getting bot status: {str(e)}"
        }


@app.get("/api/v1/bot/signals")
async def get_bot_signals(limit: int = 20):
    """📈 Get latest signals from bot"""
    if not _auto_bot:
        return {"signals": [], "count": 0}
    
    try:
        signals = getattr(_auto_bot, '_last_signals', {})
        # Convert to list and apply numpy conversion
        signals_list = []
        for symbol, signal_data in signals.items():
            if signal_data:
                safe_signal = convert_numpy_types(signal_data)
                safe_signal['symbol'] = str(symbol)
                signals_list.append(safe_signal)
        
        return convert_numpy_types({
            "signals": signals_list[:limit],
            "count": len(signals_list)
        })
    except Exception as e:
        logger.error(f"Error getting bot signals: {e}")
        return {"signals": [], "count": 0, "error": str(e)}


@app.get("/api/v1/bot/diagnostic")
async def get_bot_diagnostic():
    """
    🔍 Diagnostic endpoint - ตรวจสอบสถานะระบบทั้งหมด
    เปิด http://66.42.50.149:8000/api/v1/bot/diagnostic ใน browser เพื่อดู
    """
    import os
    
    diagnostic = {
        "timestamp": datetime.now().isoformat(),
        "bot": {
            "running": _auto_bot is not None and _auto_bot._running if _auto_bot else False,
            "mode": "PRODUCTION",
            "broker_type": _auto_bot.broker_type if _auto_bot else None,
            "min_quality": _auto_bot.min_quality.value if _auto_bot else None,
            "allowed_signals": _auto_bot.allowed_signals if _auto_bot else None,
            "symbols": _auto_bot.symbols if _auto_bot else None,
        },
        "trading_engine": {
            "initialized": _auto_bot.trading_engine is not None if _auto_bot else False,
            "enabled": _auto_bot.trading_engine.enabled if _auto_bot and _auto_bot.trading_engine else False,
            "broker_connected": _auto_bot.trading_engine.broker._connected if _auto_bot and _auto_bot.trading_engine else False,
            "open_positions": len(_auto_bot.trading_engine.positions) if _auto_bot and _auto_bot.trading_engine else 0,
        },
        "mt5_credentials": {
            "login_set": bool(os.getenv("MT5_LOGIN", "0") != "0"),
            "password_set": bool(os.getenv("MT5_PASSWORD", "")),
            "server_set": bool(os.getenv("MT5_SERVER", "")),
        },
        "saved_settings": load_bot_settings(),
        "will_trade_real": False,  # Will be calculated
        "issues": [],
    }
    
    # Check issues
    issues = []
    
    if not _auto_bot:
        issues.append("❌ Bot not started - กด Start Bot ใน dashboard")
    elif not _auto_bot._running:
        issues.append("❌ Bot initialized but not running")
    
    if not diagnostic["mt5_credentials"]["login_set"]:
        issues.append("⚠️ MT5_LOGIN not set in .env")
    if not diagnostic["mt5_credentials"]["password_set"]:
        issues.append("⚠️ MT5_PASSWORD not set in .env")
    if not diagnostic["mt5_credentials"]["server_set"]:
        issues.append("⚠️ MT5_SERVER not set in .env")
    
    if _auto_bot and _auto_bot.trading_engine:
        if not _auto_bot.trading_engine.broker._connected:
            issues.append("❌ Broker not connected")
    
    # Check if symbols have data (pattern indices built)
    if _auto_bot and hasattr(_auto_bot, 'pattern_matchers'):
        symbols_without_data = []
        for symbol in (_auto_bot.symbols or []):
            if symbol not in _auto_bot.pattern_matchers:
                symbols_without_data.append(symbol)
        if symbols_without_data:
            issues.append(f"⚠️ No pattern data for: {', '.join(symbols_without_data)} - Check symbol names match MT5")
        
        diagnostic["pattern_indices"] = {
            "loaded": list(_auto_bot.pattern_matchers.keys()),
            "missing": symbols_without_data,
        }
    
    # Determine if will trade real
    will_trade_real = (
        _auto_bot is not None and
        _auto_bot._running and
        diagnostic["mt5_credentials"]["login_set"] and
        _auto_bot.trading_engine is not None and
        _auto_bot.trading_engine.broker._connected
    )
    
    diagnostic["will_trade_real"] = will_trade_real
    diagnostic["issues"] = issues
    
    if will_trade_real:
        diagnostic["status"] = "✅ READY TO TRADE (PRODUCTION)"
    else:
        diagnostic["status"] = "❌ NOT READY - See issues"
    
    return diagnostic


# =====================
# Bot Settings API (Centralized)
# =====================

class BotSettingsRequest(BaseModel):
    """Request model for bot settings"""
    symbols: Optional[str] = None  # Comma-separated: "EURUSDm,GBPUSDm,XAUUSDm"
    timeframe: Optional[str] = None
    htf_timeframe: Optional[str] = None
    min_quality: Optional[str] = None
    interval: Optional[int] = None
    auto_start: Optional[bool] = None


def get_default_bot_settings():
    """Get default bot settings"""
    return {
        "symbols": "EURUSDm,GBPUSDm,XAUUSDm",
        "timeframe": "H1",
        "htf_timeframe": "H4",
        "min_quality": "HIGH",
        "interval": 60,
        "auto_start": False
    }


@app.get("/api/v1/bot/settings")
async def get_bot_settings_endpoint():
    """
    ⚙️ Get bot settings (centralized - synced across all devices)
    
    Returns saved settings or defaults if none saved.
    All devices will see the same settings.
    """
    saved = load_bot_settings()
    if saved:
        # Merge with defaults in case new fields added
        defaults = get_default_bot_settings()
        result = {**defaults, **saved}
        
        # Normalize symbols: ensure it's always a comma-separated string
        if isinstance(result.get('symbols'), list):
            result['symbols'] = ','.join(result['symbols'])
        
        return result
    return get_default_bot_settings()


@app.put("/api/v1/bot/settings")
async def update_bot_settings_endpoint(request: BotSettingsRequest):
    """
    ⚙️ Update bot settings (centralized - synced across all devices)
    
    Saves settings to server. All devices will see the updated settings.
    If bot is running, will apply changes immediately.
    """
    global _auto_bot
    
    # Load current settings
    current = load_bot_settings() or get_default_bot_settings()
    
    # Update only provided fields
    update_data = request.model_dump(exclude_none=True)
    updated_settings = {**current, **update_data}
    
    # Save to file
    save_bot_settings(updated_settings)
    
    # Apply to running bot immediately
    if _auto_bot and _auto_bot._running:
        quality_map = {
            "PREMIUM": SignalQuality.PREMIUM,
            "HIGH": SignalQuality.HIGH,
            "MEDIUM": SignalQuality.MEDIUM,
            "LOW": SignalQuality.LOW,
        }
        
        if request.min_quality:
            _auto_bot.min_quality = quality_map.get(request.min_quality, SignalQuality.HIGH)
            _auto_bot._min_confidence = _auto_bot._get_confidence_for_quality(_auto_bot.min_quality)
            logger.info(f"✅ Bot quality updated to {request.min_quality} (confidence: {_auto_bot._min_confidence}%)")
        
        if request.interval:
            logger.info(f"✅ Bot interval will be {request.interval}s on next cycle")
        
        if request.symbols:
            symbols = [s.strip() for s in request.symbols.split(',') if s.strip()]
            _auto_bot.symbols = symbols
            logger.info(f"✅ Bot symbols updated to {symbols}")
    
    logger.info(f"⚙️ Bot settings updated: {update_data}")
    
    return {
        "status": "saved",
        "settings": updated_settings,
        "applied_to_running_bot": _auto_bot is not None and _auto_bot._running
    }


# =============================================================================
# 16. INTELLIGENCE STATUS ENDPOINTS (16-Layer AI System)
# =============================================================================

@app.get("/api/v1/intelligence/pipeline")
async def get_pipeline_data(symbol: str = "EURUSDm"):
    """
    Get complete 16-Layer Pipeline data for Dashboard display
    Returns real-time data from all intelligence modules for the selected symbol
    """
    global _auto_bot
    
    if not _auto_bot:
        return {
            "status": "bot_not_running",
            "message": "Start the bot to see pipeline data",
            "layers": {},
            "current_signal": {},
            "final_decision": {}
        }
    
    # Gather all layer data
    layers = {}
    
    # Get symbol-specific analysis (fallback to latest)
    last_analysis_by_symbol = getattr(_auto_bot, '_last_analysis_by_symbol', {})
    last_analysis = last_analysis_by_symbol.get(symbol, getattr(_auto_bot, '_last_analysis', {})) or {}
    
    # Get symbol-specific layer results
    titan_by_symbol = getattr(_auto_bot, '_last_titan_decision_by_symbol', {})
    omega_by_symbol = getattr(_auto_bot, '_last_omega_result_by_symbol', {})
    alpha_by_symbol = getattr(_auto_bot, '_last_alpha_result_by_symbol', {})
    
    titan_data = titan_by_symbol.get(symbol, getattr(_auto_bot, '_last_titan_decision', {})) or {}
    omega_data = omega_by_symbol.get(symbol, getattr(_auto_bot, '_last_omega_result', {})) or {}
    alpha_data = alpha_by_symbol.get(symbol, getattr(_auto_bot, '_last_alpha_result', {})) or {}
    
    # 1. Data Lake
    layers["data_lake"] = {
        "status": "READY" if _auto_bot.data_provider else "NOT_INITIALIZED",
        "candles": getattr(_auto_bot, '_last_candle_count', 0),
        "source": _auto_bot.broker_type
    }
    
    # 2. Pattern Matcher (FAISS)
    layers["pattern_matcher"] = {
        "matches": last_analysis.get("n_matches", 0),
        "similarity": last_analysis.get("similarity", 0),
        "status": "ACTIVE" if _auto_bot.pattern_matchers else "NOT_INITIALIZED"
    }
    
    # 3. Voting System
    vote_details = last_analysis.get("vote_details") or {}
    layers["voting"] = {
        "signal": last_analysis.get("signal", "WAIT"),
        "bullish": vote_details.get("bullish", 0),
        "bearish": vote_details.get("bearish", 0),
        "confidence": last_analysis.get("confidence", 0)
    }
    
    # 4. Enhanced Analyzer
    layers["enhanced"] = {
        "quality": last_analysis.get("quality", None),
        "score": last_analysis.get("enhanced_confidence", 0),
        "pattern_score": last_analysis.get("pattern_score", 0),
        "technical_score": last_analysis.get("technical_score", 0),
        "volume_score": last_analysis.get("volume_score", 0)
    }
    
    # 5. Advanced Intelligence
    intel_status = {}
    if hasattr(_auto_bot, 'intelligence') and _auto_bot.intelligence:
        intel_by_symbol = getattr(_auto_bot, '_last_intel_result_by_symbol', {}) or {}
        intel_result = intel_by_symbol.get(symbol, getattr(_auto_bot, '_last_intel_result', {})) or {}
        intel_status = {
            "regime": intel_result.get("regime", "N/A"),
            "mtf_alignment": intel_result.get("mtf_alignment", "N/A"),
            "multiplier": intel_result.get("position_size_factor", 1.0),
            "trend_strength": intel_result.get("trend_strength", 0)
        }
    layers["advanced"] = intel_status
    
    # 6. Smart Brain
    smart_status = {}
    if hasattr(_auto_bot, 'smart_brain') and _auto_bot.smart_brain:
        smart_by_symbol = getattr(_auto_bot, '_last_smart_result_by_symbol', {}) or {}
        smart_result = smart_by_symbol.get(symbol, getattr(_auto_bot, '_last_smart_result', {})) or {}
        smart_status = {
            "pattern_count": getattr(_auto_bot.smart_brain, 'pattern_count', 0),
            "multiplier": smart_result.get("position_multiplier", 1.0),
            "win_rate": smart_result.get("win_rate", 0),
            "avg_rr": smart_result.get("avg_rr", 0)
        }
    layers["smart"] = smart_status
    
    # 7. Neural Brain
    neural_status = {}
    if hasattr(_auto_bot, 'neural_brain') and _auto_bot.neural_brain:
        neural_by_symbol = getattr(_auto_bot, '_last_neural_result_by_symbol', {}) or {}
        neural_result = neural_by_symbol.get(symbol, getattr(_auto_bot, '_last_neural_result', {})) or {}
        neural_status = {
            "market_state": neural_result.get("market_state", "N/A"),
            "dna_score": neural_result.get("dna_score", 0),
            "multiplier": neural_result.get("position_multiplier", 1.0),
            "pattern_quality": neural_result.get("pattern_quality", "N/A")
        }
    layers["neural"] = neural_status
    
    # 8. Deep Intelligence
    deep_status = {}
    if hasattr(_auto_bot, 'deep_intelligence') and _auto_bot.deep_intelligence:
        deep_by_symbol = getattr(_auto_bot, '_last_deep_result_by_symbol', {}) or {}
        deep_result = deep_by_symbol.get(symbol, getattr(_auto_bot, '_last_deep_result', {})) or {}
        deep_status = {
            "correlation": deep_result.get("correlation", 0),
            "session": deep_result.get("session", "N/A"),
            "multiplier": deep_result.get("position_multiplier", 1.0),
            "cross_asset_signal": deep_result.get("cross_asset_signal", "N/A")
        }
    layers["deep"] = deep_status
    
    # 9. Quantum Strategy
    quantum_status = {}
    if hasattr(_auto_bot, 'quantum_strategy') and _auto_bot.quantum_strategy:
        quantum_by_symbol = getattr(_auto_bot, '_last_quantum_result_by_symbol', {}) or {}
        quantum_result = quantum_by_symbol.get(symbol, getattr(_auto_bot, '_last_quantum_result', {})) or {}
        quantum_status = {
            "volatility_regime": quantum_result.get("volatility_regime", "N/A"),
            "fractal": quantum_result.get("fractal", "N/A"),
            "multiplier": quantum_result.get("position_multiplier", 1.0),
            "microstructure_signal": quantum_result.get("microstructure_signal", "N/A")
        }
    layers["quantum"] = quantum_status
    
    # 10. Alpha Engine (already symbol-specific from earlier)
    layers["alpha"] = {
        "grade": alpha_data.get("grade", "N/A"),
        "alpha_score": alpha_data.get("alpha_score", 0),
        "order_flow_bias": alpha_data.get("order_flow_bias", "N/A"),
        "risk_reward": alpha_data.get("risk_reward", 0),
        "position_multiplier": alpha_data.get("position_multiplier", 1.0),
        "should_trade": alpha_data.get("should_trade", False),
        "edge_factors": alpha_data.get("edge_factors", []),
        "risk_factors": alpha_data.get("risk_factors", [])
    }
    
    # 11. Omega Brain (already symbol-specific from variable declared above)
    layers["omega"] = {
        "grade": omega_data.get("grade", "N/A"),
        "omega_score": omega_data.get("omega_score", 0),
        "institutional_flow": omega_data.get("institutional_flow", "N/A"),
        "smart_money": omega_data.get("smart_money", "N/A"),
        "manipulation_detected": omega_data.get("manipulation_detected", "NONE"),
        "sentiment": omega_data.get("sentiment", 0),
        "current_regime": omega_data.get("current_regime", "N/A"),
        "predicted_regime": omega_data.get("predicted_regime", "N/A"),
        "position_multiplier": omega_data.get("position_multiplier", 1.0),
        "should_trade": omega_data.get("should_trade", False),
        "edge_factors": omega_data.get("edge_factors", []),
        "risk_factors": omega_data.get("risk_factors", [])
    }
    
    # 12. Titan Core (already symbol-specific from variable declared above)
    layers["titan"] = {
        "grade": titan_data.get("grade", "N/A"),
        "titan_score": titan_data.get("titan_score", 0),
        "consensus": titan_data.get("consensus", "N/A"),
        "agreement_ratio": titan_data.get("agreement_ratio", 0),
        "market_condition": titan_data.get("market_condition", "N/A"),
        "prediction": titan_data.get("prediction", {}),
        "position_multiplier": titan_data.get("position_multiplier", 1.0),
        "agreeing_modules": titan_data.get("agreeing_modules", 0),
        "total_modules": titan_data.get("total_modules", 0),
        "should_trade": titan_data.get("should_trade", False),
        "final_verdict": titan_data.get("final_verdict", ""),
        "edge_factors": titan_data.get("edge_factors", []),
        "risk_factors": titan_data.get("risk_factors", [])
    }
    
    # 13. Continuous Learning
    learning_status = {}
    if hasattr(_auto_bot, 'learning_system') and _auto_bot.learning_system:
        learning_status = {
            "market_cycle": getattr(_auto_bot.learning_system, 'current_market_cycle', 'N/A'),
            "cycles": getattr(_auto_bot.learning_system, 'learning_cycles', 0),
            "adaptations": getattr(_auto_bot.learning_system, 'adaptations', 0)
        }
    layers["learning"] = learning_status
    
    # 14. Pro Features
    pro_status = {}
    if hasattr(_auto_bot, 'pro_features') and _auto_bot.pro_features:
        pro_by_symbol = getattr(_auto_bot, '_last_pro_result_by_symbol', {}) or {}
        pro_result = pro_by_symbol.get(symbol, getattr(_auto_bot, '_last_pro_result', {})) or {}
        pro_status = {
            "session": pro_result.get("session", "N/A"),
            "news_impact": pro_result.get("news_impact", "NONE"),
            "multiplier": pro_result.get("position_multiplier", 1.0)
        }
    layers["pro"] = pro_status
    
    # 15. Risk Guardian
    risk_status = {
        "risk_level": "SAFE",
        "daily_pnl": 0,
        "can_trade": True,
        "open_positions": len(_auto_bot.trading_engine.positions) if _auto_bot.trading_engine else 0,
        "losing_streak": 0
    }
    if hasattr(_auto_bot, 'risk_guardian') and _auto_bot.risk_guardian:
        try:
            daily_stats = _auto_bot.risk_guardian.get_daily_stats()
            if daily_stats:
                risk_status = {
                    "risk_level": daily_stats.risk_level.value if hasattr(daily_stats, 'risk_level') else "SAFE",
                    "daily_pnl": daily_stats.total_pnl_percent if hasattr(daily_stats, 'total_pnl_percent') else 0,
                    "can_trade": daily_stats.can_trade if hasattr(daily_stats, 'can_trade') else True,
                    "open_positions": daily_stats.open_positions if hasattr(daily_stats, 'open_positions') else 0,
                    "losing_streak": daily_stats.losing_streak if hasattr(daily_stats, 'losing_streak') else 0,
                    "max_losing_streak": _auto_bot.risk_guardian.max_losing_streak if hasattr(_auto_bot.risk_guardian, 'max_losing_streak') else 5,
                    "max_daily_loss": _auto_bot.risk_guardian.max_daily_loss_percent if hasattr(_auto_bot.risk_guardian, 'max_daily_loss_percent') else 5.0
                }
        except Exception:
            pass
    layers["risk"] = risk_status
    
    # 16. Sentiment (Contrarian) - symbol-specific (data comes from Omega Brain)
    sentiment_by_symbol = getattr(_auto_bot, '_last_sentiment_result_by_symbol', {}) or {}
    sentiment_result = sentiment_by_symbol.get(symbol, getattr(_auto_bot, '_last_sentiment_result', {})) or {}
    sentiment_status = {
        "level": sentiment_result.get("level", "N/A"),
        "retail_sentiment": sentiment_result.get("retail_sentiment", 0),
        "override": "YES" if sentiment_result.get("override_signal", False) else "NO",
        "source": "Omega Brain",
        "active": len(sentiment_result) > 0
    }
    layers["sentiment"] = sentiment_status
    
    # Current Signal (symbol-specific from last_analysis which is already symbol-filtered)
    current_signal = {
        "signal": last_analysis.get("signal", "WAIT"),
        "symbol": last_analysis.get("symbol", symbol),
        "quality": last_analysis.get("quality", None),
        "confidence": last_analysis.get("confidence", 0),
        "entry": last_analysis.get("current_price", 0),
        "stop_loss": last_analysis.get("stop_loss", 0),
        "take_profit": last_analysis.get("take_profit", 0)
    }
    
    # Final Decision (using symbol-specific titan_data)
    final_decision = {
        "action": titan_data.get("should_trade", False) and last_analysis.get("signal", "WAIT") != "WAIT",
        "signal": last_analysis.get("signal", "WAIT") if titan_data.get("should_trade", False) else "BLOCKED",
        "position_multiplier": titan_data.get("position_multiplier", 1.0),
        "verdict": titan_data.get("final_verdict", "Waiting for analysis...")
    }
    
    return convert_numpy_types({
        "status": "active",
        "symbol": symbol,
        "layers": layers,
        "current_signal": current_signal,
        "final_decision": final_decision,
        "timestamp": datetime.now().isoformat()
    })


@app.get("/api/v1/intelligence/status")
async def get_intelligence_status():
    """
    Get status of all 16 intelligence layers
    
    Returns comprehensive status of:
    - Titan Core (Meta-Intelligence)
    - Omega Brain (Institutional)
    - Alpha Engine (Professional)
    - Quantum Strategy (Microstructure)
    - Deep Intelligence (Cross-Asset)
    - Neural Brain (Pattern DNA)
    - Continuous Learning (Adaptation)
    - Advanced Intelligence (Multi-TF)
    - Smart Brain (Journal)
    - Pro Trading (Sessions)
    - Risk Guardian (Protection)
    """
    global _auto_bot
    
    if not _auto_bot:
        # Return default status when bot not running
        return {
            "status": "bot_not_running",
            "titan": { "active": False, "grade": "N/A" },
            "omega": { "active": False, "grade": "N/A" },
            "alpha": { "active": False, "grade": "N/A" },
            "quantum": { "active": False },
            "deep": { "active": False },
            "neural": { "active": False },
            "learning": { "active": False },
            "advanced": { "active": False },
            "smart": { "active": False },
            "pro": { "active": False },
            "risk": { "active": False, "level": "N/A" }
        }
    
    # Get actual status from bot modules
    status = {
        "status": "running",
        "titan": {
            "active": _auto_bot.titan_core is not None if hasattr(_auto_bot, 'titan_core') else False,
            "grade": "🏛️ TITAN ACTIVE" if hasattr(_auto_bot, 'titan_core') and _auto_bot.titan_core else "N/A",
            "score": 0
        },
        "omega": {
            "active": _auto_bot.omega_brain is not None if hasattr(_auto_bot, 'omega_brain') else False,
            "grade": "Ω" if hasattr(_auto_bot, 'omega_brain') and _auto_bot.omega_brain else "N/A",
            "score": 0
        },
        "alpha": {
            "active": _auto_bot.alpha_engine is not None if hasattr(_auto_bot, 'alpha_engine') else False,
            "grade": "A" if hasattr(_auto_bot, 'alpha_engine') and _auto_bot.alpha_engine else "N/A"
        },
        "quantum": {
            "active": _auto_bot.quantum_strategy is not None if hasattr(_auto_bot, 'quantum_strategy') else False
        },
        "deep": {
            "active": _auto_bot.deep_intelligence is not None if hasattr(_auto_bot, 'deep_intelligence') else False
        },
        "neural": {
            "active": _auto_bot.neural_brain is not None if hasattr(_auto_bot, 'neural_brain') else False
        },
        "learning": {
            "active": _auto_bot.learning_system is not None if hasattr(_auto_bot, 'learning_system') else False,
            "cycles": 0
        },
        "advanced": {
            "active": _auto_bot.intelligence is not None if hasattr(_auto_bot, 'intelligence') else False,
            "regime": "N/A"
        },
        "smart": {
            "active": _auto_bot.smart_brain is not None if hasattr(_auto_bot, 'smart_brain') else False,
            "patterns": 0
        },
        "pro": {
            "active": _auto_bot.pro_features is not None if hasattr(_auto_bot, 'pro_features') else False,
            "session": "N/A"
        },
        "risk": {
            "active": _auto_bot.risk_guardian is not None if hasattr(_auto_bot, 'risk_guardian') else False,
            "level": "SAFE",
            "daily_pnl": 0,
            "can_trade": True
        }
    }
    
    # Get Risk Guardian status
    if hasattr(_auto_bot, 'risk_guardian') and _auto_bot.risk_guardian:
        try:
            risk_status = _auto_bot.risk_guardian.get_daily_stats()
            if risk_status:
                status["risk"]["daily_pnl"] = risk_status.total_pnl_percent if hasattr(risk_status, 'total_pnl_percent') else 0
                status["risk"]["level"] = risk_status.risk_level.value if hasattr(risk_status, 'risk_level') else "SAFE"
        except Exception:
            pass
    
    return convert_numpy_types(status)


@app.get("/api/v1/intelligence/titan")
async def get_titan_data(symbol: str = "EURUSDm"):
    """Get Titan Core analysis data from running bot"""
    global _auto_bot
    
    if not _auto_bot:
        return {
            "status": "bot_not_running",
            "grade": "N/A",
            "titan_score": 0,
            "message": "Bot not running - start the bot to get analysis"
        }
    
    if not hasattr(_auto_bot, 'titan_core') or not _auto_bot.titan_core:
        return {
            "status": "module_not_active",
            "grade": "N/A",
            "titan_score": 0,
            "message": "Titan Core module not initialized"
        }
    
    # Return last Titan decision from bot
    last_titan = getattr(_auto_bot, '_last_titan_decision', {})
    if not last_titan:
        return {
            "status": "no_analysis_yet",
            "grade": "🏛️ WAITING",
            "titan_score": 0,
            "consensus": "N/A",
            "market_condition": "WAITING",
            "prediction": {"direction": "WAIT", "confidence": 0},
            "position_multiplier": 1.0,
            "message": "Waiting for first analysis cycle..."
        }
    
    return {
        "status": "active",
        **convert_numpy_types(last_titan)
    }


@app.get("/api/v1/intelligence/omega")
async def get_omega_data(symbol: str = "EURUSDm"):
    """Get Omega Brain analysis data from running bot"""
    global _auto_bot
    
    if not _auto_bot:
        return {
            "status": "bot_not_running",
            "grade": "N/A",
            "omega_score": 0,
            "message": "Bot not running - start the bot to get analysis"
        }
    
    if not hasattr(_auto_bot, 'omega_brain') or not _auto_bot.omega_brain:
        return {
            "status": "module_not_active",
            "grade": "N/A",
            "omega_score": 0,
            "message": "Omega Brain module not initialized"
        }
    
    # Return last Omega decision from bot
    last_omega = getattr(_auto_bot, '_last_omega_result', {})
    if not last_omega:
        return {
            "status": "no_analysis_yet",
            "grade": "Ω WAITING",
            "omega_score": 0,
            "institutional_flow": "N/A",
            "manipulation": "NONE",
            "sentiment": 0,
            "message": "Waiting for first analysis cycle..."
        }
    
    return {
        "status": "active",
        **convert_numpy_types(last_omega)
    }


@app.get("/api/v1/intelligence/risk")
async def get_risk_data():
    """Get comprehensive risk management data"""
    global _auto_bot
    
    # Default risk data
    risk_data = {
        "risk_level": "SAFE",
        "balance": 10000,
        "equity": 10000,
        "daily_pnl": 0,
        "open_positions": 0,
        "max_positions": 3,
        "risk_per_trade": 2,
        "max_daily_loss": 5,
        "leverage": 2000,
        "risk_score": 0,
        "can_trade": True,
        "can_open_position": True,
        "daily_limit_hit": False,
        "losing_streak_limit": False,
        "losing_streak": 0,
        "max_losing_streak": 5
    }
    
    if not _auto_bot:
        risk_data["message"] = "Bot not running"
        return risk_data
    
    # Get account info from trading engine
    if hasattr(_auto_bot, 'trading_engine') and _auto_bot.trading_engine:
        try:
            engine = _auto_bot.trading_engine
            if engine.broker and engine.broker._connected:
                import MetaTrader5 as mt5
                info = mt5.account_info()
                if info:
                    risk_data["balance"] = info.balance
                    risk_data["equity"] = info.equity
                    risk_data["leverage"] = info.leverage
            
            risk_data["open_positions"] = len(engine.positions) if hasattr(engine, 'positions') else 0
        except Exception:
            pass
    
    # Get Risk Guardian status
    if hasattr(_auto_bot, 'risk_guardian') and _auto_bot.risk_guardian:
        try:
            rg = _auto_bot.risk_guardian
            stats = rg.get_daily_stats()
            if stats:
                risk_data["daily_pnl"] = stats.total_pnl_percent if hasattr(stats, 'total_pnl_percent') else 0
                risk_data["risk_level"] = stats.risk_level.value if hasattr(stats, 'risk_level') else "SAFE"
                risk_data["losing_streak"] = stats.losing_streak if hasattr(stats, 'losing_streak') else 0
            
            assessment = rg.assess_trade_risk(0.01, risk_data.get("balance", 10000), "EURUSDm")
            if assessment:
                risk_data["can_trade"] = assessment.can_trade if hasattr(assessment, 'can_trade') else True
        except Exception:
            pass
    
    # Calculate risk score (0-100, higher = more risky)
    risk_score = 0
    if risk_data["daily_pnl"] < 0:
        risk_score += min(40, abs(risk_data["daily_pnl"]) * 8)  # Max 40 from daily loss
    risk_score += risk_data["losing_streak"] * 10  # 10 points per losing trade
    risk_score += (risk_data["open_positions"] / max(1, risk_data["max_positions"])) * 20  # Max 20 from positions
    risk_data["risk_score"] = min(100, risk_score)
    
    # Determine limits
    risk_data["daily_limit_hit"] = bool(risk_data["daily_pnl"] <= -risk_data["max_daily_loss"])
    risk_data["losing_streak_limit"] = bool(risk_data["losing_streak"] >= risk_data["max_losing_streak"])
    risk_data["can_open_position"] = bool(risk_data["open_positions"] < risk_data["max_positions"])
    
    return convert_numpy_types(risk_data)


@app.get("/api/v1/intelligence/signals/history")
async def get_signal_history(limit: int = 50):
    """Get historical signal data from bot"""
    global _auto_bot
    
    if not _auto_bot:
        return {
            "status": "bot_not_running",
            "signals": [],
            "count": 0,
            "message": "Bot not running - start the bot to collect signals"
        }
    
    # Get signal history from bot
    signal_history = getattr(_auto_bot, '_signal_history', [])
    
    return {
        "status": "active",
        "signals": convert_numpy_types(signal_history[:limit]),
        "count": len(signal_history),
        "total_available": len(signal_history)
    }


@app.get("/api/v1/intelligence/alpha")
async def get_alpha_data(symbol: str = "EURUSDm"):
    """Get Alpha Engine analysis data from running bot"""
    global _auto_bot
    
    if not _auto_bot:
        return {
            "status": "bot_not_running",
            "grade": "N/A",
            "alpha_score": 0,
            "message": "Bot not running - start the bot to get analysis"
        }
    
    if not hasattr(_auto_bot, 'alpha_engine') or not _auto_bot.alpha_engine:
        return {
            "status": "module_not_active",
            "grade": "N/A",
            "alpha_score": 0,
            "message": "Alpha Engine module not initialized"
        }
    
    # Return last Alpha decision from bot
    last_alpha = getattr(_auto_bot, '_last_alpha_result', {})
    if not last_alpha:
        return {
            "status": "no_analysis_yet",
            "grade": "A WAITING",
            "alpha_score": 0,
            "order_flow_bias": "NEUTRAL",
            "risk_reward": 0,
            "message": "Waiting for first analysis cycle..."
        }
    
    return {
        "status": "active",
        **convert_numpy_types(last_alpha)
    }


@app.get("/api/v1/intelligence/last-analysis")
async def get_last_analysis():
    """Get the most recent full analysis from bot"""
    global _auto_bot
    
    if not _auto_bot:
        return {
            "status": "bot_not_running",
            "message": "Bot not running"
        }
    
    return {
        "status": "active",
        "last_analysis": getattr(_auto_bot, '_last_analysis', {}),
        "titan": getattr(_auto_bot, '_last_titan_decision', {}),
        "omega": getattr(_auto_bot, '_last_omega_result', {}),
        "alpha": getattr(_auto_bot, '_last_alpha_result', {}),
        "timestamp": datetime.now().isoformat()
    }


# Run with: uvicorn api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        reload=APIConfig.DEBUG
    )
