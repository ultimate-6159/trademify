"""
Trademify API Security Module
============================
Authentication, Authorization, and Input Validation

Security Features:
- API Key Authentication
- JWT Token Support (optional)
- Rate Limiting
- Input Validation
- CORS Configuration
"""
import os
import re
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from functools import wraps

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel, validator, Field

logger = logging.getLogger(__name__)

# ===========================================
# API Key Configuration
# ===========================================

# Load API keys from environment
API_KEY_NAME = "X-API-Key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

def get_valid_api_keys() -> List[str]:
    """
    Get list of valid API keys from environment
    Set TRADEMIFY_API_KEYS as comma-separated keys
    Example: TRADEMIFY_API_KEYS=key1,key2,key3
    """
    keys_str = os.getenv("TRADEMIFY_API_KEYS", "")
    if not keys_str:
        # Default development key (CHANGE IN PRODUCTION!)
        logger.warning("⚠️ No API keys configured! Using development mode.")
        return ["dev-key-change-me-in-production"]
    
    return [k.strip() for k in keys_str.split(",") if k.strip()]


def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)


async def verify_api_key(
    api_key_header: str = Security(API_KEY_HEADER),
    api_key_query: str = Security(API_KEY_QUERY),
) -> str:
    """
    Verify API key from header or query parameter
    
    Usage in endpoint:
        @app.get("/protected")
        async def protected_endpoint(api_key: str = Depends(verify_api_key)):
            ...
    """
    api_key = api_key_header or api_key_query
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide via X-API-Key header or api_key query parameter.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    valid_keys = get_valid_api_keys()
    
    # Constant-time comparison to prevent timing attacks
    key_valid = False
    for valid_key in valid_keys:
        if secrets.compare_digest(api_key, valid_key):
            key_valid = True
            break
    
    if not key_valid:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )
    
    return api_key


async def optional_api_key(
    api_key_header: str = Security(API_KEY_HEADER),
    api_key_query: str = Security(API_KEY_QUERY),
) -> Optional[str]:
    """
    Optional API key verification - doesn't fail if no key provided
    Useful for endpoints that work differently with/without auth
    """
    api_key = api_key_header or api_key_query
    
    if not api_key:
        return None
    
    valid_keys = get_valid_api_keys()
    
    for valid_key in valid_keys:
        if secrets.compare_digest(api_key, valid_key):
            return api_key
    
    return None


# ===========================================
# CORS Configuration
# ===========================================

def get_cors_origins() -> List[str]:
    """
    Get allowed CORS origins from environment
    Set CORS_ORIGINS as comma-separated URLs
    Example: CORS_ORIGINS=http://localhost:3000,https://app.trademify.com
    
    Use "*" for development only!
    """
    origins_str = os.getenv("CORS_ORIGINS", "")
    
    if not origins_str:
        # Default: allow localhost for development
        logger.warning("⚠️ No CORS origins configured! Allowing localhost only.")
        return [
            "http://localhost:3000",
            "http://localhost:5173",  # Vite default
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ]
    
    if origins_str.strip() == "*":
        logger.warning("⚠️ CORS allowing all origins - NOT RECOMMENDED for production!")
        return ["*"]
    
    return [o.strip() for o in origins_str.split(",") if o.strip()]


# ===========================================
# Input Validation
# ===========================================

# Valid symbol patterns
# Note: Many brokers add suffixes to symbols (e.g., EURUSDm, EURUSD.pro, EURUSD-ECN)
SYMBOL_PATTERNS = {
    "forex": re.compile(r"^[A-Z]{6}[A-Za-z._-]*$"),  # EURUSD, EURUSDm, EURUSD.pro
    "crypto": re.compile(r"^[A-Z0-9]{2,10}USDT?[A-Za-z._-]*$"),  # BTCUSDT, BTCUSDTm
    "commodity": re.compile(r"^(XAU|XAG)(USD)?[A-Za-z._-]*$"),  # XAUUSD, XAUUSDm, XAGUSD
    "index": re.compile(r"^[A-Z]{2,5}\d*[A-Za-z._-]*$"),  # US30, NAS100, US30m
}

# Valid timeframes
VALID_TIMEFRAMES = {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"}

# Valid signal qualities
VALID_QUALITIES = {"PREMIUM", "HIGH", "MEDIUM", "LOW"}

# Valid signals
VALID_SIGNALS = {"STRONG_BUY", "BUY", "WAIT", "SELL", "STRONG_SELL"}


def validate_symbol(symbol: str) -> str:
    """
    Validate trading symbol format
    
    Raises:
        ValueError if symbol is invalid
    
    Returns:
        Cleaned symbol (uppercase, stripped)
    """
    if not symbol:
        raise ValueError("Symbol cannot be empty")
    
    # Clean and uppercase
    symbol = symbol.strip().upper()
    
    # Check length
    if len(symbol) < 3 or len(symbol) > 12:
        raise ValueError(f"Invalid symbol length: {symbol}")
    
    # Check against patterns
    for pattern_name, pattern in SYMBOL_PATTERNS.items():
        if pattern.match(symbol):
            return symbol
    
    # Allow known symbols explicitly
    known_symbols = {
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "XAUUSD", "XAGUSD",
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    }
    
    if symbol in known_symbols:
        return symbol
    
    raise ValueError(f"Invalid symbol format: {symbol}")


def validate_symbols(symbols: List[str]) -> List[str]:
    """Validate a list of symbols"""
    if not symbols:
        raise ValueError("At least one symbol required")
    
    if len(symbols) > 20:
        raise ValueError("Maximum 20 symbols allowed")
    
    validated = []
    for symbol in symbols:
        validated.append(validate_symbol(symbol))
    
    return validated


def validate_timeframe(timeframe: str) -> str:
    """Validate timeframe"""
    if not timeframe:
        raise ValueError("Timeframe cannot be empty")
    
    tf = timeframe.strip().upper()
    
    if tf not in VALID_TIMEFRAMES:
        raise ValueError(f"Invalid timeframe: {tf}. Valid: {VALID_TIMEFRAMES}")
    
    return tf


def validate_quality(quality: str) -> str:
    """Validate signal quality"""
    if not quality:
        raise ValueError("Quality cannot be empty")
    
    q = quality.strip().upper()
    
    if q not in VALID_QUALITIES:
        raise ValueError(f"Invalid quality: {q}. Valid: {VALID_QUALITIES}")
    
    return q


def validate_signal(signal: str) -> str:
    """Validate signal type"""
    if not signal:
        raise ValueError("Signal cannot be empty")
    
    s = signal.strip().upper()
    
    if s not in VALID_SIGNALS:
        raise ValueError(f"Invalid signal: {s}. Valid: {VALID_SIGNALS}")
    
    return s


def validate_risk_percent(risk: float) -> float:
    """Validate risk percentage"""
    if risk <= 0:
        raise ValueError("Risk must be positive")
    
    if risk > 10.0:
        raise ValueError("Risk cannot exceed 10%")
    
    return round(risk, 2)


def validate_interval(interval: int) -> int:
    """Validate check interval in seconds"""
    if interval < 10:
        raise ValueError("Interval must be at least 10 seconds")
    
    if interval > 3600:
        raise ValueError("Interval cannot exceed 3600 seconds (1 hour)")
    
    return interval


# ===========================================
# Validated Request Models
# ===========================================

class ValidatedStartBotRequest(BaseModel):
    """Validated request for starting bot - PRODUCTION MODE"""
    symbols: List[str] = Field(default=["EURUSDm", "GBPUSDm", "XAUUSDm"])
    timeframe: str = Field(default="H1")
    htf_timeframe: str = Field(default="H4")
    min_quality: str = Field(default="MEDIUM")
    interval: int = Field(default=60, ge=10, le=3600)
    auto_start: bool = Field(default=True)
    broker_type: str = Field(default="MT5")
    allowed_signals: List[str] = Field(default=["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"])
    
    @validator('symbols', pre=True)
    def validate_symbols_list(cls, v):
        if isinstance(v, str):
            v = [s.strip() for s in v.split(",")]
        return validate_symbols(v)
    
    @validator('timeframe')
    def validate_tf(cls, v):
        return validate_timeframe(v)
    
    @validator('htf_timeframe')
    def validate_htf(cls, v):
        return validate_timeframe(v)
    
    @validator('min_quality')
    def validate_qual(cls, v):
        return validate_quality(v)
    
    @validator('broker_type')
    def validate_broker(cls, v):
        v = v.strip().upper()
        if v not in {"MT5", "BINANCE"}:
            raise ValueError("broker_type must be MT5 or BINANCE")
        return v
    
    @validator('allowed_signals', pre=True)
    def validate_signals_list(cls, v):
        if isinstance(v, str):
            v = [s.strip() for s in v.split(",")]
        return [validate_signal(s) for s in v]


class ValidatedTradeRequest(BaseModel):
    """Validated request for manual trade"""
    symbol: str
    side: str
    quantity: float = Field(gt=0, le=100)  # Max 100 lots
    order_type: str = Field(default="MARKET")
    price: Optional[float] = Field(default=None, gt=0)
    stop_loss: float = Field(..., gt=0)  # REQUIRED!
    take_profit: Optional[float] = Field(default=None, gt=0)
    
    @validator('symbol')
    def validate_sym(cls, v):
        return validate_symbol(v)
    
    @validator('side')
    def validate_side(cls, v):
        v = v.strip().upper()
        if v not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        return v
    
    @validator('order_type')
    def validate_order_type(cls, v):
        v = v.strip().upper()
        if v not in {"MARKET", "LIMIT", "STOP"}:
            raise ValueError("order_type must be MARKET, LIMIT, or STOP")
        return v


# ===========================================
# Rate Limiting (Simple In-Memory)
# ===========================================

class RateLimiter:
    """
    Simple in-memory rate limiter
    
    For production, consider using Redis-based rate limiting
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                t for t in self.requests[client_id] if t > minute_ago
            ]
        else:
            self.requests[client_id] = []
        
        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Add request
        self.requests[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        if client_id not in self.requests:
            return self.requests_per_minute
        
        recent = [t for t in self.requests[client_id] if t > minute_ago]
        return max(0, self.requests_per_minute - len(recent))


# Global rate limiter
rate_limiter = RateLimiter(requests_per_minute=120)


async def check_rate_limit(request: Request):
    """
    Rate limit middleware dependency
    
    Usage:
        @app.get("/endpoint")
        async def endpoint(request: Request, _: None = Depends(check_rate_limit)):
            ...
    """
    client_id = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making more requests.",
            headers={"Retry-After": "60"},
        )


# ===========================================
# Security Headers Middleware
# ===========================================

def add_security_headers(response):
    """Add security headers to response"""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response
