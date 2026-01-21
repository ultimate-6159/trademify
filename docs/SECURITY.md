# Trademify Security Documentation

## Overview

This document describes the security measures implemented in Trademify to protect the trading system from unauthorized access and malicious attacks.

## Security Features Implemented

### 1. API Key Authentication

All sensitive endpoints require API Key authentication:

**How to Authenticate:**
```bash
# Via Header (Recommended)
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/bot/start

# Via Query Parameter (Alternative)
curl "http://localhost:8000/api/v1/bot/start?api_key=your-api-key"
```

**Configuration:**
Set `TRADEMIFY_API_KEYS` environment variable with comma-separated keys:
```bash
TRADEMIFY_API_KEYS=key1,key2,key3
```

**Generate Secure Keys:**
```python
import secrets
print(secrets.token_urlsafe(32))
```

### 2. Protected Endpoints

| Endpoint | Method | Auth Required |
|----------|--------|---------------|
| `/api/v1/bot/start` | POST | ✅ |
| `/api/v1/bot/stop` | POST | ✅ |
| `/api/v1/bot/settings` | PUT | ✅ |
| `/api/v1/trading/start` | POST | ✅ |
| `/api/v1/trading/stop` | POST | ✅ |
| `/api/v1/trading/pause` | POST | ✅ |
| `/api/v1/trading/resume` | POST | ✅ |
| `/api/v1/trading/settings` | PUT | ✅ |
| `/api/v1/trading/positions` | POST | ✅ |
| `/api/v1/trading/positions/{id}` | PUT/DELETE | ✅ |
| `/api/v1/trading/signal` | POST | ✅ |

### 3. CORS Configuration

CORS is now configurable via environment variables:

```bash
# Development (localhost only - default)
# No configuration needed

# Production (specific domains)
CORS_ORIGINS=https://app.trademify.com,https://admin.trademify.com

# ⚠️ NOT RECOMMENDED for production
CORS_ORIGINS=*
```

### 4. Input Validation

All inputs are validated using Pydantic validators:

- **Symbols**: Must match forex/crypto/commodity patterns
- **Timeframes**: Must be M1, M5, M15, M30, H1, H4, D1, W1, MN1
- **Quality**: Must be PREMIUM, HIGH, MEDIUM, LOW
- **Signals**: Must be STRONG_BUY, BUY, WAIT, SELL, STRONG_SELL
- **Risk**: Must be 0-10%
- **Intervals**: Must be 10-3600 seconds

### 5. Mandatory Stop Loss

**All trades MUST have a Stop Loss.** If not provided:
- Auto-calculated at 2% from entry price
- Direction validated (SL below price for BUY, above for SELL)
- Invalid SL direction causes trade rejection

### 6. Security Headers

All responses include security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`

### 7. Rate Limiting

In-memory rate limiting at 120 requests/minute per IP.

## Production Checklist

### Required Before Going Live:

- [ ] Generate secure API keys
- [ ] Set `TRADEMIFY_API_KEYS` environment variable
- [ ] Set `CORS_ORIGINS` to your specific domains
- [ ] Remove development keys from any configs
- [ ] Enable HTTPS (use reverse proxy like nginx)
- [ ] Review and test all protected endpoints

### Environment Variables:

```bash
# Required for Production
TRADEMIFY_API_KEYS=your-32-char-secure-key-here
CORS_ORIGINS=https://your-domain.com

# Broker Credentials (keep secure!)
MT5_LOGIN=your-login
MT5_PASSWORD=your-password
MT5_SERVER=your-server

# OR for Binance
BINANCE_API_KEY=your-key
BINANCE_API_SECRET=your-secret
```

## Security Module Location

All security code is centralized in: `/backend/api/security.py`

This module provides:
- `verify_api_key()` - FastAPI dependency for auth
- `get_cors_origins()` - CORS configuration
- `check_rate_limit()` - Rate limiting dependency
- `ValidatedStartBotRequest` - Validated bot start request
- `ValidatedTradeRequest` - Validated trade request
- Input validators for symbols, timeframes, etc.

## Reporting Security Issues

If you discover a security vulnerability, please report it privately.
Do not create public issues for security vulnerabilities.
