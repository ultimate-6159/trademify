"""
?? TRADEMIFY LOGGING CONFIGURATION
===================================
Enterprise-grade logging with rotation for 10+ year operation

Features:
- Log rotation (10MB per file, 100 backups)
- Daily rotation option
- JSON structured logging
- Console + File output
- Color-coded console output
- Memory-efficient handlers

Usage:
    from config.logging_config import setup_logging
    setup_logging()
"""

import os
import sys
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

# =====================================================
# ?? LOG DIRECTORIES
# =====================================================

def get_log_directory() -> Path:
    """Get the log directory path, create if not exists"""
    # Try multiple locations
    possible_paths = [
        Path("C:/trademify/logs"),
        Path(__file__).parent.parent.parent / "logs",
        Path.cwd() / "logs",
    ]
    
    for path in possible_paths:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            continue
    
    # Fallback to temp
    import tempfile
    temp_path = Path(tempfile.gettempdir()) / "trademify_logs"
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


# =====================================================
# ?? COLORED CONSOLE FORMATTER
# =====================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt, datefmt)
        self.use_colors = sys.stdout.isatty()  # Only use colors if terminal
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color if terminal supports it
        if self.use_colors:
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            
            # Color the message for errors and warnings
            if record.levelno >= logging.ERROR:
                record.msg = f"{self.COLORS['ERROR']}{record.msg}{self.RESET}"
            elif record.levelno >= logging.WARNING:
                record.msg = f"{self.COLORS['WARNING']}{record.msg}{self.RESET}"
        
        return super().format(record)


# =====================================================
# ?? JSON FORMATTER (for structured logging)
# =====================================================

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging (useful for log aggregation)"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data, ensure_ascii=False)


# =====================================================
# ?? LOGGING CONFIGURATION
# =====================================================

class LoggingConfig:
    """Logging configuration settings"""
    
    # File rotation settings
    MAX_BYTES = 10 * 1024 * 1024  # 10MB per file
    BACKUP_COUNT = 100            # Keep 100 backup files (1GB total)
    
    # Time-based rotation
    WHEN = 'midnight'             # Rotate at midnight
    INTERVAL = 1                  # Every 1 day
    TIME_BACKUP_COUNT = 30        # Keep 30 days of logs
    
    # Log levels
    CONSOLE_LEVEL = logging.INFO
    FILE_LEVEL = logging.DEBUG
    
    # Format strings
    CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Files
    MAIN_LOG_FILE = "trademify.log"
    ERROR_LOG_FILE = "errors.log"
    TRADE_LOG_FILE = "trades.log"
    JSON_LOG_FILE = "structured.jsonl"


def setup_logging(
    log_dir: Optional[Path] = None,
    console_level: int = LoggingConfig.CONSOLE_LEVEL,
    file_level: int = LoggingConfig.FILE_LEVEL,
    use_json: bool = False,
    use_time_rotation: bool = False,
) -> logging.Logger:
    """
    Setup enterprise-grade logging with rotation
    
    Args:
        log_dir: Custom log directory (default: auto-detect)
        console_level: Console logging level (default: INFO)
        file_level: File logging level (default: DEBUG)
        use_json: Enable JSON structured logging
        use_time_rotation: Use time-based rotation instead of size-based
    
    Returns:
        Root logger instance
    """
    # Get log directory
    log_path = log_dir or get_log_directory()
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # =====================================================
    # Console Handler (colorized)
    # =====================================================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter(
        fmt=LoggingConfig.CONSOLE_FORMAT,
        datefmt=LoggingConfig.DATE_FORMAT
    ))
    root_logger.addHandler(console_handler)
    
    # =====================================================
    # Main Log File Handler (rotating)
    # =====================================================
    main_log_file = log_path / LoggingConfig.MAIN_LOG_FILE
    
    if use_time_rotation:
        main_handler = TimedRotatingFileHandler(
            filename=str(main_log_file),
            when=LoggingConfig.WHEN,
            interval=LoggingConfig.INTERVAL,
            backupCount=LoggingConfig.TIME_BACKUP_COUNT,
            encoding='utf-8'
        )
    else:
        main_handler = RotatingFileHandler(
            filename=str(main_log_file),
            maxBytes=LoggingConfig.MAX_BYTES,
            backupCount=LoggingConfig.BACKUP_COUNT,
            encoding='utf-8'
        )
    
    main_handler.setLevel(file_level)
    main_handler.setFormatter(logging.Formatter(
        fmt=LoggingConfig.FILE_FORMAT,
        datefmt=LoggingConfig.DATE_FORMAT
    ))
    root_logger.addHandler(main_handler)
    
    # =====================================================
    # Error Log File Handler (errors only)
    # =====================================================
    error_log_file = log_path / LoggingConfig.ERROR_LOG_FILE
    
    error_handler = RotatingFileHandler(
        filename=str(error_log_file),
        maxBytes=LoggingConfig.MAX_BYTES,
        backupCount=50,  # Keep 50 error log backups
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        fmt=LoggingConfig.FILE_FORMAT,
        datefmt=LoggingConfig.DATE_FORMAT
    ))
    root_logger.addHandler(error_handler)
    
    # =====================================================
    # JSON Structured Log Handler (optional)
    # =====================================================
    if use_json:
        json_log_file = log_path / LoggingConfig.JSON_LOG_FILE
        
        json_handler = RotatingFileHandler(
            filename=str(json_log_file),
            maxBytes=LoggingConfig.MAX_BYTES,
            backupCount=LoggingConfig.BACKUP_COUNT,
            encoding='utf-8'
        )
        json_handler.setLevel(file_level)
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)
    
    # =====================================================
    # Configure third-party loggers (reduce noise)
    # =====================================================
    noisy_loggers = [
        'uvicorn',
        'uvicorn.error',
        'uvicorn.access',
        'fastapi',
        'httpx',
        'httpcore',
        'asyncio',
        'urllib3',
        'watchfiles',
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # MT5 logger - reduce but keep important messages
    logging.getLogger('MetaTrader5').setLevel(logging.INFO)
    
    # Log startup message
    root_logger.info(f"?? Logging initialized - Directory: {log_path}")
    root_logger.info(f"   Main log: {main_log_file}")
    root_logger.info(f"   Error log: {error_log_file}")
    if use_json:
        root_logger.info(f"   JSON log: {log_path / LoggingConfig.JSON_LOG_FILE}")
    
    return root_logger


def get_trade_logger() -> logging.Logger:
    """
    Get a dedicated logger for trade operations
    
    Logs to separate file for easy trade analysis
    """
    log_path = get_log_directory()
    trade_log_file = log_path / LoggingConfig.TRADE_LOG_FILE
    
    trade_logger = logging.getLogger('trademify.trades')
    
    # Only add handler if not already added
    if not trade_logger.handlers:
        handler = RotatingFileHandler(
            filename=str(trade_log_file),
            maxBytes=LoggingConfig.MAX_BYTES,
            backupCount=LoggingConfig.BACKUP_COUNT,
            encoding='utf-8'
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(message)s",
            datefmt=LoggingConfig.DATE_FORMAT
        ))
        trade_logger.addHandler(handler)
        trade_logger.setLevel(logging.INFO)
    
    return trade_logger


def log_trade(
    action: str,
    symbol: str,
    side: str,
    price: float,
    lot: float,
    sl: float = 0,
    tp: float = 0,
    pnl: float = 0,
    **extra
):
    """
    Log a trade event with structured data
    
    Args:
        action: OPEN, CLOSE, MODIFY, CANCEL
        symbol: Trading symbol
        side: BUY or SELL
        price: Entry/Exit price
        lot: Lot size
        sl: Stop loss
        tp: Take profit
        pnl: Profit/Loss (for close)
        **extra: Additional data
    """
    trade_logger = get_trade_logger()
    
    message = f"{action} | {symbol} | {side} | Price: {price:.2f} | Lot: {lot}"
    
    if sl > 0:
        message += f" | SL: {sl:.2f}"
    if tp > 0:
        message += f" | TP: {tp:.2f}"
    if pnl != 0:
        pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
        message += f" | PnL: {pnl_str}"
    
    for key, value in extra.items():
        message += f" | {key}: {value}"
    
    trade_logger.info(message)


# =====================================================
# ?? LOG CLEANUP UTILITIES
# =====================================================

def cleanup_old_logs(days: int = 30, log_dir: Optional[Path] = None):
    """
    Clean up log files older than specified days
    
    Args:
        days: Delete logs older than this many days
        log_dir: Log directory (default: auto-detect)
    """
    import time
    
    log_path = log_dir or get_log_directory()
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    
    deleted_count = 0
    deleted_size = 0
    
    for log_file in log_path.glob("*.log*"):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                size = log_file.stat().st_size
                log_file.unlink()
                deleted_count += 1
                deleted_size += size
        except Exception as e:
            logging.warning(f"Failed to delete {log_file}: {e}")
    
    if deleted_count > 0:
        logging.info(f"?? Cleaned up {deleted_count} old log files ({deleted_size / 1024 / 1024:.2f} MB)")


def get_log_stats() -> dict:
    """Get statistics about log files"""
    log_path = get_log_directory()
    
    total_size = 0
    file_count = 0
    oldest_file = None
    newest_file = None
    
    for log_file in log_path.glob("*.log*"):
        try:
            stat = log_file.stat()
            total_size += stat.st_size
            file_count += 1
            
            mtime = datetime.fromtimestamp(stat.st_mtime)
            if oldest_file is None or mtime < oldest_file[1]:
                oldest_file = (log_file.name, mtime)
            if newest_file is None or mtime > newest_file[1]:
                newest_file = (log_file.name, mtime)
        except Exception:
            continue
    
    return {
        "log_directory": str(log_path),
        "total_files": file_count,
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "oldest_file": {
            "name": oldest_file[0] if oldest_file else None,
            "date": oldest_file[1].isoformat() if oldest_file else None
        },
        "newest_file": {
            "name": newest_file[0] if newest_file else None,
            "date": newest_file[1].isoformat() if newest_file else None
        }
    }


# =====================================================
# ?? AUTO-SETUP ON IMPORT (optional)
# =====================================================

def auto_setup():
    """Auto-setup logging when module is imported"""
    # Only setup if not already configured
    if not logging.getLogger().handlers:
        setup_logging()


# Uncomment to auto-setup on import:
# auto_setup()


if __name__ == "__main__":
    # Test logging setup
    setup_logging(use_json=True)
    
    logger = logging.getLogger("test")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test trade logging
    log_trade(
        action="OPEN",
        symbol="XAUUSDm",
        side="BUY",
        price=2650.50,
        lot=0.01,
        sl=2640.00,
        tp=2670.00,
        confidence=85.5
    )
    
    # Print log stats
    stats = get_log_stats()
    print(f"\n?? Log Stats: {json.dumps(stats, indent=2)}")
