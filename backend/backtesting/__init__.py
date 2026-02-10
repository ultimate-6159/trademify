"""
Trademify Backtesting Engine
ระบบ Backtest ย้อนหลังสูงสุด 10 ปี - รองรับทุกขนาดบัญชี $500 - $5,000,000
"""

from .backtest_engine import (
    BacktestEngine, 
    BacktestResult, 
    BacktestConfig,
    RiskProfile,
    get_dynamic_risk_settings
)
from .data_loader import HistoricalDataLoader
from .report_generator import BacktestReporter

__all__ = [
    'BacktestEngine',
    'BacktestResult', 
    'BacktestConfig',
    'RiskProfile',
    'get_dynamic_risk_settings',
    'HistoricalDataLoader',
    'BacktestReporter'
]
