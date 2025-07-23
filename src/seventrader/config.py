
import os
import json
import logging
import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot_advanced.log'),
        logging.StreamHandler()
    ]
)

# Enhanced Configuration
@dataclass
class Config:
    # Exchange Configuration
    EXCHANGE_NAME: str = os.getenv("EXCHANGE_NAME", "binance")
    API_KEY: str = os.getenv("EXCHANGE_API_KEY", "")
    API_SECRET: str = os.getenv("EXCHANGE_API_SECRET", "")
    PASSPHRASE: str = os.getenv("EXCHANGE_PASSPHRASE", "")
    SANDBOX: bool = os.getenv("EXCHANGE_SANDBOX", "false").lower() == "true"
    
    # Expanded Trading Universe
    SYMBOLS: List[str] = field(default_factory=lambda: json.loads(os.getenv("SYMBOLS", json.dumps([
    "EOS/USDC",
    "RSR/USDC",
    "ZRO/USDC",
    "MASK/USDC",
    "LPT/USDC",
    "KAITO/USDC",
    "W/USDC",
    "ACX/USDC",
    "PENDLE/USDC",
    "LTC/USDC",
    "ONDO/USDC",
    "ETH/USDT",
    "WIF/USDC",
    "TRUMP/USDC",
    "BTC/USDT",
    "ENA/USDC",
    "AVAX/USDC",
    "SEI/USDC",
    "UNI/USDC",
    "SHIB/USDC",
    "AAVE/USDC",
    "CRV/USDC",
    "HBAR/USDT",
    "TAO/USDC",
    "DOT/USDC",
    "ALGO/USDC",
    "FET/USDC",
    "ARB/USDC"
]))))
    
    # Multi-timeframe analysis
    TIMEFRAMES: List[str] = field(default_factory=lambda: json.loads(os.getenv("TIMEFRAMES", '["15m", "1h", "4h", "1d", "1w"]')))
    PRIMARY_TIMEFRAME: str = "1h"
    CONFIRMATION_TIMEFRAMES: List[str] = field(default_factory=lambda: ["4h", "1d"])
    
    LOOKBACK_PERIODS: int = int(os.getenv("LOOKBACK_PERIODS", "500"))
    UPDATE_INTERVAL: int = int(os.getenv("UPDATE_INTERVAL", "180"))
    
    # Technical Indicators
    RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
    RSI_OVERSOLD: int = int(os.getenv("RSI_OVERSOLD", "25"))
    RSI_OVERBOUGHT: int = int(os.getenv("RSI_OVERBOUGHT", "75"))
    MACD_FAST: int = int(os.getenv("MACD_FAST", "12"))
    MACD_SLOW: int = int(os.getenv("MACD_SLOW", "26"))
    MACD_SIGNAL: int = int(os.getenv("MACD_SIGNAL", "9"))
    BB_PERIOD: int = int(os.getenv("BB_PERIOD", "20"))
    BB_STD: float = float(os.getenv("BB_STD", "2.0"))
    EMA_FAST: int = int(os.getenv("EMA_FAST", "8"))
    EMA_MEDIUM: int = int(os.getenv("EMA_MEDIUM", "21"))
    EMA_SLOW: int = int(os.getenv("EMA_SLOW", "50"))
    EMA_TREND: int = int(os.getenv("EMA_TREND", "200"))
    STOCH_K: int = int(os.getenv("STOCH_K", "14"))
    STOCH_D: int = int(os.getenv("STOCH_D", "3"))
    STOCH_SMOOTH: int = int(os.getenv("STOCH_SMOOTH", "3"))
    ADX_PERIOD: int = int(os.getenv("ADX_PERIOD", "14"))
    ADX_TREND_THRESHOLD: float = float(os.getenv("ADX_TREND_THRESHOLD", "25"))
    ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", "14"))
    VOLUME_SMA: int = int(os.getenv("VOLUME_SMA", "20"))
    VOLUME_SPIKE_THRESHOLD: float = float(os.getenv("VOLUME_SPIKE_THRESHOLD", "2.0"))
    
    # Signal Thresholds
    BUY_THRESHOLD: float = float(os.getenv("BUY_THRESHOLD", "0.20"))
    SELL_THRESHOLD: float = float(os.getenv("SELL_THRESHOLD", "-0.20"))
    STRONG_SIGNAL_THRESHOLD: float = float(os.getenv("STRONG_SIGNAL_THRESHOLD", "0.85"))
    
    # Market Regime Detection
    VOLATILITY_LOOKBACK: int = int(os.getenv("VOLATILITY_LOOKBACK", "30"))
    TREND_STRENGTH_PERIOD: int = int(os.getenv("TREND_STRENGTH_PERIOD", "20"))
    
    # Risk Management
    STOP_LOSS_ATR_MULTIPLIER: float = float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", "2.5"))
    TAKE_PROFIT_ATR_MULTIPLIER: float = float(os.getenv("TAKE_PROFIT_ATR_MULTIPLIER", "5.0"))
    POSITION_SIZE_PCT: float = float(os.getenv("POSITION_SIZE_PCT", "0.02"))
    
    # Threading
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "8"))

    # Telegram Notifications
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    NOTIFICATION_COOLDOWN_SECONDS: int = int(os.getenv("NOTIFICATION_COOLDOWN_SECONDS", "3600"))
