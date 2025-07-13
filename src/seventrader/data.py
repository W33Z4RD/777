
import os
import json
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt

from .config import Config

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

class DatabaseManager:
    def __init__(self, db_path="trading_bot_advanced.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                timeframe TEXT,
                volatility REAL,
                volume_sma REAL,
                UNIQUE(symbol, timestamp, timeframe)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                signal_type TEXT,
                strength REAL,
                confidence REAL,
                timeframe TEXT,
                indicators TEXT,
                price REAL,
                volume REAL,
                market_regime TEXT,
                trend_direction TEXT,
                volatility_state TEXT,
                executed BOOLEAN DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_market_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        try:
            conn = sqlite3.connect(self.db_path)
            df_copy = df.copy()
            if hasattr(df_copy.index[0], 'strftime'):
                df_copy['timestamp'] = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
            else:
                df_copy['timestamp'] = df_copy.index.astype(str)
            
            df_copy['symbol'] = symbol
            df_copy['timeframe'] = timeframe
            df_copy['volatility'] = df_copy.get('volatility', 0)
            df_copy['volume_sma'] = df_copy.get('volume_sma', 0)
            
            columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                      'timeframe', 'volatility', 'volume_sma']
            df_to_save = df_copy[columns]
            
            data_to_insert = [tuple(x) for x in df_to_save.to_numpy()]
            
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, open, high, low, close, volume, timeframe, volatility, volume_sma)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_to_insert)
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error saving market data: {e}")

    def save_signal_data(self, signal: dict):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Convert indicators dict to JSON string
            indicators_json = json.dumps(signal.get('indicators', {}), cls=NumpyJSONEncoder)

            data_to_insert = (
                signal.get('symbol'),
                signal.get('timestamp').strftime('%Y-%m-%d %H:%M:%S'),
                signal.get('type'),
                signal.get('strength'),
                signal.get('confidence'),
                signal.get('timeframe'),
                indicators_json,
                signal.get('price'),
                signal.get('volume'),
                signal.get('market_regime'),
                signal.get('trend_direction'),
                signal.get('volatility_state')
            )

            cursor.execute('''
                INSERT INTO signals (
                    symbol, timestamp, signal_type, strength, confidence, timeframe,
                    indicators, price, volume, market_regime, trend_direction, volatility_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_to_insert)

            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error saving signal data: {e}")

    def load_market_data_for_backtest(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT timestamp, open, high, low, close, volume 
                FROM market_data 
                WHERE symbol = ? AND timeframe = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, start_date, end_date))
            conn.close()

            if df.empty:
                return pd.DataFrame()

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error loading backtest data for {symbol}: {e}")
            return pd.DataFrame()

class EnhancedMarketDataManager:
    def __init__(self, config: Config):
        self.config = config
        self.exchange = self._initialize_exchange()
        self.db = DatabaseManager()
    
    def _initialize_exchange(self):
        """Initialize exchange with enhanced error handling"""
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
            exchange.load_markets()
            logging.info("Connected to Binance for enhanced data fetching")
            return exchange
        except Exception as e:
            logging.error(f"Failed to initialize exchange: {e}")
            raise
    
    def fetch_ohlcv_with_indicators(self, symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
        """Fetch OHLCV data and add basic indicators"""
        try:
            if limit is None:
                limit = self.config.LOOKBACK_PERIODS
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                logging.warning(f"No data received for {symbol} {timeframe}")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add basic indicators for database storage
            returns = df['close'].pct_change()
            df['volatility'] = returns.rolling(self.config.VOLATILITY_LOOKBACK).std() * np.sqrt(252)
            df['volume_sma'] = df['volume'].rolling(self.config.VOLUME_SMA).mean()
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
        """Maintain compatibility with existing code"""
        return self.fetch_ohlcv_with_indicators(symbol, timeframe, limit)
    
    def collect_all_data_parallel(self):
        """Collect data for all symbols in parallel"""
        logging.info("Starting parallel data collection...")
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = []
            
            for symbol in self.config.SYMBOLS:
                for timeframe in self.config.TIMEFRAMES:
                    future = executor.submit(self._collect_single_data, symbol, timeframe)
                    futures.append(future)
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        symbol, timeframe, df = result
                        self.db.save_market_data(symbol, timeframe, df)
                except Exception as e:
                    logging.error(f"Error in parallel data collection: {e}")
        
        logging.info("Parallel data collection completed")
    
    def _collect_single_data(self, symbol: str, timeframe: str) -> Optional[Tuple[str, str, pd.DataFrame]]:
        """Collect data for a single symbol/timeframe pair"""
        try:
            df = self.fetch_ohlcv_with_indicators(symbol, timeframe)
            if not df.empty:
                return symbol, timeframe, df
            return None
        except Exception as e:
            logging.error(f"Error collecting data for {symbol} {timeframe}: {e}")
            return None
