
import sqlite3
import pandas as pd
import logging

class HistoricalDataManager:
    def __init__(self, db_path='trading_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)

    def fetch_ohlcv(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetches historical OHLCV data from the database for a specific date range."""
        try:
            table_name = f"{symbol.replace('/', '_')}_{timeframe}"
            query = f"SELECT * FROM {table_name} WHERE timestamp BETWEEN ? AND ?"
            df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
            if df.empty:
                return pd.DataFrame()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error loading data for {symbol} from {self.db_path}: {e}")
            return pd.DataFrame()

    def __del__(self):
        self.conn.close()
