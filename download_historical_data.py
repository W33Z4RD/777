import sys
import os
from datetime import datetime, timedelta
import time
import pandas as pd

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from seventrader.config import Config
from seventrader.data import EnhancedMarketDataManager

def download_all_data():
    """
    Downloads all available historical OHLCV data for all symbols and timeframes in the config.
    """
    config = Config()
    data_manager = EnhancedMarketDataManager(config)
    
    for symbol in config.SYMBOLS:
        for timeframe in config.TIMEFRAMES:
            start_date_str = "2015-01-01"
            end_date_str = datetime.now().strftime('%Y-%m-%d')
            
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

            print(f"Starting data download for {symbol} from {start_date_str} to {end_date_str} for timeframe {timeframe}")

            total_days = (end_date - start_date).days
            downloaded_days = 0

            current_date = start_date
            while current_date < end_date:
                since = int(current_date.timestamp() * 1000)
                print(f"Fetching data for {symbol} ({timeframe}) starting from {current_date.strftime('%Y-%m-%d')}")
                try:
                    data = data_manager.exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                    
                    if not data:
                        print(f"No more data found for {symbol} ({timeframe}) after {current_date.strftime('%Y-%m-%d')}. Stopping.")
                        break

                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    if df.empty:
                        print(f"No data for {current_date.strftime('%Y-%m-%d')}")
                        # If no data is found, we jump a larger interval to speed up the process
                        # as exchanges usually don't have data for very old dates.
                        current_date += timedelta(days=30)
                        continue

                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    data_manager.db.save_market_data(symbol, timeframe, df)

                    latest_timestamp_in_batch = df.index[-1]
                    print(f"Saved {len(df)} data points. Latest timestamp: {latest_timestamp_in_batch}")

                    downloaded_days = (latest_timestamp_in_batch - start_date).days
                    progress = (downloaded_days / total_days) * 100
                    print(f"Progress: {progress:.2f}%")

                    current_date = latest_timestamp_in_batch + timedelta(milliseconds=1)

                    time.sleep(data_manager.exchange.rateLimit / 1000)

                except Exception as e:
                    print(f"An error occurred: {e}")
                    time.sleep(5)

            print(f"Data download complete for {symbol} ({timeframe}).")


if __name__ == "__main__":
    download_all_data()