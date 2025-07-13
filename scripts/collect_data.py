
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from datetime import datetime

from seventrader.config import Config
from seventrader.data import EnhancedMarketDataManager

def run():
    """Runs the data collection for the 777 bot."""
    print("📥 Downloading historical data for backtesting...")

    # Load configuration
    config = Config()

    # Initialize the data manager
    data_manager = EnhancedMarketDataManager(config)

    # --- Data Collection Parameters ---
    symbol = "BTC/USDT"
    timeframe = "1h"
    start_date_str = "2023-01-01 00:00:00"
    end_date_str = "2023-12-31 23:59:59"
    # ----------------------------------

    # Convert string dates to milliseconds timestamp
    start_timestamp_ms = int(datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    end_timestamp_ms = int(datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)

    # Collect data
    data_manager.collect_all_data_parallel(start_timestamp_ms, end_timestamp_ms)

    print("✅ Data collection complete.")

if __name__ == "__main__":
    run()
