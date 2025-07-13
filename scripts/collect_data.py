
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seventrader.config import Config
from seventrader.data import EnhancedMarketDataManager

def run():
    """Runs the data collection for the 777 bot."""
    print("📥 Downloading historical data for backtesting...")

    # Load configuration
    config = Config()

    # Initialize the data manager
    data_manager = EnhancedMarketDataManager(config)

    # Collect data
    data_manager.collect_all_data_parallel()

    print("✅ Data collection complete.")

if __name__ == "__main__":
    run()
