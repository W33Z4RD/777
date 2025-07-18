
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seventrader.config import Config
from seventrader.backtesting import Backtester

def run():
    """Runs the backtest for the 777 bot."""
    print("🚀 Starting backtest for the new 777PROJECT bot...")

    # Load configuration
    config = Config()

    # --- Backtest Parameters ---
    symbol = "BTC/USDT"
    start_date = "2023-01-01"
    end_date = "2025-12-31"
    fee_rate = 0.001  # 0.1% trading fee
    slippage_pct = 0.0005 # 0.05% slippage
    # ---------------------------

    # Initialize the backtester with fees and slippage
    backtester = Backtester(config, fee_rate=fee_rate, slippage_pct=slippage_pct)

    # Run the backtest
    backtester.run(start_date, end_date, symbol)

    print("✅ Backtest complete.")

if __name__ == "__main__":
    run()
