
import sys
import os
import joblib
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from seventrader.config import Config
from seventrader.data import DatabaseManager
from seventrader.signals import AdvancedTechnicalAnalysis
from seventrader.models.models import SimpleMLModel

def prepare_features_and_labels(config, ta_engine, df):
    """Prepares features and labels from a given dataframe."""
    features_list = []
    labels = []
    label_lookahead = 12  # e.g., 12 hours on a 1h timeframe

    for i in range(config.LOOKBACK_PERIODS, len(df) - label_lookahead):
        historical_df = df.iloc[i - config.LOOKBACK_PERIODS : i]
        indicators = ta_engine.calculate_advanced_indicators(historical_df)
        
        if not indicators:
            continue

        # Prepare features
        feature_keys = [
            'price_change', 'volume_ratio', 'volatility', 'rsi', 'macd_histogram',
            'bb_position', 'trend_alignment', 'trend_strength', 'stoch_k', 'adx'
        ]
        features = {key: indicators.get(key) for key in feature_keys}
        
        if any(value is None for value in features.values()):
            continue

        # Generate label
        future_price = df['close'].iloc[i + label_lookahead]
        current_price = indicators['current_price']
        price_increase_threshold = 0.01
        label = 1 if (future_price - current_price) / current_price > price_increase_threshold else 0
        
        features_list.append(features)
        labels.append(label)

    if not features_list:
        return pd.DataFrame(), pd.Series()

    return pd.DataFrame(features_list), pd.Series(labels)

def run():
    """Runs the model training process."""
    print("🚀 Starting model training process...")

    config = Config()
    db_manager = DatabaseManager()
    ta_engine = AdvancedTechnicalAnalysis(config)
    model = SimpleMLModel()

    # --- Training Parameters ---
    symbol = "BTC/USDT"
    timeframe = "1h"
    start_date = "2023-01-01 00:00:00"
    end_date = "2023-12-31 23:59:59"
    model_filename = "trained_model.joblib"
    # ---------------------------

    print(f"Loading data for {symbol} from {start_date} to {end_date}...")
    df = db_manager.load_market_data_for_backtest(symbol, timeframe, start_date, end_date)

    if df.empty:
        print(f"No data found for {symbol}. Please run data collection first.")
        return

    print("Preparing features and labels for the model...")
    X, y = prepare_features_and_labels(config, ta_engine, df)

    if X.empty:
        print("Could not generate features for model training. Exiting.")
        return

    print(f"Training model on {len(X)} samples...")
    model.train(X, y)

    print(f"Saving trained model to {model_filename}...")
    joblib.dump(model, model_filename)

    print("✅ Model training complete.")

if __name__ == "__main__":
    run()
