# tests/test_signals.py

import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from seventrader.config import Config
from seventrader.signals import AdvancedSignalGenerator

class TestAdvancedSignalGenerator(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.config = Config()
        # Override config for testing purposes
        self.config.LOOKBACK_PERIODS = 100
        self.config.MODEL_PATH = 'test_model.joblib' # Use a dummy path
        self.signal_generator = AdvancedSignalGenerator(self.config)
        # Crucially, disable the model for this unit test to isolate indicator logic
        self.signal_generator.model = None

        # Mock the DatabaseManager to prevent actual DB calls
        self.patcher_db = patch('seventrader.signals.DatabaseManager')
        self.MockDatabaseManager = self.patcher_db.start()
        self.signal_generator.db = self.MockDatabaseManager()

    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.config.MODEL_PATH):
            os.remove(self.config.MODEL_PATH)
        self.patcher_db.stop()

    def _create_mock_indicators(self, rsi_val, macd_hist_val, bb_pos_val, ema_fast_val, ema_slow_val, stoch_k_val, stoch_d_val, adx_val, trend_alignment_val, trend_strength_val, volatility_val, volume_ratio_val):
        """Creates a mock indicators dictionary with specific values."""
        return {
            'current_price': 100.0,
            'previous_close': 99.0,
            'price_change': 0.01,
            'current_volume': 10000,
            'volume_sma': 5000,
            'volume_ratio': volume_ratio_val,
            'volume_spike': volume_ratio_val > self.config.VOLUME_SPIKE_THRESHOLD,
            'volatility': volatility_val,
            'volatility_percentile': 50,
            'rsi': rsi_val,
            'rsi_sma': rsi_val,
            'macd': 0.1 if macd_hist_val > 0 else -0.1,
            'macd_signal': 0,
            'macd_histogram': macd_hist_val,
            'macd_crossover': macd_hist_val > 0,
            'macd_crossunder': macd_hist_val < 0,
            'bb_upper': 105,
            'bb_middle': 100,
            'bb_lower': 95,
            'bb_position': bb_pos_val,
            'bb_squeeze': False,
            'ema_fast': ema_fast_val,
            'ema_medium': (ema_fast_val + ema_slow_val) / 2,
            'ema_slow': ema_slow_val,
            'ema_trend': ema_slow_val - 5,
            'trend_alignment': trend_alignment_val,
            'trend_strength': trend_strength_val,
            'stoch_k': stoch_k_val,
            'stoch_d': stoch_d_val,
            'adx': adx_val,
            'strong_trend': adx_val > self.config.ADX_TREND_THRESHOLD,
            'atr': 1.0,
            'atr_percentage': 0.01,
            'support': 90,
            'resistance': 110,
            'market_regime': 'normal_volatility_trending' if adx_val > self.config.ADX_TREND_THRESHOLD else 'normal_volatility_ranging'
        }

    @patch.object(AdvancedSignalGenerator, 'ta')
    def test_generate_strong_buy_signal(self, mock_ta):
        """Test that a strong buy signal is generated under ideal bullish conditions."""
        mock_indicators = self._create_mock_indicators(
            rsi_val=20, macd_hist_val=0.5, bb_pos_val=0.05, 
            ema_fast_val=102, ema_slow_val=98, 
            stoch_k_val=10, stoch_d_val=15, 
            adx_val=50, trend_alignment_val=1.0, trend_strength_val=0.05,
            volatility_val=0.1, volume_ratio_val=2.5
        )
        mock_ta.calculate_advanced_indicators.return_value = mock_indicators

        # Create a dummy DataFrame, as generate_signal still expects one
        df = pd.DataFrame({'close': [100, 101, 102]})
        
        signal = self.signal_generator.generate_signal('BTC/USDT', df, '1h')
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal['type'], 'STRONG_BUY')
        self.assertGreaterEqual(signal['strength'], self.config.STRONG_SIGNAL_THRESHOLD)
        self.assertGreaterEqual(signal['confidence'], 0.7)
        self.assertIn('RSI oversold at 20.0', signal['reasoning'])

    @patch.object(AdvancedSignalGenerator, 'ta')
    def test_generate_strong_sell_signal(self, mock_ta):
        """Test that a strong sell signal is generated under ideal bearish conditions."""
        mock_indicators = self._create_mock_indicators(
            rsi_val=80, macd_hist_val=-0.5, bb_pos_val=0.95, 
            ema_fast_val=98, ema_slow_val=102, 
            stoch_k_val=90, stoch_d_val=85, 
            adx_val=50, trend_alignment_val=-1.0, trend_strength_val=0.05,
            volatility_val=0.1, volume_ratio_val=2.5
        )
        mock_ta.calculate_advanced_indicators.return_value = mock_indicators

        df = pd.DataFrame({'close': [100, 101, 102]})

        signal = self.signal_generator.generate_signal('BTC/USDT', df, '1h')
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal['type'], 'STRONG_SELL')
        self.assertLessEqual(signal['strength'], -self.config.STRONG_SIGNAL_THRESHOLD)
        self.assertGreaterEqual(signal['confidence'], 0.7)
        self.assertIn('RSI overbought at 80.0', signal['reasoning'])

    @patch.object(AdvancedSignalGenerator, 'ta')
    def test_generate_hold_signal(self, mock_ta):
        """Test that a hold signal is generated under neutral conditions."""
        mock_indicators = self._create_mock_indicators(
            rsi_val=50, macd_hist_val=0, bb_pos_val=0.5, 
            ema_fast_val=100, ema_slow_val=100, 
            stoch_k_val=50, stoch_d_val=50, 
            adx_val=15, trend_alignment_val=0.0, trend_strength_val=0.01,
            volatility_val=0.3, volume_ratio_val=1.0
        )
        mock_ta.calculate_advanced_indicators.return_value = mock_indicators

        df = pd.DataFrame({'close': [100, 101, 102]})

        signal = self.signal_generator.generate_signal('BTC/USDT', df, '1h')
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal['type'], 'HOLD')

if __name__ == '__main__':
    unittest.main()
