# src/seventrader/ml/advanced_signals.py

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM features disabled.")

class AdvancedFeatureEngine:
    """Advanced feature engineering for crypto trading"""
    
    def __init__(self, lookback_window=50):
        self.lookback_window = lookback_window
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features from OHLCV data"""
        if len(df) < self.lookback_window:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['price_ma_5'] = df['close'].rolling(5).mean()
        features['price_ma_20'] = df['close'].rolling(20).mean()
        features['price_to_ma_5'] = df['close'] / features['price_ma_5']
        features['price_to_ma_20'] = df['close'] / features['price_ma_20']
        
        # Volatility features
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['vol_ratio'] = features['volatility_5'] / features['volatility_20']
        
        # Volume features
        features['volume_ma'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
        features['price_volume_corr'] = features['returns'].rolling(20).corr(df['volume'].pct_change())
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(df['close'])
        macd, macd_signal = self._calculate_macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd - macd_signal
        
        # Market structure
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Trend features
        features['trend_5'] = self._calculate_trend(df['close'], 5)
        features['trend_20'] = self._calculate_trend(df['close'], 20)
        
        # Clean and return
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices):
        """Calculate MACD"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_trend(self, prices, window):
        """Calculate trend using linear regression slope"""
        def get_slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope / y.iloc[-1]  # Normalize by price
            except:
                return 0
        
        return prices.rolling(window).apply(get_slope, raw=False)

class MLSignalPredictor:
    """Machine Learning signal predictor"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_engine = AdvancedFeatureEngine()
        self.is_trained = False
        
        # Initialize models
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'xgb': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
        }
        
        for model_name in self.models:
            self.scalers[model_name] = StandardScaler()
    
    def prepare_training_data(self, symbol_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from multiple symbols"""
        all_features = []
        all_labels = []
        
        for symbol, df in symbol_data.items():
            if len(df) < 100:
                continue
            
            # Create features
            features_df = self.feature_engine.create_features(df)
            if features_df.empty:
                continue
            
            # Create labels (future returns)
            future_returns = df['close'].shift(-5).pct_change()  # 5 periods ahead
            labels = (future_returns > 0.02).astype(int)  # 1 if return > 2%
            
            # Align features and labels
            aligned_data = pd.concat([features_df, labels.rename('label')], axis=1).dropna()
            
            if len(aligned_data) > 50:
                all_features.append(aligned_data.drop('label', axis=1))
                all_labels.append(aligned_data['label'])
        
        if not all_features:
            return pd.DataFrame(), pd.Series()
        
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = pd.concat(all_labels, ignore_index=True)
        
        return combined_features, combined_labels
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train all models"""
        if len(X) < 100:
            logging.error("Insufficient training data")
            return
        
        logging.info(f"Training ML models on {len(X)} samples with {len(X.columns)} features")
        
        for model_name, model in self.models.items():
            try:
                # Scale features
                X_scaled = self.scalers[model_name].fit_transform(X)
                
                # Train model
                model.fit(X_scaled, y)
                
                # Validate
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    model.fit(X_scaled[train_idx], y.iloc[train_idx])
                    score = model.score(X_scaled[val_idx], y.iloc[val_idx])
                    scores.append(score)
                
                avg_score = np.mean(scores)
                logging.info(f"{model_name} - Validation Accuracy: {avg_score:.3f}")
                
            except Exception as e:
                logging.error(f"Error training {model_name}: {e}")
        
        self.is_trained = True
    
    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate predictions from all models"""
        if not self.is_trained:
            return {'prediction': 0.5, 'confidence': 0.0}
        
        # Create features
        features_df = self.feature_engine.create_features(df)
        if features_df.empty:
            return {'prediction': 0.5, 'confidence': 0.0}
        
        # Get latest features
        latest_features = features_df.iloc[-1:].values
        
        predictions = []
        
        for model_name, model in self.models.items():
            try:
                # Scale features
                features_scaled = self.scalers[model_name].transform(latest_features)
                
                # Get prediction probability
                pred_proba = model.predict_proba(features_scaled)[0, 1]
                predictions.append(pred_proba)
                
            except Exception as e:
                logging.error(f"Error predicting with {model_name}: {e}")
        
        if not predictions:
            return {'prediction': 0.5, 'confidence': 0.0}
        
        # Ensemble prediction
        ensemble_pred = np.mean(predictions)
        confidence = 1.0 - np.std(predictions)  # Lower std = higher confidence
        
        return {
            'prediction': ensemble_pred,
            'confidence': confidence,
            'individual_predictions': dict(zip(self.models.keys(), predictions))
        }
    
    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        logging.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.is_trained = model_data['is_trained']
            logging.info(f"Models loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading models: {e}")

# Integration with existing system
def integrate_ml_signals(existing_signal: Dict, df: pd.DataFrame, ml_predictor: MLSignalPredictor) -> Dict:
    """Integrate ML predictions with existing TA signals"""
    
    # Get ML prediction
    ml_result = ml_predictor.predict(df)
    
    # Combine signals
    ta_strength = existing_signal.get('strength', 0)
    ml_prediction = ml_result.get('prediction', 0.5)
    ml_confidence = ml_result.get('confidence', 0)
    
    # Convert ML prediction to signal strength (-1 to 1)
    ml_strength = (ml_prediction - 0.5) * 2
    
    # Weighted combination
    if ml_confidence > 0.7:
        combined_strength = (ta_strength * 0.3) + (ml_strength * 0.7)
        combined_confidence = existing_signal.get('confidence', 0) * 0.5 + ml_confidence * 0.5
    else:
        combined_strength = (ta_strength * 0.7) + (ml_strength * 0.3)
        combined_confidence = existing_signal.get('confidence', 0) * 0.8 + ml_confidence * 0.2
    
    # Update signal
    enhanced_signal = existing_signal.copy()
    enhanced_signal.update({
        'strength': combined_strength,
        'confidence': combined_confidence,
        'ml_prediction': ml_prediction,
        'ml_confidence': ml_confidence,
        'ml_details': ml_result.get('individual_predictions', {})
    })
    
    # Update signal type based on enhanced strength
    if combined_confidence > 0.6:
        if combined_strength > 0.7:
            enhanced_signal['type'] = 'STRONG_BUY'
        elif combined_strength > 0.3:
            enhanced_signal['type'] = 'BUY'
        elif combined_strength < -0.7:
            enhanced_signal['type'] = 'STRONG_SELL'
        elif combined_strength < -0.3:
            enhanced_signal['type'] = 'SELL'
        else:
            enhanced_signal['type'] = 'HOLD'
    else:
        enhanced_signal['type'] = 'HOLD'
    
    return enhanced_signal
