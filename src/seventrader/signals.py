import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats

import joblib

from .config import Config
from .data import DatabaseManager

# Enhanced Technical Analysis Engine
class AdvancedTechnicalAnalysis:
    def __init__(self, config: Config):
        self.config = config
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        try:
            if len(df) < max(self.config.EMA_TREND, self.config.LOOKBACK_PERIODS // 2):
                logging.warning("Insufficient data for advanced indicator calculation")
                return {}
            
            indicators = {}
            
            # Price and basic info
            indicators['current_price'] = df['close'].iloc[-1]
            indicators['previous_close'] = df['close'].iloc[-2]
            indicators['price_change'] = (indicators['current_price'] - indicators['previous_close']) / indicators['previous_close']
            
            # Volume analysis
            indicators['current_volume'] = df['volume'].iloc[-1]
            indicators['volume_sma'] = df['volume'].rolling(self.config.VOLUME_SMA).mean().iloc[-1]
            indicators['volume_ratio'] = indicators['current_volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
            indicators['volume_spike'] = indicators['volume_ratio'] > self.config.VOLUME_SPIKE_THRESHOLD
            
            # Volatility analysis
            returns = df['close'].pct_change()
            indicators['volatility'] = returns.rolling(self.config.VOLATILITY_LOOKBACK).std().iloc[-1] * np.sqrt(252)
            indicators['volatility_percentile'] = stats.percentileofscore(
                returns.rolling(self.config.VOLATILITY_LOOKBACK).std().dropna()[-100:], 
                indicators['volatility']
            )
            
            # RSI with multiple timeframes
            indicators['rsi'] = df.ta.rsi(close=df['close'], length=self.config.RSI_PERIOD).iloc[-1]
            indicators['rsi_sma'] = df.ta.rsi(close=df['close'], length=self.config.RSI_PERIOD).rolling(5).mean().iloc[-1]
            
            # MACD with enhanced analysis
            macd = df.ta.macd(close=df['close'], fast=self.config.MACD_FAST, 
                             slow=self.config.MACD_SLOW, signal=self.config.MACD_SIGNAL)
            macd_line = f'MACD_{self.config.MACD_FAST}_{self.config.MACD_SLOW}_{self.config.MACD_SIGNAL}'
            macd_signal = f'MACDs_{self.config.MACD_FAST}_{self.config.MACD_SLOW}_{self.config.MACD_SIGNAL}'
            macd_hist = f'MACDh_{self.config.MACD_FAST}_{self.config.MACD_SLOW}_{self.config.MACD_SIGNAL}'
            
            indicators['macd'] = macd[macd_line].iloc[-1]
            indicators['macd_signal'] = macd[macd_signal].iloc[-1]
            indicators['macd_histogram'] = macd[macd_hist].iloc[-1]
            indicators['macd_crossover'] = (macd[macd_line].iloc[-1] > macd[macd_signal].iloc[-1]) and \
                                          (macd[macd_line].iloc[-2] <= macd[macd_signal].iloc[-2])
            indicators['macd_crossunder'] = (macd[macd_line].iloc[-1] < macd[macd_signal].iloc[-1]) and \
                                           (macd[macd_line].iloc[-2] >= macd[macd_signal].iloc[-2])
            
            # Bollinger Bands with position analysis
            bbands = df.ta.bbands(close=df['close'], length=self.config.BB_PERIOD, std=self.config.BB_STD)
            bb_upper = f'BBU_{self.config.BB_PERIOD}_{self.config.BB_STD}'
            bb_middle = f'BBM_{self.config.BB_PERIOD}_{self.config.BB_STD}'
            bb_lower = f'BBL_{self.config.BB_PERIOD}_{self.config.BB_STD}'
            
            indicators['bb_upper'] = bbands[bb_upper].iloc[-1]
            indicators['bb_middle'] = bbands[bb_middle].iloc[-1]
            indicators['bb_lower'] = bbands[bb_lower].iloc[-1]
            indicators['bb_position'] = (indicators['current_price'] - indicators['bb_lower']) / \
                                       (indicators['bb_upper'] - indicators['bb_lower'])
            indicators['bb_squeeze'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle'] < 0.1
            
            # Multiple EMAs for trend analysis
            indicators['ema_fast'] = df.ta.ema(close=df['close'], length=self.config.EMA_FAST).iloc[-1]
            indicators['ema_medium'] = df.ta.ema(close=df['close'], length=self.config.EMA_MEDIUM).iloc[-1]
            indicators['ema_slow'] = df.ta.ema(close=df['close'], length=self.config.EMA_SLOW).iloc[-1]
            indicators['ema_trend'] = df.ta.ema(close=df['close'], length=self.config.EMA_TREND).iloc[-1]
            
            # Trend strength analysis
            indicators['trend_alignment'] = self._calculate_trend_alignment(indicators)
            indicators['trend_strength'] = self._calculate_trend_strength(df)
            
            # Stochastic Oscillator
            stoch = df.ta.stoch(high=df['high'], low=df['low'], close=df['close'], 
                               k=self.config.STOCH_K, d=self.config.STOCH_D, smooth_k=self.config.STOCH_SMOOTH)
            indicators['stoch_k'] = stoch[f'STOCHk_{self.config.STOCH_K}_{self.config.STOCH_D}_{self.config.STOCH_SMOOTH}'].iloc[-1]
            indicators['stoch_d'] = stoch[f'STOCHd_{self.config.STOCH_K}_{self.config.STOCH_D}_{self.config.STOCH_SMOOTH}'].iloc[-1]
            
            # ADX for trend strength
            adx = df.ta.adx(high=df['high'], low=df['low'], close=df['close'], length=self.config.ADX_PERIOD)
            indicators['adx'] = adx[f'ADX_{self.config.ADX_PERIOD}'].iloc[-1]
            indicators['strong_trend'] = indicators['adx'] > self.config.ADX_TREND_THRESHOLD
            
            # ATR for volatility
            indicators['atr'] = df.ta.atr(high=df['high'], low=df['low'], close=df['close'], 
                                        length=self.config.ATR_PERIOD).iloc[-1]
            indicators['atr_percentage'] = indicators['atr'] / indicators['current_price']
            
            # Support and Resistance levels
            indicators['support'], indicators['resistance'] = self._calculate_support_resistance(df)
            
            # Market regime detection
            indicators['market_regime'] = self._detect_market_regime(df, indicators)
            
            return indicators
            
        except Exception as e:
            logging.error(f"Error calculating advanced indicators: {e}")
            return {}
    
    def _calculate_trend_alignment(self, indicators: Dict) -> float:
        """Calculate trend alignment score based on EMA positioning"""
        try:
            current_price = indicators['current_price']
            ema_fast = indicators['ema_fast']
            ema_medium = indicators['ema_medium']
            ema_slow = indicators['ema_slow']
            ema_trend = indicators['ema_trend']
            
            # Perfect bullish alignment: price > ema_fast > ema_medium > ema_slow > ema_trend
            # Perfect bearish alignment: price < ema_fast < ema_medium < ema_slow < ema_trend
            
            if (current_price > ema_fast > ema_medium > ema_slow > ema_trend):
                return 1.0  # Perfect bullish alignment
            elif (current_price < ema_fast < ema_medium < ema_slow < ema_trend):
                return -1.0  # Perfect bearish alignment
            else:
                # Partial alignment scoring
                score = 0
                if current_price > ema_fast:
                    score += 0.25
                if ema_fast > ema_medium:
                    score += 0.25
                if ema_medium > ema_slow:
                    score += 0.25
                if ema_slow > ema_trend:
                    score += 0.25
                
                # Adjust for bearish conditions
                if current_price < ema_fast:
                    score -= 0.25
                if ema_fast < ema_medium:
                    score -= 0.25
                if ema_medium < ema_slow:
                    score -= 0.25
                if ema_slow < ema_trend:
                    score -= 0.25
                
                return score
                
        except Exception as e:
            logging.error(f"Error calculating trend alignment: {e}")
            return 0
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using linear regression"""
        try:
            prices = df['close'].iloc[-self.config.TREND_STRENGTH_PERIOD:].values
            x = np.arange(len(prices))
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            
            # Normalize slope by price level
            normalized_slope = slope / prices[-1]
            
            # R-squared indicates how well the trend fits
            trend_strength = abs(normalized_slope) * (r_value ** 2)
            
            return trend_strength if slope > 0 else -trend_strength
            
        except Exception as e:
            logging.error(f"Error calculating trend strength: {e}")
            return 0
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate dynamic support and resistance levels"""
        try:
            # Use recent price action for dynamic levels
            recent_highs = df['high'].iloc[-50:].rolling(window=5).max()
            recent_lows = df['low'].iloc[-50:].rolling(window=5).min()
            
            # Find significant levels using clustering
            highs = recent_highs.dropna().values
            lows = recent_lows.dropna().values
            
            if len(highs) > 5 and len(lows) > 5:
                # Simple approach: use recent swing highs/lows
                resistance = np.percentile(highs, 80)
                support = np.percentile(lows, 20)
            else:
                # Fallback to simple percentiles
                resistance = df['high'].iloc[-20:].max()
                support = df['low'].iloc[-20:].min()
            
            return support, resistance
            
        except Exception as e:
            logging.error(f"Error calculating support/resistance: {e}")
            return df['low'].iloc[-20:].min(), df['high'].iloc[-20:].max()
    
    def _detect_market_regime(self, df: pd.DataFrame, indicators: Dict) -> str:
        """Detect current market regime"""
        try:
            volatility = indicators.get('volatility', 0)
            trend_strength = indicators.get('trend_strength', 0)
            adx = indicators.get('adx', 0)
            
            # Define regime thresholds
            high_vol_threshold = 0.8  # 80% annualized volatility
            low_vol_threshold = 0.3   # 30% annualized volatility
            strong_trend_threshold = 0.02
            
            if volatility > high_vol_threshold:
                if abs(trend_strength) > strong_trend_threshold:
                    return "high_volatility_trending"
                else:
                    return "high_volatility_ranging"
            elif volatility < low_vol_threshold:
                if abs(trend_strength) > strong_trend_threshold:
                    return "low_volatility_trending"
                else:
                    return "low_volatility_ranging"
            else:
                if abs(trend_strength) > strong_trend_threshold:
                    return "normal_volatility_trending"
                else:
                    return "normal_volatility_ranging"
                    
        except Exception as e:
            logging.error(f"Error detecting market regime: {e}")
            return "unknown"

class AdvancedSignalGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.ta = AdvancedTechnicalAnalysis(config)
        self.db = DatabaseManager()
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model from disk."""
        try:
            model_path = "trained_model.joblib"
            if os.path.exists(model_path):
                logging.info(f"Loading pre-trained model from {model_path}")
                return joblib.load(model_path)
            else:
                logging.warning("No pre-trained model found. ML signals will be disabled.")
                return None
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
    
    def _calculate_trade_parameters(self, current_price: float, signal_type: str, atr: float) -> Dict:
        """Calculate entry, stop-loss, and take-profit levels."""
        params = {
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None
        }

        # Default multipliers, can be moved to config
        stop_loss_multiplier = 1.5
        take_profit_multiplier = 2.0

        if 'BUY' in signal_type:
            params['stop_loss'] = current_price - (atr * stop_loss_multiplier)
            params['take_profit'] = current_price + (atr * take_profit_multiplier)
        elif 'SELL' in signal_type:
            params['stop_loss'] = current_price + (atr * stop_loss_multiplier)
            params['take_profit'] = current_price - (atr * take_profit_multiplier)
            
        return params

    def generate_signal(self, symbol: str, df: pd.DataFrame, timeframe: str = "1h") -> Optional[Dict]:
        """Generate sophisticated trading signals using a machine learning model."""
        try:
            indicators = self.ta.calculate_advanced_indicators(df)
            
            if not indicators or indicators.get('current_price') is None:
                logging.warning(f"Could not calculate indicators for {symbol}")
                return None

            # Prepare features for the model
            features = self._prepare_features(indicators)
            if features.empty:
                return None

            # Get prediction from the model
            components = self._calculate_signal_components(indicators)
            if self.model and hasattr(self.model, 'is_trained') and self.model.is_trained:
                prediction = self.model.predict(features)
                if prediction is None:
                    logging.warning(f"Could not get a prediction for {symbol}. Using rule-based fallback.")
                    signal_strength = self._calculate_weighted_signal_strength(components)
                else:
                    # Convert prediction to signal strength (-1 to 1)
                    signal_strength = (prediction * 2) - 1  # Scale from [0, 1] to [-1, 1]
            else:
                # Fallback to rule-based if model is not available
                signal_strength = self._calculate_weighted_signal_strength(components)

            confidence = self._calculate_signal_confidence(indicators, components)
            print(f"DEBUG: Signal Strength: {signal_strength:.2f}, Confidence: {confidence:.2f}")
            
            logging.info(f"[{symbol}] ML Signal Strength: {signal_strength:.2f}, Confidence: {confidence:.2f}")

            # Determine signal type with enhanced logic
            signal_type = self._determine_signal_type(signal_strength, confidence, indicators)
            
            # Generate detailed reasoning
            reasoning = self._generate_signal_reasoning(components, indicators)
            
            # Calculate trade parameters if it's a trading signal
            trade_params = {}
            if signal_type != 'HOLD':
                trade_params = self._calculate_trade_parameters(
                    indicators['current_price'], 
                    signal_type, 
                    indicators['atr']
                )

            signal = {
                'symbol': symbol,
                'type': signal_type,
                'strength': signal_strength,
                'confidence': confidence,
                'timeframe': timeframe,
                'price': indicators['current_price'],
                'volume': indicators['current_volume'],
                'indicators': indicators,
                'components': {},
                'reasoning': reasoning,
                'market_regime': indicators.get('market_regime', 'unknown'),
                'trend_direction': self._get_trend_direction(indicators),
                'volatility_state': self._get_volatility_state(indicators),
                'timestamp': datetime.now(),
                **trade_params
            }
            
            # Save enhanced signal
            self.db.save_signal_data(signal)
            
            return signal
            
        except Exception as e:
            logging.error(f"Error generating ML signal for {symbol}: {e}")
            return None

    def _prepare_features(self, indicators: Dict) -> pd.DataFrame:
        """Prepare features from indicators for the ML model."""
        # Select a subset of indicators as features
        feature_keys = [
            'price_change', 'volume_ratio', 'volatility', 'rsi', 'macd_histogram',
            'bb_position', 'trend_alignment', 'trend_strength', 'stoch_k', 'adx'
        ]
        
        features = {key: indicators.get(key) for key in feature_keys}
        
        # Handle missing values (e.g., from insufficient data)
        if any(value is None for value in features.values()):
            logging.warning("Some indicators are missing. Cannot create features.")
            return pd.DataFrame()

        return pd.DataFrame([features])

    
    
    def _calculate_signal_components(self, indicators: Dict) -> Dict:
        """Calculate individual signal components"""
        components = {}
        
        # RSI Component
        rsi = indicators.get('rsi', 50)
        if rsi < self.config.RSI_OVERSOLD:
            components['rsi'] = min((self.config.RSI_OVERSOLD - rsi) / 10, 1.0)
        elif rsi > self.config.RSI_OVERBOUGHT:
            components['rsi'] = max((self.config.RSI_OVERBOUGHT - rsi) / 10, -1.0)
        else:
            components['rsi'] = 0
        
        # MACD Component
        macd_strength = 0
        if indicators.get('macd_crossover'):
            macd_strength += 0.7
        elif indicators.get('macd_crossunder'):
            macd_strength -= 0.7
        
        if indicators.get('macd_histogram', 0) > 0:
            macd_strength += 0.3
        elif indicators.get('macd_histogram', 0) < 0:
            macd_strength -= 0.3
        
        components['macd'] = max(min(macd_strength, 1.0), -1.0)
        
        # Trend Component
        trend_alignment = indicators.get('trend_alignment', 0)
        trend_strength = indicators.get('trend_strength', 0)
        components['trend'] = trend_alignment * min(abs(trend_strength) * 20, 1.0)
        
        # Bollinger Bands Component
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position <= 0.1:
            components['bollinger'] = 0.8  # Near lower band - bullish
        elif bb_position >= 0.9:
            components['bollinger'] = -0.8  # Near upper band - bearish
        else:
            components['bollinger'] = 0
        
        # Volume Component
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > self.config.VOLUME_SPIKE_THRESHOLD:
            components['volume'] = min((volume_ratio - 1) / 2, 1.0)
        else:
            components['volume'] = 0
        
        # Stochastic Component
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        if stoch_k < 20 and stoch_d < 20:
            components['stochastic'] = 0.6
        elif stoch_k > 80 and stoch_d > 80:
            components['stochastic'] = -0.6
        else:
            components['stochastic'] = 0
        
        # ADX Component (trend strength filter)
        adx = indicators.get('adx', 0)
        components['adx_filter'] = min(adx / 50, 1.0)  # Strength multiplier
        
        return components
    
    def _calculate_weighted_signal_strength(self, components: Dict) -> float:
        print(f"DEBUG: Components: {components}")
        """Calculate weighted signal strength"""
        weights = {
            'trend': 0.35,
            'macd': 0.25,
            'rsi': 0.15,
            'bollinger': 0.10,
            'volume': 0.10,
            'stochastic': 0.05
        }
        
        weighted_strength = 0
        for component, weight in weights.items():
            if component in components:
                weighted_strength += components[component] * weight
        
        # Apply a less aggressive ADX filter
        adx_multiplier = components.get('adx_filter', 0.5)
        # This new formula boosts the signal in non-trending markets
        weighted_strength *= (0.6 + 0.4 * adx_multiplier)
        print(f"DEBUG: Weighted Strength before final clamp: {weighted_strength:.2f}")
        
        return max(min(weighted_strength, 1.0), -1.0)
    
    def _calculate_signal_confidence(self, indicators: Dict, components: Dict) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence_factors = []
        
        # Trend alignment confidence
        trend_alignment = abs(indicators.get('trend_alignment', 0))
        confidence_factors.append(trend_alignment)
        
        # ADX confidence (strong trends are more reliable)
        adx = indicators.get('adx', 0)
        confidence_factors.append(min(adx / 50, 1.0))
        
        # Volume confirmation
        volume_ratio = indicators.get('volume_ratio', 1)
        confidence_factors.append(min(volume_ratio / 3, 1.0))
        
        # Component agreement (how many components agree)
        positive_components = sum(1 for comp in components.values() if isinstance(comp, (int, float)) and comp > 0.3)
        negative_components = sum(1 for comp in components.values() if isinstance(comp, (int, float)) and comp < -0.3)
        total_components = len([comp for comp in components.values() if isinstance(comp, (int, float))])
        
        if total_components > 0:
            agreement = max(positive_components, negative_components) / total_components
            confidence_factors.append(agreement)
        
        # Volatility adjustment (lower confidence in high volatility)
        volatility = indicators.get('volatility', 0.5)
        vol_confidence = max(0.3, 1.0 - (volatility / 2.0))
        confidence_factors.append(vol_confidence)
        
        return np.mean(confidence_factors)
    
    def _determine_signal_type(self, signal_strength: float, confidence: float, indicators: Dict) -> str:
        """Determine signal type with enhanced logic"""
        # Apply confidence threshold
        min_confidence = 0.4
        if confidence < min_confidence:
            return "HOLD"
        
        # Check for strong signals
        if signal_strength >= self.config.STRONG_SIGNAL_THRESHOLD and confidence >= 0.7:
            return "STRONG_BUY"
        elif signal_strength <= -self.config.STRONG_SIGNAL_THRESHOLD and confidence >= 0.7:
            return "STRONG_SELL"
        elif signal_strength >= self.config.BUY_THRESHOLD:
            return "BUY"
        elif signal_strength <= self.config.SELL_THRESHOLD:
            return "SELL"
        else:
            return "HOLD"
    
    def _generate_signal_reasoning(self, components: Dict, indicators: Dict) -> List[str]:
        """Generate detailed reasoning for the signal"""
        reasons = []
        
        # Trend analysis
        trend_alignment = indicators.get('trend_alignment', 0)
        if trend_alignment > 0.5:
            reasons.append("Strong bullish trend alignment across all EMAs")
        elif trend_alignment < -0.5:
            reasons.append("Strong bearish trend alignment across all EMAs")
        
        # MACD analysis
        if indicators.get('macd_crossover'):
            reasons.append("MACD bullish crossover detected")
        elif indicators.get('macd_crossunder'):
            reasons.append("MACD bearish crossover detected")
        
        # RSI analysis
        rsi = indicators.get('rsi', 50)
        if rsi < self.config.RSI_OVERSOLD:
            reasons.append(f"RSI oversold at {rsi:.1f}")
        elif rsi > self.config.RSI_OVERBOUGHT:
            reasons.append(f"RSI overbought at {rsi:.1f}")
        
        # Bollinger Bands analysis
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position <= 0.1:
            reasons.append("Price touching lower Bollinger Band")
        elif bb_position >= 0.9:
            reasons.append("Price touching upper Bollinger Band")
        
        # Volume analysis
        if indicators.get('volume_spike'):
            reasons.append(f"Volume spike detected (ratio: {indicators.get('volume_ratio', 1):.1f})")
        
        # ADX analysis
        adx = indicators.get('adx', 0)
        if adx > self.config.ADX_TREND_THRESHOLD:
            reasons.append(f"Strong trend confirmed by ADX ({adx:.1f})")
        
        # Market regime
        regime = indicators.get('market_regime', 'unknown')
        if 'trending' in regime:
            reasons.append(f"Market in trending regime: {regime}")
        elif 'ranging' in regime:
            reasons.append(f"Market in ranging regime: {regime}")
        
        return reasons
    
    def _get_trend_direction(self, indicators: Dict) -> str:
        """Get trend direction"""
        trend_alignment = indicators.get('trend_alignment', 0)
        if trend_alignment > 0.3:
            return "bullish"
        elif trend_alignment < -0.3:
            return "bearish"
        else:
            return "neutral"
    
    def _get_volatility_state(self, indicators: Dict) -> str:
        """Get volatility state"""
        volatility = indicators.get('volatility', 0.5)
        if volatility > 0.8:
            return "high"
        elif volatility < 0.3:
            return "low"
        else:
            return "normal"