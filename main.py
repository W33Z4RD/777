import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from seventrader.config import Config
from seventrader.data import EnhancedMarketDataManager
from seventrader.signals import AdvancedSignalGenerator
from seventrader.portfolio import PortfolioManager
from seventrader.backtesting import Backtester


load_dotenv()


# Multi-timeframe Analysis
class MultiTimeframeAnalysis:
    def __init__(self, config: Config):
        self.config = config
        self.signal_generator = AdvancedSignalGenerator(config)
    
    def analyze_multiple_timeframes(self, symbol: str, market_data_manager) -> Dict:
        """Analyze signal across multiple timeframes"""
        try:
            signals = {}
            
            for timeframe in self.config.TIMEFRAMES:
                try:
                    df = market_data_manager.fetch_ohlcv(symbol, timeframe, 
                                                        limit=self.config.LOOKBACK_PERIODS)
                    if not df.empty:
                        signal = self.signal_generator.generate_signal(symbol, df, timeframe)
                        if signal:
                            signals[timeframe] = signal
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logging.error(f"Error analyzing {symbol} on {timeframe}: {e}")
                    continue
            
            # Aggregate signals across timeframes
            aggregated_signal = self._aggregate_timeframe_signals(signals)
            
            return {
                'symbol': symbol,
                'timeframe_signals': signals,
                'aggregated_signal': aggregated_signal,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return {}
    
    def _aggregate_timeframe_signals(self, signals: Dict) -> Dict:
        """Aggregate signals from different timeframes"""
        if not signals:
            return {}
        
        # Timeframe weights (higher timeframes have more weight)
        timeframe_weights = {
            '15m': 0.1,
            '1h': 0.2,
            '4h': 0.3,
            '1d': 0.3,
            '1w': 0.1
        }
        
        # Calculate weighted strength
        weighted_strength = 0
        total_weight = 0
        signal_types = []
        
        for tf, signal in signals.items():
            weight = timeframe_weights.get(tf, 0.1)
            
            # Convert signal type to numeric value
            signal_numeric = self._signal_type_to_numeric(signal['type'])
            
            weighted_strength += signal_numeric * weight
            total_weight += weight
            signal_types.append(signal['type'])
        
        if total_weight > 0:
            weighted_strength /= total_weight
        
        # Determine aggregated signal type
        aggregated_type = self._numeric_to_signal_type(weighted_strength)
        
        # Calculate consensus
        consensus = self._calculate_consensus(signal_types)
        
        return {
            'type': aggregated_type,
            'strength': weighted_strength,
            'consensus': consensus,
            'timeframe_count': len(signals)
        }
    
    def _signal_type_to_numeric(self, signal_type: str) -> float:
        """Convert signal type to numeric value"""
        mapping = {
            'STRONG_BUY': 1.0,
            'BUY': 0.5,
            'HOLD': 0.0,
            'SELL': -0.5,
            'STRONG_SELL': -1.0
        }
        return mapping.get(signal_type, 0.0)
    
    def _numeric_to_signal_type(self, numeric_value: float) -> str:
        """Convert numeric value to signal type"""
        if numeric_value >= 0.75:
            return 'STRONG_BUY'
        elif numeric_value >= 0.25:
            return 'BUY'
        elif numeric_value <= -0.75:
            return 'STRONG_SELL'
        elif numeric_value <= -0.25:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_consensus(self, signal_types: List[str]) -> float:
        """Calculate consensus among timeframes"""
        if not signal_types:
            return 0.0
        
        # Count each signal type
        signal_counts = {}
        for signal_type in signal_types:
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        
        # Find the most common signal
        max_count = max(signal_counts.values())
        total_signals = len(signal_types)
        
        return max_count / total_signals

# Enhanced Trading Bot
class NotificationManager:
    def __init__(self, config: Config):
        self.config = config
        self.last_notification_time: Dict[str, datetime] = {}

    def send_telegram_notification(self, signal: Dict):
        """Send a notification to Telegram with a cooldown period."""
        if not self.config.TELEGRAM_BOT_TOKEN or not self.config.TELEGRAM_CHAT_ID:
            logging.warning("Telegram credentials not set. Skipping notification.")
            return

        symbol = signal['symbol']
        now = datetime.now()

        # Check if a notification for this symbol has been sent recently
        last_time = self.last_notification_time.get(symbol)
        if last_time and (now - last_time).total_seconds() < self.config.NOTIFICATION_COOLDOWN_SECONDS:
            logging.info(f"Skipping notification for {symbol} due to cooldown.")
            return

        # Format the message
        message = self._format_signal_message(signal)

        # Send the message
        url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': self.config.TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            logging.info(f"Sent Telegram notification for {symbol}")
            self.last_notification_time[symbol] = now
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send Telegram notification: {e}")

    def _format_signal_message(self, signal: Dict) -> str:
        """Format the signal into a human-readable message."""
        message = f"*🚨 New Trading Signal 🚨*\n\n"
        message += f"*Symbol:* {signal['symbol']}\n"
        message += f"*Type:* {signal['type']}\n"
        message += f"*Price:* ${signal['price']:.4f}\n"
        
        if 'entry_price' in signal:
            message += f"*Entry:* ${signal['entry_price']:.4f}\n"
        if 'stop_loss' in signal and signal['stop_loss'] is not None:
            message += f"*Stop-Loss:* ${signal['stop_loss']:.4f}\n"
        if 'take_profit' in signal and signal['take_profit'] is not None:
            message += f"*Take-Profit:* ${signal['take_profit']:.4f}\n"

        message += f"*Strength:* {signal['strength']:.2f}\n"
        message += f"*Confidence:* {signal['confidence']:.1%}\n"
        message += f"*Market Regime:* {signal.get('market_regime', 'N/A')}\n"
        
        reasoning = signal.get('reasoning')
        if reasoning:
            message += "\n*Key Factors:*\n"
            for reason in reasoning:
                message += f"- {reason}\n"
        
        return message

class AdvancedTradingBot:
    def __init__(self):
        self.config = Config()
        self.market_data = EnhancedMarketDataManager(self.config)
        self.signal_generator = AdvancedSignalGenerator(self.config)
        self.mtf_analyzer = MultiTimeframeAnalysis(self.config)
        self.portfolio_manager = PortfolioManager(self.config)
        self.notifier = NotificationManager(self.config)
        
        logging.info(f"Advanced Trading Bot initialized with {len(self.config.SYMBOLS)} symbols")

    
    def analyze_market_comprehensive(self):
        """Comprehensive market analysis with all enhancements"""
        logging.info("Starting comprehensive market analysis...")
        
        # Collect fresh data in parallel
        self.market_data.collect_all_data_parallel()
        
        # Analyze signals for primary timeframe
        primary_signals = []
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = []
            
            for symbol in self.config.SYMBOLS:
                future = executor.submit(self._analyze_single_symbol, symbol)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        primary_signals.append(result)
                except Exception as e:
                    logging.error(f"Error in symbol analysis: {e}")
        
        # Multi-timeframe analysis for top signals
        top_signals = self._filter_top_signals(primary_signals)
        mtf_analyses = []
        
        for signal in top_signals:
            mtf_analysis = self.mtf_analyzer.analyze_multiple_timeframes(
                signal['symbol'], self.market_data
            )
            if mtf_analysis:
                mtf_analyses.append(mtf_analysis)
        
        # Portfolio-level analysis
        portfolio_metrics = self.portfolio_manager.calculate_portfolio_metrics(primary_signals)
        
        return {
            'primary_signals': primary_signals,
            'multi_timeframe_analyses': mtf_analyses,
            'portfolio_metrics': portfolio_metrics,
            'analysis_timestamp': datetime.now()
        }
    
    def _analyze_single_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze a single symbol"""
        try:
            df = self.market_data.fetch_ohlcv(symbol, self.config.PRIMARY_TIMEFRAME, 
                                            limit=self.config.LOOKBACK_PERIODS)
            
            if df.empty:
                logging.warning(f"No data available for {symbol}")
                return None
            
            signal = self.signal_generator.generate_signal(symbol, df, self.config.PRIMARY_TIMEFRAME)
            return signal
            
        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _filter_top_signals(self, signals: List[Dict], top_n: int = 10) -> List[Dict]:
        """Filter top signals based on strength and confidence"""
        if not signals:
            return []
        
        # Filter out HOLD signals
        actionable_signals = [s for s in signals if s['type'] != 'HOLD']
        
        # Sort by combined score (strength * confidence)
        actionable_signals.sort(
            key=lambda x: abs(x['strength']) * x['confidence'], 
            reverse=True
        )
        
        return actionable_signals[:top_n]
    
    def display_comprehensive_analysis(self, analysis: Dict):
        """Display comprehensive analysis results"""
        print("\n" + "="*100)
        print(f"ADVANCED CRYPTO TRADING ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        # Primary signals summary
        primary_signals = analysis.get('primary_signals', [])
        actionable_signals = [s for s in primary_signals if s['type'] != 'HOLD']
        
        print(f"\nSIGNALS SUMMARY:")
        print(f"Total symbols analyzed: {len(primary_signals)}")
        print(f"Actionable signals: {len(actionable_signals)}")
        
        # Display top signals
        top_signals = self._filter_top_signals(primary_signals, 5)
        
        print(f"\nTOP 5 TRADING OPPORTUNITIES:")
        print("-" * 100)
        
        for i, signal in enumerate(top_signals, 1):
            print(f"\n{i}. {signal['symbol']} - {signal['type']}")
            print(f"   Price: ${signal['price']:.4f}")
            print(f"   Strength: {signal['strength']:.2f} | Confidence: {signal['confidence']:.1%}")
            print(f"   Market Regime: {signal.get('market_regime', 'N/A')}")
            print(f"   Trend: {signal.get('trend_direction', 'N/A')} | Volatility: {signal.get('volatility_state', 'N/A')}")
            
            # Show key reasons
            reasoning = signal.get('reasoning', [])
            if reasoning:
                print(f"   Key Factors: {'; '.join(reasoning[:3])}")
        
        # Multi-timeframe analysis
        mtf_analyses = analysis.get('multi_timeframe_analyses', [])
        if mtf_analyses:
            print(f"\nMULTI-TIMEFRAME CONFIRMATION:")
            print("-" * 100)
            
            for mtf in mtf_analyses:
                symbol = mtf['symbol']
                agg_signal = mtf.get('aggregated_signal', {})
                
                print(f"\n{symbol}:")
                print(f"   Aggregated Signal: {agg_signal.get('type', 'N/A')}")
                print(f"   Timeframe Consensus: {agg_signal.get('consensus', 0):.1%}")
                print(f"   Analyzed Timeframes: {agg_signal.get('timeframe_count', 0)}")
        
        # Portfolio metrics
        portfolio_metrics = analysis.get('portfolio_metrics', {})
        if portfolio_metrics:
            print(f"\nPORTFOLIO ANALYSIS:")
            print("-" * 100)
            print(f"Diversification Score: {portfolio_metrics.get('diversification_score', 0):.2f}")
            print(f"Buy Signals: {portfolio_metrics.get('buy_signals', 0)}")
            print(f"Sell Signals: {portfolio_metrics.get('sell_signals', 0)}")
        
        print("\n" + "="*100)
        print("DISCLAIMER: This is automated analysis. Always do your own research before trading.")
        print("="*100)
    
    def run_analysis_cycle(self):
        """Run a single comprehensive analysis cycle"""
        try:
            analysis = self.analyze_market_comprehensive()
            self.display_comprehensive_analysis(analysis)
            
            actionable_signals = [s for s in analysis.get('primary_signals', []) if s['type'] != 'HOLD']
            for signal in actionable_signals:
                self.notifier.send_telegram_notification(signal)
            
            return len(actionable_signals)
            
        except Exception as e:
            logging.error(f"Error in comprehensive analysis cycle: {e}")
            return 0
    
    def run_continuous(self):
        """Run bot continuously with enhanced features"""
        logging.info("Starting Advanced Trading Bot in continuous mode...")
        
        while True:
            try:
                signal_count = self.run_analysis_cycle()
                
                if signal_count > 0:
                    print(f"\n🎯 Generated {signal_count} trading opportunities!")
                else:
                    print(f"\n📊 No strong signals found. Market analysis complete.")
                
                print(f"⏰ Next analysis in {self.config.UPDATE_INTERVAL//60} minutes...")
                time.sleep(self.config.UPDATE_INTERVAL)
                
            except KeyboardInterrupt:
                logging.info("Advanced Trading Bot stopped by user")
                break
            except Exception as e:
                logging.error(f"Unexpected error in continuous mode: {e}")
                time.sleep(60)

if __name__ == "__main__":
    print("🚀 Advanced Crypto Trading Bot - Phase 2")
    print("Features: Multi-timeframe analysis, 40+ currencies, advanced indicators")
    print("=" * 70)
    
    bot = AdvancedTradingBot()
    
    print("\nChoose mode:")
    print("1. Single Comprehensive Analysis")
    print("2. Continuous Advanced Monitoring")
    print("3. Run Backtest")
    print("4. Send Telegram Test Message")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        bot.run_analysis_cycle()
    elif choice == "2":
        bot.run_continuous()
    elif choice == "3":
        start_date = input("Enter backtest start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter backtest end date (YYYY-MM-DD): ").strip()
        symbol = input(f"Enter symbol to backtest (e.g., BTC/USDT): ").strip().upper()
        
        backtester = Backtester(bot.config)
        backtester.run(start_date, end_date, symbol)
    elif choice == "4":
        bot.notifier.send_telegram_notification({
            'symbol': 'TEST',
            'type': 'TEST',
            'price': 0,
            'strength': 0,
            'confidence': 0,
            'market_regime': 'N/A',
            'reasoning': ['This is a test message.']
        })
    else:
        print("Invalid choice. Exiting.")