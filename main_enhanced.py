#!/usr/bin/env python3
"""
Enhanced Production Crypto Trading Bot
Simple implementation with ML and Coinbase integration
"""

import sys
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Import existing modules
from seventrader.config import Config
from seventrader.data import EnhancedMarketDataManager
from seventrader.signals import AdvancedSignalGenerator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import new modules
from seventrader.ml.advanced_signals import MLSignalPredictor, integrate_ml_signals
from seventrader.exchanges.coinbase_pro import CoinbaseProTrader

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

class EnhancedTradingBot:
    """Enhanced trading bot with ML and Coinbase integration"""
    
    def __init__(self):
        self.config = Config()
        
        # Existing components
        self.market_data = EnhancedMarketDataManager(self.config)
        self.signal_generator = AdvancedSignalGenerator(self.config)
        self.notifier = NotificationManager(self.config)
        
        # New ML component
        self.ml_predictor = MLSignalPredictor()
        self._load_ml_models()
        
        # Coinbase integration
        self.coinbase_trader = None
        self.live_trading_enabled = False
        self._initialize_coinbase()
        
        # Statistics
        self.stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'session_start': datetime.now()
        }
        
        logging.info("Enhanced Trading Bot initialized")
    
    def _load_ml_models(self):
        """Load pre-trained ML models"""
        model_path = "models/ml_models.joblib"
        if os.path.exists(model_path):
            try:
                self.ml_predictor.load_models(model_path)
                logging.info("✅ ML models loaded successfully")
            except Exception as e:
                logging.warning(f"Could not load ML models: {e}")
        else:
            logging.info("ℹ️ No pre-trained ML models found")
    
    def _initialize_coinbase(self):
        """Initialize Coinbase Pro connection"""
        try:
            if self.config.API_KEY and self.config.API_SECRET:
                self.coinbase_trader = CoinbaseProTrader(
                    self.config.API_KEY,
                    self.config.API_SECRET,
                    sandbox=self.config.SANDBOX
                )
                
                # Test connection
                balance = self.coinbase_trader.get_account_balance('USD')
                if balance is not None:
                    self.live_trading_enabled = True
                    logging.info(f"✅ Coinbase connected. USD Balance: ${balance:.2f}")
                else:
                    logging.warning("⚠️ Coinbase connection test failed")
                    
            else:
                logging.info("ℹ️ Coinbase credentials not configured")
                
        except Exception as e:
            logging.error(f"❌ Coinbase initialization failed: {e}")
    
    def train_ml_models(self, retrain: bool = False) -> bool:
        """Train ML models using historical data"""
        print("🤖 Training ML models...")
        
        try:
            # Collect training data
            print("📥 Collecting training data...")
            symbol_data = {}
            
            for symbol in self.config.SYMBOLS[:10]:  # Limit to 10 symbols for training
                try:
                    df = self.market_data.fetch_ohlcv(symbol, self.config.PRIMARY_TIMEFRAME, limit=1000)
                    if not df.empty and len(df) >= 500:
                        symbol_data[symbol] = df
                        print(f"✅ {symbol}: {len(df)} data points")
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print(f"❌ {symbol}: {e}")
                    continue
            
            if len(symbol_data) < 3:
                print("❌ Insufficient data for training")
                return False
            
            # Prepare training data
            print("⚙️ Preparing features...")
            X, y = self.ml_predictor.prepare_training_data(symbol_data)
            
            if len(X) < 100:
                print("❌ Insufficient training samples")
                return False
            
            # Train models
            print(f"🔄 Training on {len(X)} samples...")
            self.ml_predictor.train(X, y)
            
            # Save models
            os.makedirs("models", exist_ok=True)
            self.ml_predictor.save_models("models/ml_models.joblib")
            
            print("✅ ML models trained and saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ ML training failed: {e}")
            logging.error(f"ML training error: {e}")
            return False
    
    def generate_enhanced_signal(self, symbol: str, df) -> Optional[Dict]:
        """Generate enhanced signal combining TA and ML"""
        try:
            # Generate traditional TA signal
            ta_signal = self.signal_generator.generate_signal(symbol, df, self.config.PRIMARY_TIMEFRAME)
            if not ta_signal:
                return None
            
            # Integrate ML prediction if models are available
            if self.ml_predictor.is_trained:
                enhanced_signal = integrate_ml_signals(ta_signal, df, self.ml_predictor)
                enhanced_signal['generation_method'] = 'ml_enhanced'
            else:
                enhanced_signal = ta_signal
                enhanced_signal['generation_method'] = 'traditional_ta'
            
            self.stats['signals_generated'] += 1
            return enhanced_signal
            
        except Exception as e:
            logging.error(f"Error generating enhanced signal for {symbol}: {e}")
            return ta_signal if 'ta_signal' in locals() else None
    
    def run_enhanced_analysis(self) -> Dict:
        """Run enhanced market analysis"""
        print("🧠 Running enhanced market analysis...")
        
        # Collect fresh data
        self.market_data.collect_all_data_parallel()
        
        # Generate enhanced signals
        enhanced_signals = []
        
        for symbol in self.config.SYMBOLS:
            try:
                df = self.market_data.fetch_ohlcv(symbol, self.config.PRIMARY_TIMEFRAME, 
                                                limit=self.config.LOOKBACK_PERIODS)
                
                if df.empty:
                    continue
                
                enhanced_signal = self.generate_enhanced_signal(symbol, df)
                if enhanced_signal and enhanced_signal['type'] != 'HOLD':
                    enhanced_signals.append(enhanced_signal)
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logging.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by confidence and strength
        enhanced_signals.sort(
            key=lambda x: x['confidence'] * abs(x['strength']), 
            reverse=True
        )
        
        return {
            'signals': enhanced_signals,
            'ml_enabled': self.ml_predictor.is_trained,
            'live_trading': self.live_trading_enabled,
            'timestamp': datetime.now()
        }
    
    def execute_live_trades(self, signals: List[Dict]) -> List[Dict]:
        """Execute live trades on Coinbase Pro"""
        if not self.live_trading_enabled:
            print("⚠️ Live trading not enabled")
            return []
        
        results = []
        
        # Get account balance
        balance = self.coinbase_trader.get_account_balance('USD')
        if balance < 10:
            print(f"⚠️ Insufficient balance: ${balance:.2f}")
            return []
        
        # Execute top signals only
        top_signals = [s for s in signals if s['confidence'] > 0.7][:2]  # Top 2 high-confidence signals
        
        for signal in top_signals:
            try:
                print(f"🎯 Executing: {signal['symbol']} {signal['type']}")
                
                result = self.coinbase_trader.execute_signal(
                    signal['symbol'], 
                    signal['type'], 
                    balance,
                    self.config.POSITION_SIZE_PCT
                )
                
                if result.success:
                    self.stats['trades_executed'] += 1
                    self.stats['successful_trades'] += 1
                    print(f"✅ Trade executed: {result.filled_size:.6f} @ ${result.filled_price:.2f}")
                    
                    # Send notification
                    self._send_trade_notification(signal, result)
                else:
                    print(f"❌ Trade failed: {result.error}")
                
                results.append({
                    'symbol': signal['symbol'],
                    'success': result.success,
                    'price': result.filled_price,
                    'size': result.filled_size,
                    'error': result.error
                })
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error executing trade: {e}")
                continue
        
        return results
    
    def _send_trade_notification(self, signal: Dict, result):
        """Send trade notification via Telegram"""
        try:
            import requests
            
            if not self.config.TELEGRAM_BOT_TOKEN or not self.config.TELEGRAM_CHAT_ID:
                return
            
            message = f"🎯 *TRADE EXECUTED*\n\n"
            message += f"*Symbol:* {signal['symbol']}\n"
            message += f"*Type:* {signal['type']}\n"
            message += f"*Price:* ${result.filled_price:.2f}\n"
            message += f"*Size:* {result.filled_size:.6f}\n"
            message += f"*Fee:* ${result.fee:.2f}\n"
            message += f"*Confidence:* {signal['confidence']:.1%}\n"
            
            if signal.get('ml_confidence'):
                message += f"*ML Confidence:* {signal['ml_confidence']:.1%}\n"
            
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                'chat_id': self.config.TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            requests.post(url, json=data, timeout=10)
            
        except Exception as e:
            logging.error(f"Error sending notification: {e}")
    
    def display_analysis_results(self, analysis: Dict):
        """Display enhanced analysis results"""
        signals = analysis['signals']
        
        print("\n" + "="*100)
        print(f"🤖 ENHANCED CRYPTO ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        # Status
        print(f"\n📊 STATUS:")
        print(f"   ML Models: {'✅ Active' if analysis['ml_enabled'] else '❌ Not Trained'}")
        print(f"   Live Trading: {'✅ Enabled' if analysis['live_trading'] else '❌ Disabled'}")
        print(f"   Signals Found: {len(signals)}")
        
        if self.coinbase_trader:
            portfolio = self.coinbase_trader.get_portfolio_summary()
            print(f"   Account Value: ${portfolio.get('total_value_usd', 0):.2f}")
        
        # Top signals
        if signals:
            print(f"\n🎯 TOP TRADING OPPORTUNITIES:")
            print("-" * 100)
            
            for i, signal in enumerate(signals[:5], 1):
                method_icon = "🤖" if signal.get('generation_method') == 'ml_enhanced' else "📊"
                
                print(f"\n{i}. {method_icon} {signal['symbol']} - {signal['type']}")
                print(f"   💎 Score: {signal['confidence'] * abs(signal['strength']):.3f}")
                print(f"   📈 Strength: {signal['strength']:.3f} | Confidence: {signal['confidence']:.1%}")
                print(f"   💰 Price: ${signal['price']:.4f}")
                
                if signal.get('ml_confidence'):
                    print(f"   🤖 ML Confidence: {signal['ml_confidence']:.1%}")
                
                # Show key factors
                reasoning = signal.get('reasoning', [])[:3]
                if reasoning:
                    print(f"   🎯 Factors: {'; '.join(reasoning)}")
        
        # Session stats
        session_time = (datetime.now() - self.stats['session_start']).total_seconds() / 3600
        print(f"\n📈 SESSION STATS:")
        print(f"   Duration: {session_time:.1f}h")
        print(f"   Signals: {self.stats['signals_generated']}")
        print(f"   Trades: {self.stats['trades_executed']}")
        print(f"   Success Rate: {(self.stats['successful_trades']/max(1,self.stats['trades_executed'])):.1%}")
        
        print("\n" + "="*100)
    
    def run_continuous_mode(self):
        """Run bot in continuous mode"""
        print("🚀 Starting continuous enhanced trading mode...")
        
        while True:
            try:
                # Run analysis
                analysis = self.run_enhanced_analysis()
                self.display_analysis_results(analysis)
                
                # Execute trades if enabled and signals are strong
                signals = analysis['signals']
                if self.live_trading_enabled and signals:
                    strong_signals = [s for s in signals if s['confidence'] > 0.75]
                    
                    if strong_signals:
                        print(f"\n🎯 Executing {len(strong_signals)} strong signals...")
                        results = self.execute_live_trades(strong_signals)
                        
                        successful = sum(1 for r in results if r['success'])
                        print(f"✅ {successful}/{len(results)} trades successful")
                
                # Wait for next cycle
                print(f"\n⏰ Next analysis in {self.config.UPDATE_INTERVAL//60} minutes...")
                time.sleep(self.config.UPDATE_INTERVAL)
                
            except KeyboardInterrupt:
                print("\n👋 Bot stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in continuous mode: {e}")
                print(f"❌ Error: {e}")
                time.sleep(60)

    def run_single_analysis_and_notify(self):
        """Run a single analysis and send notifications for actionable signals."""
        print("🧠 Running single enhanced market analysis with notifications...")
        analysis = self.run_enhanced_analysis()
        self.display_analysis_results(analysis)

        actionable_signals = [s for s in analysis.get('signals', []) if s['type'] != 'HOLD']
        print(f"\nFound {len(actionable_signals)} actionable signals. Sending notifications...")

        for signal in actionable_signals:
            self.notifier.send_telegram_notification(signal)

        print("✅ Notifications sent.")

    def run_continuous_analysis_and_notify(self):
        """Run the bot in continuous analysis mode with notifications."""
        print("🚀 Starting continuous analysis mode with notifications...")
        while True:
            try:
                self.run_single_analysis_and_notify()
                print(f"\n⏰ Next analysis in {self.config.UPDATE_INTERVAL//60} minutes...")
                time.sleep(self.config.UPDATE_INTERVAL)
            except KeyboardInterrupt:
                print("\n👋 Bot stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in continuous analysis mode: {e}")
                print(f"❌ Error: {e}")
                time.sleep(60)

def main():
    """Main function"""
    print("🚀 ENHANCED CRYPTO TRADING BOT v2.0")
    print("ML-Powered Trading with Coinbase Pro Integration")
    print("=" * 60)
    
    try:
        bot = EnhancedTradingBot()
        
        print("\n🎛️ ENHANCED MODES:")
        print("1. 🧠 Enhanced Analysis (Single Run)")
        print("2. 🔄 Continuous Enhanced Trading")
        print("3. 🤖 Train ML Models")
        print("4. 💼 Check Coinbase Status")
        print("5. 📊 Run Backtest")
        print("6. 📥 Download Historical Data")
        print("7. 📱 Test Notification")
        print("8. 🔬 Test Coinbase Connection")
        print("9. 🔔 Single Analysis with Notification")
        print("10. 📡 Continuous Analysis with Notification")
        
        choice = input("\nChoose mode (1-10): ").strip()
        
        if choice == "1":
            analysis = bot.run_enhanced_analysis()
            bot.display_analysis_results(analysis)
            
        elif choice == "2":
            if bot.live_trading_enabled:
                confirm = input("\n⚠️ LIVE TRADING - Type 'YES' to confirm: ")
                if confirm == "YES":
                    bot.run_continuous_mode()
                else:
                    print("❌ Cancelled")
            else:
                print("❌ Live trading not available - check Coinbase credentials")
                
        elif choice == "3":
            success = bot.train_ml_models()
            if success:
                print("✅ ML models trained successfully!")
            else:
                print("❌ ML training failed")
                
        elif choice == "4":
            if bot.coinbase_trader:
                portfolio = bot.coinbase_trader.get_portfolio_summary()
                print("\n💼 Coinbase Portfolio:")
                print(f"Total Value: ${portfolio.get('total_value_usd', 0):.2f}")
                
                balances = portfolio.get('balances', {})
                for currency, balance in balances.items():
                    if balance > 0:
                        print(f"{currency}: {balance:.6f}")
            else:
                print("❌ Coinbase not connected")
                
        elif choice == "5":
            from seventrader.backtesting import Backtester
            from seventrader.historical_data import HistoricalDataManager

            data_source = input("Choose data source for backtest (live/historical): ").strip().lower()

            if data_source == 'live':
                data_manager = bot.market_data
            elif data_source == 'historical':
                data_manager = HistoricalDataManager()
            else:
                print("Invalid data source. Exiting.")
                return

            start_date = input("Start date (YYYY-MM-DD): ").strip()
            end_date = input("End date (YYYY-MM-DD): ").strip()
            symbol = input("Symbol (e.g., BTC/USDT): ").strip()
            
            backtester = Backtester(bot.config, data_manager)
            backtester.run(start_date, end_date, symbol)

        elif choice == "6":
            from download_historical_data import download_all_data
            download_all_data()

        elif choice == "7":
            bot._send_trade_notification(
                {'symbol': 'TEST', 'type': 'TEST', 'confidence': 0.9},
                type('obj', (object,), {'filled_price': 50000, 'filled_size': 0.001, 'fee': 1.0})()
            )
            print("📱 Test notification sent!")
            
        elif choice == "8":
            if bot.coinbase_trader:
                success, message = bot.coinbase_trader.test_connection()
                if success:
                    print(f"✅ {message}")
                else:
                    print(f"❌ {message}")
            else:
                print("❌ Coinbase not connected")
        
        elif choice == "9":
            bot.run_single_analysis_and_notify()

        elif choice == "10":
            bot.run_continuous_analysis_and_notify()
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
