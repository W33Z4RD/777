import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .config import Config
from .data import DatabaseManager
from .signals import AdvancedSignalGenerator

# Backtesting Engine
@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    size: float = 0
    pnl: float = 0
    status: str = "OPEN"
    signal_type: str = ""

class SimulatedPortfolio:
    def __init__(self, initial_cash: float = 10000.0, fee_rate: float = 0.001, slippage_pct: float = 0.0005):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        self.positions: Dict[str, Trade] = {}
        self.trade_history: List[Trade] = []
        self.equity_curve = pd.DataFrame(columns=['timestamp', 'equity'])

    def record_equity(self, timestamp: datetime, current_prices: Dict[str, float]):
        total_value = self.cash
        for symbol, trade in self.positions.items():
            current_price = current_prices.get(symbol, trade.entry_price)
            total_value += trade.size * current_price
        
        new_row = pd.DataFrame([{'timestamp': timestamp, 'equity': total_value}])
        self.equity_curve = pd.concat([self.equity_curve, new_row], ignore_index=True)

    def open_position(self, symbol: str, signal_type: str, price: float, timestamp: datetime, size_pct: float):
        if symbol in self.positions:
            logging.warning(f"Position already open for {symbol}. Skipping new position.")
            return

        # Apply slippage
        if "BUY" in signal_type:
            price *= (1 + self.slippage_pct)
        else: # Slippage for short selling
            price *= (1 - self.slippage_pct)

        trade_size_usd = self.cash * size_pct
        if self.cash < trade_size_usd:
            logging.warning(f"Insufficient cash to open position for {symbol}.")
            return
        
        size = trade_size_usd / price
        fee = trade_size_usd * self.fee_rate
        self.cash -= (trade_size_usd + fee)

        trade = Trade(
            symbol=symbol,
            entry_time=timestamp,
            entry_price=price,
            size=size,
            signal_type=signal_type
        )
        self.positions[symbol] = trade
        logging.info(f"Opened {signal_type} position for {symbol} at {price:.4f} (fee: ${fee:.2f})")

    def close_position(self, symbol: str, price: float, timestamp: datetime):
        if symbol not in self.positions:
            logging.warning(f"No open position found for {symbol} to close.")
            return

        trade = self.positions.pop(symbol)
        trade.exit_time = timestamp
        
        # Apply slippage
        if "BUY" in trade.signal_type:
            price *= (1 - self.slippage_pct)
        else: # Slippage for short covering
            price *= (1 + self.slippage_pct)
            
        trade.exit_price = price

        trade_value = trade.size * trade.exit_price
        fee = trade_value * self.fee_rate
        
        if "BUY" in trade.signal_type:
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.size - fee
        else: # Assumes SHORT if not BUY
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.size - fee

        self.cash += (trade.entry_price * trade.size) + trade.pnl
        trade.status = "CLOSED"
        self.trade_history.append(trade)
        logging.info(f"Closed position for {symbol} at {price:.4f}. PnL: {trade.pnl:.2f} (fee: ${fee:.2f})")

    def check_stops(self, symbol: str, current_price: float, timestamp: datetime, atr: float, stop_loss_atr_multiplier: float, take_profit_atr_multiplier: float):
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]
        
        if "BUY" in trade.signal_type:
            stop_loss_price = trade.entry_price - (atr * stop_loss_atr_multiplier)
            take_profit_price = trade.entry_price + (atr * take_profit_atr_multiplier)

            if current_price <= stop_loss_price:
                logging.info(f"Stop loss triggered for {symbol} at {current_price:.4f}")
                self.close_position(symbol, current_price, timestamp)
            elif current_price >= take_profit_price:
                logging.info(f"Take profit triggered for {symbol} at {current_price:.4f}")
                self.close_position(symbol, current_price, timestamp)
        else: # Handle short positions
            stop_loss_price = trade.entry_price + (atr * stop_loss_atr_multiplier)
            take_profit_price = trade.entry_price - (atr * take_profit_atr_multiplier)

            if current_price >= stop_loss_price:
                logging.info(f"Stop loss triggered for {symbol} at {current_price:.4f}")
                self.close_position(symbol, current_price, timestamp)
            elif current_price <= take_profit_price:
                logging.info(f"Take profit triggered for {symbol} at {current_price:.4f}")
                self.close_position(symbol, current_price, timestamp)
        
    def get_summary(self):
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trade_history)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        final_equity = self.cash
        
        print("\n--- Backtest Summary ---")
        print(f"Initial Cash: ${self.initial_cash:,.2f}")
        print(f"Final Equity: ${final_equity:,.2f}")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Total Return: {((final_equity / self.initial_cash) - 1):.2%}")
        print("-" * 20)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average PnL per Trade: ${avg_pnl:,.2f}")
        print("------------------------\n")


class Backtester:
    def __init__(self, config: Config, fee_rate: float = 0.001, slippage_pct: float = 0.0005):
        self.config = config
        self.db = DatabaseManager()
        self.signal_generator = AdvancedSignalGenerator(config)
        self.portfolio = SimulatedPortfolio(initial_cash=10000.0, fee_rate=fee_rate, slippage_pct=slippage_pct)

    def run(self, start_date: str, end_date: str, symbol: str):
        logging.info(f"Starting backtest for {symbol} from {start_date} to {end_date}...")

        # Load data
        df = self.db.load_market_data_for_backtest(symbol, self.config.PRIMARY_TIMEFRAME, start_date, end_date)
        if df.empty:
            logging.error(f"No data found for {symbol} in the given date range.")
            return

        # Iterate through the data, simulating time
        for i in range(self.config.LOOKBACK_PERIODS, len(df)):
            historical_df = df.iloc[i - self.config.LOOKBACK_PERIODS : i]
            current_timestamp = historical_df.index[-1]
            current_price = historical_df['close'].iloc[-1]

            # Check for stop loss / take profit on open positions
            if symbol in self.portfolio.positions:
                atr = signal['indicators']['atr'] if signal and 'indicators' in signal and 'atr' in signal['indicators'] else 0
                if atr > 0:
                    self.portfolio.check_stops(
                        symbol, current_price, current_timestamp, atr,
                        self.config.STOP_LOSS_ATR_MULTIPLIER, self.config.TAKE_PROFIT_ATR_MULTIPLIER
                    )

            # Generate signal based on historical data up to this point
            signal = self.signal_generator.generate_signal(symbol, historical_df, self.config.PRIMARY_TIMEFRAME)

            if not signal:
                continue

            # Execute trade based on signal
            if "BUY" in signal['type']:
                self.portfolio.open_position(
                    symbol, signal['type'], signal['price'], current_timestamp, self.config.POSITION_SIZE_PCT
                )
            elif "SELL" in signal['type']:
                # Close existing long position on a sell signal
                if symbol in self.portfolio.positions:
                    self.portfolio.close_position(symbol, signal['price'], current_timestamp)
            
            # Record equity at each step
            self.portfolio.record_equity(current_timestamp, {symbol: current_price})

        # Close any open positions at the end of the backtest
        if symbol in self.portfolio.positions:
            final_price = df['close'].iloc[-1]
            final_timestamp = df.index[-1]
            self.portfolio.close_position(symbol, final_price, final_timestamp)

        # Print summary
        self.portfolio.get_summary()
        
        # Optional: Plot equity curve
        if not self.portfolio.equity_curve.empty:
            try:
                import matplotlib.pyplot as plt
                self.portfolio.equity_curve.set_index('timestamp')['equity'].plot(
                    title=f"Equity Curve for {symbol}",
                    figsize=(12, 6)
                )
                plt.xlabel("Date")
                plt.ylabel("Portfolio Equity ($)")
                plt.grid(True)
                plt.show()
            except ImportError:
                logging.warning("Matplotlib not installed. Cannot plot equity curve. `pip install matplotlib`")
