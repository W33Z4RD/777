# 777PROJECT - Advanced Crypto Trading Bot

## Project Overview
The 777PROJECT is an advanced, modular, and highly configurable cryptocurrency trading bot designed for comprehensive market analysis, sophisticated signal generation using machine learning, and robust backtesting capabilities. It aims to identify trading opportunities across multiple timeframes and notify users via Telegram.

## Features
-   **Multi-Timeframe Analysis**: Aggregates signals from various timeframes (e.g., 15m, 1h, 4h, 1d, 1w) for higher conviction.
-   **Advanced Technical Indicators**: Utilizes a wide array of technical indicators (RSI, MACD, Bollinger Bands, EMAs, Stochastic, ADX, ATR) for in-depth market analysis.
-   **Machine Learning Integration**: Employs a Logistic Regression model to enhance signal generation and confidence.
-   **Parallel Data Collection**: Efficiently fetches market data for multiple symbols and timeframes concurrently.
-   **SQLite Database**: Stores historical market data and generated signals for persistence and backtesting.
-   **Comprehensive Backtesting Engine**: Simulate trading strategies with realistic fees and slippage, including stop-loss and take-profit mechanisms, and visualize equity curves.
-   **Configurable Trading Universe**: Easily expand or modify the list of cryptocurrencies to monitor.
-   **Telegram Notifications**: Receive real-time trading signal alerts directly to your Telegram chat.
-   **Market Regime Detection**: Identifies current market conditions (e.g., high/low volatility, trending/ranging) to adapt strategies.
-   **Detailed Signal Reasoning**: Provides insights into why a particular signal was generated based on contributing factors.

## Installation

### Prerequisites
-   Python 3.9+
-   `pip` (Python package installer)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/777PROJECT.git
    cd 777PROJECT
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install TA-Lib (Technical Analysis Library):**
    This project uses `TA-Lib` which requires a separate installation. A helper script `install_talib.sh` is provided for Linux/macOS.
    ```bash
    chmod +x install_talib.sh
    ./install_talib.sh
    ```
    For Windows, please refer to the official TA-Lib installation guide: [https://github.com/TA-Lib/ta-lib-python#installing-ta-lib](https://github.com/TA-Lib/ta-lib-python#installing-ta-lib)

5.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of the project to store sensitive information and override default configurations.
    ```
    # .env example
    EXCHANGE_NAME=binance
    EXCHANGE_API_KEY=YOUR_BINANCE_API_KEY
    EXCHANGE_API_SECRET=YOUR_BINANCE_API_SECRET
    # EXCHANGE_PASSPHRASE=YOUR_PASSPHRASE_IF_NEEDED (e.g., for KuCoin)
    EXCHANGE_SANDBOX=false # Set to true for testing on sandbox environment

    TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID

    # Optional: Override default symbols or timeframes
    # SYMBOLS='["BTC/USDT", "ETH/USDT"]'
    # TIMEFRAMES='["1h", "4h"]'
    # UPDATE_INTERVAL=300 # 5 minutes
    ```
    Ensure you replace placeholders with your actual API keys and Telegram details.

## Usage

To run the bot, execute `main.py` and choose from the interactive menu:

```bash
python main.py
```

You will be presented with the following options:

```
🚀 Advanced Crypto Trading Bot - Phase 2
Features: Multi-timeframe analysis, 40+ currencies, advanced indicators
======================================================================

Choose mode:
1. Single Comprehensive Analysis
2. Continuous Advanced Monitoring
3. Run Backtest
4. Send Telegram Test Message

Enter your choice (1-4):
```

### 1. Single Comprehensive Analysis
Runs a one-time market analysis across all configured symbols and timeframes, displays the results, and sends Telegram notifications for actionable signals.

### 2. Continuous Advanced Monitoring
Runs the bot in a continuous loop, performing market analysis at intervals defined by `UPDATE_INTERVAL` in `config.py` (or your `.env file).

### 3. Run Backtest
Allows you to backtest the trading strategy on historical data for a specific symbol. You will be prompted to enter the start date, end date, and symbol.
Example:
```
Enter backtest start date (YYYY-MM-DD): 2023-01-01
Enter backtest end date (YYYY-MM-DD): 2023-12-31
Enter symbol to backtest (e.g., BTC/USDT): BTC/USDT
```
The backtest will display a summary of performance metrics and, if `matplotlib` is installed, an equity curve plot.

### 4. Send Telegram Test Message
Sends a test message to your configured Telegram chat to verify your `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set up correctly.

## Configuration
All core parameters are defined in `src/seventrader/config.py`. You can override these defaults by setting corresponding environment variables in your `.env` file.

Key configuration sections include:
-   **Exchange Configuration**: API keys, secret, passphrase, and sandbox mode.
-   **Trading Universe**: `SYMBOLS` to monitor.
-   **Timeframes**: `TIMEFRAMES` for multi-timeframe analysis, `PRIMARY_TIMEFRAME`, and `CONFIRMATION_TIMEFRAMES`.
-   **Technical Indicators**: Periods and thresholds for RSI, MACD, Bollinger Bands, EMAs, Stochastic, ADX, ATR, and Volume analysis.
-   **Signal Thresholds**: `BUY_THRESHOLD`, `SELL_THRESHOLD`, `STRONG_SIGNAL_THRESHOLD`.
-   **Risk Management**: `STOP_LOSS_ATR_MULTIPLIER`, `TAKE_PROFIT_ATR_MULTIPLIER`, `POSITION_SIZE_PCT`.
-   **Threading**: `MAX_WORKERS` for parallel operations.
-   **Telegram Notifications**: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `NOTIFICATION_COOLDOWN_SECONDS`.

## Architecture
The project follows a modular design:
-   `main.py`: Entry point and orchestrator.
-   `src/seventrader/config.py`: Centralized configuration management.
-   `src/seventrader/data.py`: Handles data fetching from exchanges and database interactions (SQLite).
-   `src/seventrader/signals.py`: Contains logic for technical analysis, ML-driven signal generation, and signal confidence calculation.
-   `src/seventrader/backtesting.py`: Implements the backtesting engine and simulated portfolio management.
-   `src/seventrader/models/models.py`: Defines the machine learning model used for signal prediction.

## Backtesting
The backtesting module (`src/seventrader/backtesting.py`) allows you to evaluate the strategy's performance over historical data. It simulates trades, applies fees and slippage, and tracks PnL and equity.

**Important Backtesting Parameters:**
-   `initial_cash`: Starting capital for the simulation.
-   `fee_rate`: Trading fees as a percentage (e.g., 0.001 for 0.1%).
-   `slippage_pct`: Simulated price slippage as a percentage.
-   `STOP_LOSS_ATR_MULTIPLIER`: Multiplier for ATR to set stop-loss levels.
-   `TAKE_PROFIT_ATR_MULTIPLIER`: Multiplier for ATR to set take-profit levels.
-   `POSITION_SIZE_PCT`: Percentage of available cash to use per trade.

These parameters can be adjusted in `src/seventrader/config.py`.

## Telegram Notifications
To enable Telegram notifications:
1.  **Create a Telegram Bot**: Talk to `@BotFather` on Telegram, create a new bot, and get your `TELEGRAM_BOT_TOKEN`.
2.  **Get your Chat ID**: Send a message to your new bot. Then, open your web browser and go to `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`. Look for the `"chat":{"id":...}` field to find your `TELEGRAM_CHAT_ID`.
3.  **Set Environment Variables**: Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to your `.env` file.

## Disclaimer
**Trading cryptocurrencies involves substantial risk and is not suitable for every investor.** The information provided by this bot is for educational and informational purposes only and does not constitute financial advice. Past performance is not indicative of future results. Always do your own research and consult with a qualified financial professional before making any investment decisions. The developers of this project are not responsible for any financial losses incurred.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues to improve the project.

## License
This project is open-source and available under the [MIT License](LICENSE).
