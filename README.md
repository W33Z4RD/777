# 777PROJECT -  Crypto Trading Bot

## Project Overview
The 777PROJECT is an advanced, modular, and highly configurable cryptocurrency trading bot. It leverages machine learning, comprehensive technical analysis, and multi-timeframe strategies to identify trading opportunities. The bot is capable of running in live trading mode with Coinbase, performing in-depth backtesting, and sending real-time notifications via Telegram.

## Features
-   **Machine Learning Core**: Utilizes a suite of models (Random Forest, Gradient Boosting, XGBoost) to generate and enhance trading signals.
-   **Live Trading with Coinbase**: Integrated with the Coinbase Cloud API for executing live trades.
-   **Multi-Timeframe Analysis**: Aggregates data from various timeframes (e.g., 15m, 1h, 4h, 1d) for more robust signals.
-   **Advanced Technical Analysis**: Employs a wide range of indicators from `pandas-ta`, including RSI, MACD, Bollinger Bands, EMAs, and more.
-   **Comprehensive Backtesting**: A powerful backtesting engine to simulate and evaluate strategies on historical data.
-   **Data Management**: Includes tools for downloading and storing historical market data in a local SQLite database.
-   **Telegram Notifications**: Sends detailed notifications for generated signals and executed trades.
-   **Highly Configurable**: Easily configure symbols, timeframes, and strategies through a `.env` file.
-   **Interactive CLI**: A user-friendly command-line interface to control the bot's functions.

## Installation

### Prerequisites
-   Python 3.9+
-   `pip` (Python package installer)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/W33Z4RD/777
    cd 777PROJECT
    ```

2.  **Create a Python virtual environment (Best pratice):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install TA-Lib:**
    This project requires the TA-Lib library. A helper script is provided for Linux/macOS.
    ```bash
    chmod +x install_talib.sh
    ./install_talib.sh
    ```
    For Windows, please refer to the official TA-Lib installation guides.

5.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of the project. This file will hold your API keys and custom configurations.

    ```
    # .env example
    # Coinbase API Credentials
    COINBASE_API_KEY="organizations/YOUR_ORG_ID/apiKeys/YOUR_API_KEY_UUID"
    COINBASE_API_SECRET="-----BEGIN EC PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END EC PRIVATE KEY-----"
    EXCHANGE_SANDBOX="true" # Use "true" for sandbox, "false" for live trading

    # Telegram Bot Credentials
    TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
    TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID"

    # Optional: Override default symbols or timeframes
    # SYMBOLS='["BTC/USDT", "ETH/USDT"]'
    # TIMEFRAMES='["1h", "4h"]'
    # UPDATE_INTERVAL=300 # 5 minutes
    ```

## Usage
The primary entry point for the bot is `main_enhanced.py`.

```bash
python main_enhanced.py
```

You will be presented with an interactive menu:

```
🚀 ENHANCED CRYPTO TRADING BOT v2.0
ML-Powered Trading with Coinbase Pro Integration
============================================================

🎛️ ENHANCED MODES:
1. 🧠 Enhanced Analysis (Single Run)
2. 🔄 Continuous Enhanced Trading
3. 🤖 Train ML Models
4. 💼 Check Coinbase Status
5. 📊 Run Backtest
6. 📥 Download Historical Data
7. 📱 Test Notification
8. 🔬 Test Coinbase Connection
9. 🔔 Single Analysis with Notification
10. 📡 Continuous Analysis with Notification

Choose mode (1-10):
```

### Modes of Operation
-   **1. Enhanced Analysis (Single Run)**: Performs a one-time market analysis and displays the results.
-   **2. Continuous Enhanced Trading**: Runs the bot in a continuous loop. If live trading is enabled, it will execute trades based on strong signals.
-   **3. Train ML Models**: Trains the machine learning models using the historical data stored in the database.
-   **4. Check Coinbase Status**: Fetches and displays your portfolio summary from Coinbase.
-   **5. Run Backtest**: Initiates the backtesting engine for a specified symbol and date range.
-   **6. Download Historical Data**: Downloads historical OHLCV data for all configured symbols.
-   **7. Test Notification**: Sends a test message to your configured Telegram chat.
-   **8. Test Coinbase Connection**: Verifies the connection to the Coinbase API.
-   **9. Single Analysis with Notification**: Runs a single analysis and sends Telegram notifications for any actionable signals.
-   **10. Continuous Analysis with Notification**: Runs the bot in a continuous analysis loop, sending notifications for new signals.

## Project Structure
-   `main_enhanced.py`: The main entry point for the application.
-   `src/seventrader/`: The core package for the trading bot.
    -   `config.py`: Handles all configuration management.
    -   `data.py`: Manages data fetching and database interactions.
    -   `signals.py`: The primary engine for generating technical analysis-based signals.
    -   `ml/advanced_signals.py`: Contains the machine learning feature engineering and prediction logic.
    -   `exchanges/coinbase_pro.py`: Manages all interactions with the Coinbase API.
    -   `backtesting.py`: The backtesting engine for strategy simulation.
-   `scripts/`: Contains utility scripts for various tasks.
-   `tests/`: Unit tests for the project.
-   `requirements.txt`: A list of all Python dependencies.
-   `install_talib.sh`: A helper script for installing the TA-Lib dependency.

## Disclaimer
**Trading cryptocurrencies involves substantial risk and is not suitable for every investor.** This bot is provided for educational and informational purposes only. The developers of this project are not responsible for any financial losses. Always do your own research and use this software at your own risk.
