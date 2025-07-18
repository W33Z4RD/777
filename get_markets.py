
import ccxt
import json

def get_binance_markets():
    try:
        binance = ccxt.binance()
        markets = binance.load_markets()
        with open('markets.json', 'w') as f:
            json.dump(markets, f, indent=4)
        print("Markets saved to markets.json")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_binance_markets()
