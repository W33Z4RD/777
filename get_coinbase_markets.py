import ccxt
import pandas as pd
import json

def get_top_markets_by_volume_coinbase():
    """
    Fetches the top markets by volume from Coinbase.

    Returns:
        pandas.DataFrame: A DataFrame containing the top markets 
                          sorted by quote volume, or None if an 
                          error occurs.
    """
    try:
        # Initialize the Coinbase exchange
        exchange = ccxt.coinbasepro()

        # Load all markets from the exchange
        markets = exchange.load_markets()

        # Fetch tickers for all markets
        tickers = exchange.fetch_tickers()
        print("Raw tickers:", tickers)

        # Filter out tickers that are None or don't have volume data
        valid_tickers = [
            ticker for ticker in tickers.values() 
            if ticker is not None and 'info' in ticker and 'volume_24h' in ticker['info'] and ticker['info']['volume_24h'] is not None
        ]
        print("Valid tickers after filtering:", valid_tickers)

        # Sort the tickers by quote volume in descending order
        sorted_tickers = sorted(
            valid_tickers, 
            key=lambda x: float(x['info']['volume_24h']), 
            reverse=True
        )

        # Create a list of dictionaries for the top markets
        top_markets = [
            {
                'symbol': ticker['symbol'],
                'quoteVolume': float(ticker['info']['volume_24h']) if 'volume_24h' in ticker['info'] and ticker['info']['volume_24h'] is not None else 0.0,
                'baseVolume': float(ticker['info']['volume_24h']) if 'volume_24h' in ticker['info'] and ticker['info']['volume_24h'] is not None else 0.0,
                'last': float(ticker['last']) if 'last' in ticker and ticker['last'] is not None else 0.0,
            } 
            for ticker in sorted_tickers
        ]

        # Create a pandas DataFrame for better display
        df = pd.DataFrame(top_markets)

        return df

    except (ccxt.ExchangeError, ccxt.NetworkError) as error:
        print(f"An error occurred: {error}")
        return None

if __name__ == '__main__':
    top_markets_df = get_top_markets_by_volume_coinbase()
    if top_markets_df is not None:
        print("DataFrame columns:", top_markets_df.columns)
        # Get the top 100 symbols
        top_100_symbols = top_markets_df.head(100)['symbol'].tolist()
        print(json.dumps(top_100_symbols, indent=4))