
import json

def generate_symbol_list():
    with open('markets.json', 'r') as f:
        markets = json.load(f)

    symbols = []
    bases = set()

    # Prioritize USDC pairs
    for symbol, market in markets.items():
        if market['quote'] == 'USDC' and market['active']:
            symbols.append(symbol)
            bases.add(market['base'])

    # Add USDT pairs for coins not already present with USDC
    for symbol, market in markets.items():
        if market['quote'] == 'USDT' and market['base'] not in bases and market['active']:
            symbols.append(symbol)
            bases.add(market['base'])

    # Sort and limit to a reasonable number (e.g., 100)
    # A sort is not strictly necessary but makes the list cleaner
    symbols.sort()
    final_list = symbols[:100]

    print(json.dumps(final_list, indent=4))

if __name__ == "__main__":
    generate_symbol_list()
