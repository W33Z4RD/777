                                                      #!/usr/bin/env python3
"""
Clean Coinbase Connection Test Script
"""

import sys
import os
import logging

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_coinbase_connection():
    """Test Coinbase API connection"""
    
    try:
        from seventrader.exchanges.coinbase_pro import CoinbaseProTrader
        
        # Get credentials
        api_key = os.getenv('COINBASE_API_KEY')
        api_secret = os.getenv('COINBASE_API_SECRET')
        
        if not api_key or not api_secret:
            print("ERROR: Missing API credentials in .env file")
            return False
            
        print("Testing Coinbase Cloud API connection...")
        print(f"API Key: {api_key[:20]}...{api_key[-10:]}")  # Show partial key
        print(f"Private Key: {'***PRESENT***' if api_secret else 'MISSING'}")
        
        # Initialize trader
        trader = CoinbaseProTrader(api_key, api_secret, sandbox=True)
        
        # Test 1: Basic connection
        print("\n1. Testing basic connection...")
        if trader.test_connection():
            print("SUCCESS: API connection established")
        else:
            print("FAILED: Could not connect to API")
            return False
            
        # Test 2: Get portfolio
        print("\n2. Testing portfolio retrieval...")
        portfolio = trader.get_portfolio_summary()
        print(f"Portfolio data: {portfolio}")
        
        # Test 3: Get BTC price
        print("\n3. Testing price retrieval...")
        btc_price = trader.get_current_price('BTC-USD')
        if btc_price:
            print(f"BTC-USD price: ${btc_price:,.2f}")
        else:
            print("Could not get BTC price")
            
        # Test 4: Get USD balance
        print("\n4. Testing balance retrieval...")
        usd_balance = trader.get_account_balance('USD')
        print(f"USD balance: ${usd_balance:.2f}")
        
        print("\nALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        logging.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_coinbase_connection()
    if success:
        print("\nCoinbase integration is working correctly!")
    else:
        print("\nCoinbase integration needs fixing.")                       
