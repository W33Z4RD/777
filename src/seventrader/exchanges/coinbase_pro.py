# src/seventrader/exchanges/coinbase_pro.py
# src/seventrader/exchanges/coinbase_pro.py

import os
import json
import time
import logging
import requests
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


@dataclass
class TradeResult:
    """Trade execution result"""
    success: bool
    order_id: Optional[str] = None
    filled_price: float = 0.0
    filled_size: float = 0.0
    fee: float = 0.0
    error: Optional[str] = None


class CoinbaseProTrader:
    """Coinbase Cloud API integration using JWT"""

    def __init__(self, api_key: str, api_secret: str, passphrase: str = None, sandbox: bool = True):
        self.api_key = api_key  # Full API key string (organizations/...)
        self.api_key_uuid = api_key.split('/')[-1]  # Extract the UUID
        self.api_secret = api_secret
        self.sandbox = sandbox
        
        # Use the correct base URL for Coinbase Cloud API
        if self.sandbox:
            self.base_url = "https://api-sandbox.coinbase.com"
        else:
            self.base_url = "https://api.coinbase.com"
        
        logging.info(f"CoinbaseProTrader initialized (sandbox: {self.sandbox})")

    def _generate_jwt(self, method: str, path: str, body: str = "") -> str:
        """Generate a JWT for Coinbase Cloud API authentication."""
        try:
            # Fix newline replacement - double escape issue
            private_key_str = self.api_secret.replace('\\n', '\n')
            private_key_bytes = private_key_str.encode('utf-8')
            
            # Load the private key
            private_key = serialization.load_pem_private_key(
                private_key_bytes,
                password=None,
                backend=default_backend()
            )

            # Current time
            now = int(time.time())
            
            # JWT payload for Coinbase Cloud API
            jwt_payload = {
                'sub': self.api_key,  # Full API key as subject
                'iss': 'coinbase-cloud',
                'nbf': now,
                'exp': now + 120,  # 2 minutes expiration
                'aud': ['retail_rest_api_proxy'],
                'uri': f"{method} {path}"  # Method and path
            }

            logging.info(f"JWT Payload: {jwt_payload}")

            # Generate JWT with proper headers
            token = jwt.encode(
                jwt_payload,
                private_key,
                algorithm='ES256',
                headers={
                    'kid': self.api_key_uuid,  # API key UUID
                    'nonce': str(uuid.uuid4())
                }
            )
            
            return token
            
        except Exception as e:
            logging.error(f"JWT generation error: {e}")
            raise

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Optional[Dict]:
        """Make an authenticated API request to Coinbase Cloud."""
        url = f"{self.base_url}{endpoint}"
        body_json = json.dumps(data) if data else ""

        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'CoinbaseCloudTrader/1.0'
            }

            if not self.sandbox:
                # Generate JWT token
                token = self._generate_jwt(method, endpoint, body_json)
                headers['Authorization'] = f'Bearer {token}'

            # Log request details (without sensitive data)
            logging.debug(f"Making {method} request to {endpoint}")

            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, headers=headers, data=body_json, timeout=30)
            else:
                logging.error(f"Unsupported HTTP method: {method}")
                return None

            # Handle response
            if response.status_code in [200, 201]:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code} - {response.text}"}

        except Exception as e:
            logging.error(f"Request error: {e}")
            return None

    def get_account_balance(self, currency: str = 'USD') -> float:
        """Get account balance for a specific currency."""
        accounts_response = self._make_request('GET', '/api/v3/brokerage/accounts')
        
        if not accounts_response:
            logging.error("Failed to get accounts response")
            return None
            
        if 'accounts' not in accounts_response:
            logging.error(f"No accounts in response: {accounts_response}")
            return None

        for account in accounts_response['accounts']:
            if account.get('currency') == currency:
                available_balance = account.get('available_balance', {})
                return float(available_balance.get('value', 0.0))
                
        logging.warning(f"Currency {currency} not found in accounts")
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current price for a symbol (e.g., 'BTC-USD')."""
        try:
            # Use the correct endpoint for product ticker
            response = self._make_request('GET', f'/api/v3/brokerage/products/{symbol}/ticker')
            
            if response and 'price' in response:
                return float(response['price'])
            else:
                logging.warning(f"No price data for {symbol}: {response}")
                return None
                
        except Exception as e:
            logging.error(f"Error getting price for {symbol}: {e}")
            return None

    def place_market_order(self, symbol: str, side: str, size_in_quote: float) -> TradeResult:
        """Place a market order using quote currency size (e.g., USD)."""
        client_order_id = str(uuid.uuid4())
        
        order_data = {
            "client_order_id": client_order_id,
            "product_id": symbol,
            "side": side.upper(),
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": str(size_in_quote)
                }
            }
        }

        logging.info(f"Placing {side} order for {symbol}: ${size_in_quote}")
        response = self._make_request('POST', '/api/v3/brokerage/orders', order_data)

        if response and response.get('success'):
            return TradeResult(
                success=True,
                order_id=response.get('order_id', client_order_id),
                filled_price=0,  # Not immediately available
                filled_size=0,   # Not immediately available
                fee=0,           # Not immediately available
                error=None
            )
        else:
            error_msg = "Unknown error"
            if response:
                error_msg = response.get('error_response', {}).get('message', str(response))
            
            return TradeResult(
                success=False,
                error=f"Order failed: {error_msg}"
            )

    def execute_signal(self, symbol: str, signal_type: str, balance_usd: float, position_size_pct: float = 0.02) -> TradeResult:
        """Execute a trading signal."""
        cb_symbol = self._convert_symbol(symbol)
        if not cb_symbol:
            return TradeResult(success=False, error=f"Unsupported symbol: {symbol}")

        position_value_usd = balance_usd * position_size_pct
        if position_value_usd < 10:  # Coinbase minimum order size
            return TradeResult(success=False, error="Position size too small")

        if signal_type in ['BUY', 'STRONG_BUY']:
            return self.place_market_order(cb_symbol, 'buy', position_value_usd)
        elif signal_type in ['SELL', 'STRONG_SELL']:
            return self.place_market_order(cb_symbol, 'sell', position_value_usd)
        else:
            return TradeResult(success=False, error="No action for HOLD signal")

    def _convert_symbol(self, symbol: str) -> Optional[str]:
        """Convert symbol to Coinbase format (e.g., BTC/USDT -> BTC-USD)."""
        try:
            if '/' not in symbol:
                return symbol  # Already in correct format
                
            base, quote = symbol.split('/')
            
            # Convert common quote currencies to USD
            if quote in ['USDT', 'USDC']:
                return f"{base}-USD"
            else:
                return f"{base}-{quote}"
                
        except Exception as e:
            logging.error(f"Error converting symbol {symbol}: {e}")
            return None

    def get_portfolio_summary(self) -> Dict:
        """Get a summary of the portfolio."""
        accounts_response = self._make_request('GET', '/api/v3/brokerage/accounts')
        
        if not accounts_response or 'accounts' not in accounts_response:
            logging.error("Failed to get portfolio data")
            return {'total_value_usd': 0.0, 'balances': {}}

        total_value = 0.0
        balances = {}

        try:
            for account in accounts_response['accounts']:
                currency = account.get('currency')
                available_balance = account.get('available_balance', {})
                balance = float(available_balance.get('value', 0.0))

                if balance > 0:
                    balances[currency] = balance
                    
                    if currency == 'USD':
                        total_value += balance
                    else:
                        # For non-USD assets, try to get the price
                        try:
                            price = self.get_current_price(f"{currency}-USD")
                            if price and price > 0:
                                total_value += balance * price
                        except Exception as e:
                            logging.warning(f"Could not get price for {currency}: {e}")

        except Exception as e:
            logging.error(f"Error processing portfolio data: {e}")

        return {
            'total_value_usd': total_value,
            'balances': balances
        }

    def test_connection(self) -> (bool, str):
        """Test the API connection."""
        try:
            accounts = self._make_request('GET', '/api/v3/brokerage/accounts')
            if accounts and 'accounts' in accounts:
                return True, "Connection successful."
            elif accounts and 'error' in accounts:
                if "401" in accounts['error']:
                    return False, f"Connection failed: {accounts['error']}. Please check your API key and secret."
                else:
                    return False, f"Connection failed: {accounts['error']}"
            else:
                return False, "Connection failed: Unknown error."
        except Exception as e:
            logging.error(f"Connection test failed: {e}")
            return False, f"Connection test failed: {e}"
