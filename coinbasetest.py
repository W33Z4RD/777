
         import os
import json
import time
import logging
import requests
import uuid

import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
# --- REPLACE THESE WITH YOUR ACTUAL VALUES ---
API_KEY = "organizations/3ac15b87-b6d8-4c77-acdd-1bc4e457a348/apiKeys/13e432bb-22d1-4be8-8db4-b0428832ea3d" # e.g., "organizations/..."
PRIVATE_KEY = 	"-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIO6oflfNHe0aPPoJmhvl2Oe9m8Sqqp+jIG8L4k5WMWzRoAoGCCqGSM49\nAwEHoUQDQgAETv820TvKZH4Iid5Ewrs1VeI4XFV95ueXXHxiA6igBR0TGIxOd5B5\nbAZHlA2vrO20ZUibnPOIeSKJWvlFKFA9mA==\n-----END EC PRIVATE KEY-----\n" # e.g., "-----BEGIN EC PRIVATE KEY-----..."
# ---------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def generate_jwt(method: str, path: str, body: str = "") -> str:
    private_key_bytes = PRIVATE_KEY.replace("\n", "\n").encode('utf-8')
    private_key = serialization.load_pem_private_key(
        private_key_bytes,
        password=None,
        backend=default_backend()
    )

    jwt_payload = {
        'sub': API_KEY,
        'iss': 'coinbase-cloud',
        'nbf': int(time.time()),
        'exp': int(time.time()) + 60,
        'aud': ['retail_rest_api_proxy'],
        'uri': f"{method} {path}"
    }

    token = jwt.encode(
        jwt_payload,
        private_key,
        algorithm='ES256',
        headers={'kid': API_KEY.split('/')[-1], 'nonce': str(uuid.uuid4())}
    )
    return token

def make_request(method: str, endpoint: str, data: dict = None) -> dict:
    url = f"https://api.coinbase.com{endpoint}"
    body_json = json.dumps(data) if data else ""

    token = generate_jwt(method, endpoint, body_json)

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, timeout=30)
        elif method == 'POST':
            response = requests.post(url, headers=headers, data=body_json, timeout=30)
        else:
            return {"error": "Unsupported method"}

        if response.status_code in [200, 201]:
            return response.json()
        else:
            logging.error(f"API error: {response.status_code} - {response.text}")
            return {"error": f"API error: {response.status_code} - {response.text}"}

    except Exception as e:
        logging.error(f"Request error: {e}")
        return {"error": f"Request error: {e}"}

if __name__ == "__main__":
    print("Attempting to fetch Coinbase accounts...")
    accounts = make_request('GET', '/api/v3/brokerage/accounts')
    print("\n--- API Response ---")
    print(json.dumps(accounts, indent=2))
    print("--------------------")
    if "error" in accounts:
        print("\n❌ Failed to fetch accounts. Please check your API key, private key, and permissions." )
    else:
        print("\n✅ Successfully fetched accounts!" )
        total_usd_value = 0.0
        for account in accounts.get('accounts', []):
            currency = account.get('currency')
            balance = float(account.get('available_balance', {}).get('value', 0.0))
            print(f"  {currency}: {balance}")
            if currency == 'USD': # Simple sum for USD, more complex for other assets
                total_usd_value += balance
        print(f"  Total USD Value (simple sum): {total_usd_value:.2f}")
