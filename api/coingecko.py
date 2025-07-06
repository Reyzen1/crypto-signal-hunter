# api/coingecko.py
"""
This module acts as a client for the CoinGecko API.
Version 2.0: Added retry logic for network requests.
"""

import requests
import streamlit as st
import time  # Import the time module for sleeping

# The base URL for all CoinGecko API v3 endpoints.
BASE_URL = "https://api.coingecko.com/api/v3"

# Cache the list of all coins for a full day (86400 seconds)
@st.cache_data(ttl=86400)
def get_all_coins():
    """
    Fetches and caches the list of all supported coins from CoinGecko.
    Includes a retry mechanism to handle transient network errors.
    """
    url = f"{BASE_URL}/coins/list"
    retries = 3
    for i in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            return {coin['name']: coin['id'] for coin in response.json()}
        except requests.exceptions.RequestException as e:
            st.warning(f"Attempt {i + 1} failed: {e}")
            if i < retries - 1:  # If it's not the last attempt
                time.sleep(2)  # Wait for 2 seconds before the next retry
                continue
            else:  # If it was the last attempt
                st.error("Could not fetch coin list after multiple attempts.")
                return {}

# Cache market data for 5 minutes (300 seconds)
@st.cache_data(ttl=300)
def get_market_data(coin_id, vs_currency, days):
    """
    Fetches historical market data for a specific coin.
    Includes a retry mechanism.
    """
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    params = {'vs_currency': vs_currency, 'days': days}
    if days > 90:
        params['interval'] = 'daily'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 1.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    retries = 3
    for i in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.warning(f"Attempt {i + 1} to fetch market data failed: {e}")
            if i < retries - 1:
                time.sleep(2)
                continue
            else:
                st.error(f"Could not fetch market data for {coin_id} after multiple attempts.")
                return None