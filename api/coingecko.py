# api/coingecko.py
"""
This module acts as a client for the CoinGecko API.

It provides functions to fetch the list of all available cryptocurrencies
and to retrieve historical market data for a specific coin. It includes
caching to improve performance and avoid hitting API rate limits.
"""

import requests
import streamlit as st

# The base URL for all CoinGecko API v3 endpoints.
BASE_URL = "https://api.coingecko.com/api/v3"

# Cache the list of all coins for a full day (86400 seconds) as it rarely changes.
# This prevents re-fetching this large list on every script rerun.
@st.cache_data(ttl=86400)
def get_all_coins():
    """
    Fetches and caches the list of all supported coins from CoinGecko.

    Returns:
        dict: A dictionary mapping coin names to their unique CoinGecko IDs.
              Returns an empty dictionary if the API call fails.
    """
    url = f"{BASE_URL}/coins/list"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return {coin['name']: coin['id'] for coin in response.json()}
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching coin list: {e}")
        return {}

# Cache market data for 5 minutes (300 seconds) to allow for fresh data
# while still preventing excessive calls during interaction.
@st.cache_data(ttl=300)
def get_market_data(coin_id, vs_currency, days):
    """
    Fetches historical market data (price and volume) for a specific coin.

    Args:
        coin_id (str): The unique ID of the coin (e.g., 'bitcoin').
        vs_currency (str): The currency to compare against (e.g., 'usd').
        days (int): The number of days of historical data to fetch.

    Returns:
        dict: A dictionary containing 'prices' and 'total_volumes' if successful,
              otherwise None.
    """
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"
    
    params = {
        'vs_currency': vs_currency,
        'days': days
    }
    
    # For longer timeframes, CoinGecko provides daily granularity.
    if days > 90:
        params['interval'] = 'daily'
        
    # Some APIs, including CoinGecko, may block requests without a standard
    # User-Agent header. This mimics a request from a web browser.
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching market data for {coin_id}: {e}")
        return None