# config.py
"""
This file contains all configuration parameters for the project.
Final Stable Version for Portfolio - Updated with Dual Strategy Support.
"""

# --- Default UI Input Parameters ---
DEFAULT_COIN_NAME = "Bitcoin"
DEFAULT_DAYS_TO_FETCH = 365
DEFAULT_CURRENCY = "usd"

# --- Strategy Mode Configuration ---
DEFAULT_STRATEGY_MODE = 'Dynamic'
VOLATILITY_WINDOW = 20
VOLATILITY_TAKE_PROFIT_MULTIPLIER = 2.0
VOLATILITY_STOP_LOSS_MULTIPLIER = 1.0

# --- Technical Indicator Parameters (DAILY) ---
SMA_EXTRA_SHORT_PERIOD = 5
SMA_SHORT_PERIOD = 10
SMA_LONG_PERIOD = 30
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BBANDS_LENGTH = 20
BBANDS_STD = 2.0
VOLUME_SMA_PERIOD = 20

# --- Triple Barrier Method Strategy Parameters (DAILY - Fixed Percentage) ---
TAKE_PROFIT_PCT = 3.0
STOP_LOSS_PCT = 1.5
TIME_BARRIER_DAYS = 10