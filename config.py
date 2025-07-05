# config.py
"""
This file contains all the configuration parameters for the project,
including technical indicator settings and the trading strategy parameters.
"""

# Technical Indicator Parameters
SMA_SHORT_PERIOD = 10
SMA_LONG_PERIOD = 30
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BBANDS_LENGTH = 20
BBANDS_STD = 2.0
ATR_PERIOD = 14

# Triple Barrier Method Strategy Parameters
# These will be the default values, which can be overridden by user input in the UI.
TAKE_PROFIT_PCT = 3.0  # Take profit at +3%
STOP_LOSS_PCT = 1.5   # Stop loss at -1.5%
TIME_BARRIER_DAYS = 10  # Hold for a maximum of 10 days