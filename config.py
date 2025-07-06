# config.py
"""
This file contains all configuration parameters for the project.
Enhanced Version with Advanced Features - Updated with Dual Strategy Support.
"""

# --- Default UI Input Parameters ---
DEFAULT_COIN_NAME = "Bitcoin"
DEFAULT_DAYS_TO_FETCH = 365
DEFAULT_CURRENCY = "usd"

# --- Strategy Mode Configuration ---
DEFAULT_STRATEGY_MODE = 'Dynamic'
VOLATILITY_WINDOW = 20
VOLATILITY_TAKE_PROFIT_MULTIPLIER = 2.5
VOLATILITY_STOP_LOSS_MULTIPLIER = 1.2

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

# --- Advanced Technical Indicators ---
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
ATR_PERIOD = 14
CCI_PERIOD = 14
WILLIAMS_R_PERIOD = 14

# --- Feature Engineering Parameters ---
PRICE_CHANGE_PERIODS = [1, 3, 5, 7]
VOLUME_CHANGE_PERIODS = [1, 3, 5, 7]
ROLLING_CORRELATION_PERIOD = 10

# --- Enhanced Model Parameters ---
RANDOM_FOREST_ESTIMATORS = 200
RANDOM_FOREST_MAX_DEPTH = 10
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 5

# --- Triple Barrier Method Strategy Parameters (DAILY) ---
# Fixed Strategy (Original)
TAKE_PROFIT_PCT = 4.0
STOP_LOSS_PCT = 2.0
TIME_BARRIER_DAYS = 8

# --- Optimized Strategy Parameters ---
OPTIMIZED_CONFIG = {
    'Dynamic': {
        'tp_multiplier': 2.5,
        'sl_multiplier': 1.2,
        'time_barrier_days': 7,
        'description': 'بهینه شده برای بازارهای پرنوسان'
    },
    'Fixed': {
        'take_profit_pct': 4.0,
        'stop_loss_pct': 2.0,
        'time_barrier_days': 8,
        'description': 'بهینه شده برای استراتژی محافظه‌کارانه'
    }
}

# --- Risk Management Parameters ---
MAX_DRAWDOWN_THRESHOLD = 0.15  # حداکثر 15% افت
MIN_SHARPE_RATIO = 0.5  # حداقل نسبت شارپ
MIN_WIN_RATE = 0.45  # حداقل نرخ برد 45%

# --- Feature Selection Parameters ---
MAX_FEATURES_TO_SELECT = 50  # حداکثر تعداد ویژگی‌ها
FEATURE_IMPORTANCE_THRESHOLD = 0.001  # حداقل اهمیت ویژگی

# --- Model Validation Parameters ---
TIME_SERIES_CV_SPLITS = 5  # تعداد تقسیم‌بندی برای اعتبارسنجی
TEST_SIZE_RATIO = 0.2  # نسبت داده‌های تست

# --- Advanced Strategy Settings ---
ENABLE_PARAMETER_OPTIMIZATION = True  # فعال‌سازی بهینه‌سازی پارامترها
ENABLE_ENSEMBLE_MODEL = True  # استفاده از مدل ترکیبی
ENABLE_FEATURE_SELECTION = True  # فعال‌سازی انتخاب ویژگی

# --- Performance Thresholds ---
PERFORMANCE_THRESHOLDS = {
    'excellent': {'win_rate': 0.65, 'sharpe_ratio': 1.0},
    'good': {'win_rate': 0.55, 'sharpe_ratio': 0.7},
    'acceptable': {'win_rate': 0.45, 'sharpe_ratio': 0.5},
    'poor': {'win_rate': 0.35, 'sharpe_ratio': 0.3}
}

# --- UI Display Settings ---
DISPLAY_RECENT_PREDICTIONS = 20  # تعداد پیش‌بینی‌های اخیر برای نمایش
CHART_HEIGHT = 500  # ارتفاع نمودارها
ENABLE_DARK_THEME = True  # استفاده از تم تیره

# --- API Configuration ---
API_TIMEOUT = 30  # زمان انتظار برای درخواست‌های API
MAX_RETRIES = 3  # حداکثر تعداد تلاش مجدد
RETRY_DELAY = 2  # تأخیر بین تلاش‌ها (ثانیه)

# --- Caching Settings ---
COIN_LIST_CACHE_TTL = 86400  # 24 ساعت
MARKET_DATA_CACHE_TTL = 300  # 5 دقیقه

# --- Logging Configuration ---
LOG_LEVEL = 'INFO'
ENABLE_PERFORMANCE_LOGGING = True
LOG_FILE_PATH = 'crypto_analyzer.log'