# ui/messages.py
"""
All UI text messages for the Crypto Signal Hunter application.
Simple English-only version for better code organization.
"""

class Messages:
    # Dashboard Title and Header
    DASHBOARD_TITLE = "🤖 Crypto Co-pilot AI"
    DASHBOARD_SUBTITLE = "Your Intelligent Trading Assistant for Cryptocurrency Markets"
    
    # Welcome Section
    WELCOME_SECTION = """
    # 🤖 Crypto Co-pilot AI
    ### Your Intelligent Trading Assistant for Cryptocurrency Markets
    
    ---
    
    ## 🎯 What This System Does
    
    **Crypto Co-pilot AI** is an advanced machine learning system that analyzes cryptocurrency market data to provide intelligent trading signals. Here's how it helps you:
    
    ### 🔮 Key Features:
    
    - **📊 Advanced Technical Analysis**: Combines multiple indicators (RSI, MACD, Bollinger Bands, Moving Averages, Volume Analysis)
    - **🧠 Machine Learning Predictions**: Uses RandomForest algorithm to predict market movements
    - **🎯 Dual Strategy Modes**: Choose between Dynamic (volatility-based) or Fixed (percentage-based) strategies
    - **⚡ Real-time Signals**: Get instant BUY/SELL/HOLD recommendations
    - **📈 Performance Tracking**: Monitor accuracy and model performance on historical data
    - **🔄 Adaptive Learning**: Model adjusts to market conditions automatically
    
    ### 🚀 How It Works:
    
    1. **Data Collection**: Fetches real-time market data from CoinGecko API
    2. **Technical Analysis**: Calculates 15+ technical indicators
    3. **ML Training**: Trains RandomForest model on historical patterns
    4. **Signal Generation**: Provides three types of signals:
       - 🟢 **TAKE PROFIT**: Strong bullish signal (recommended BUY)
       - 🔴 **STOP LOSS**: Strong bearish signal (recommended SELL)
       - ⚪ **TIME LIMIT**: Neutral signal (recommended HOLD)
    5. **Performance Validation**: Shows accuracy on test data
    
    ### 💡 Benefits for You:
    
    - **Save Time**: No need to manually analyze complex charts
    - **Reduce Emotions**: Data-driven decisions based on AI analysis
    - **Risk Management**: Built-in stop-loss and take-profit levels
    - **Transparency**: See exactly how decisions are made
    - **Flexibility**: Customize strategy parameters to your risk tolerance
    
    ---
    
    ## 🛠️ Getting Started
    
    1. **Configure Settings**: Use the sidebar to select cryptocurrency, time range, and strategy
    2. **Choose Strategy Mode**: 
       - 🔄 **Dynamic**: Targets based on market volatility (adaptive)
       - 🎯 **Fixed**: Fixed percentage targets (predictable)
    3. **Set Parameters**: Adjust take-profit, stop-loss, and time limits
    4. **Click "Analyze Now"**: Let AI analyze the market and provide signals
    
    ---
    """
    
    # Risk Disclaimer
    RISK_DISCLAIMER = """
    ⚠️ **IMPORTANT RISK DISCLAIMER**
    
    This system is provided for EDUCATIONAL and RESEARCH purposes only. Please read carefully:
    
    **Investment Risks:**
    - Cryptocurrency trading involves HIGH RISK and can result in significant financial losses
    - Past performance does not guarantee future results
    - AI predictions are not 100% accurate and can be wrong
    - Market conditions can change rapidly and unpredictably
    
    **System Limitations:**
    - This tool is for ASSISTANCE only, not financial advice
    - Always do your own research (DYOR) before making investment decisions
    - Consider consulting with qualified financial advisors
    - Never invest more than you can afford to lose
    
    **Legal Notice:**
    - We are not licensed financial advisors
    - You are solely responsible for your trading decisions
    - Use this system at your own risk
    - We disclaim all liability for any losses incurred
    
    **By using this system, you acknowledge that you understand these risks and agree to use it responsibly.**
    """
    
    # Getting Started Message
    GETTING_STARTED_MSG = "👈 **Ready to start?** Configure your analysis settings in the sidebar and click 'Analyze Now'"
    
    # Analysis Messages
    ANALYSIS_HEADER = "Analysis for: {name} ({currency})"
    FETCHING_DATA_MSG = "Fetching data and training AI model..."
    
    # Strategy Messages
    DYNAMIC_STRATEGY_MSG = "🔄 **Dynamic Strategy Active** | TP: {tp_multiplier}x volatility | SL: {sl_multiplier}x volatility | Time: {time_barrier_days} days"
    FIXED_STRATEGY_MSG = "🎯 **Fixed Strategy Active** | TP: {take_profit_pct:.1f}% | SL: {stop_loss_pct:.1f}% | Time: {time_barrier_days} days"
    
    # Error Messages
    MODEL_TRAINING_ERROR = "🚨 AI Model could not be trained."
    INSUFFICIENT_DATA_MSG = "Not enough data remained or all outcomes were the same. Try adjusting parameters."
    UNEXPECTED_ERROR_MSG = "An unexpected error occurred: {error}"
    DATA_RETRIEVAL_ERROR = "Could not retrieve market data."
    COIN_LIST_ERROR = "Could not fetch coin list."
    
    # Display Components Messages
    # AI Prediction Section
    AI_PREDICTION_TITLE = "🔮 AI Strategy Prediction"
    AI_SIGNAL_TITLE = "🚨 AI Signal for Next Trading Period"
    MODEL_CONFIDENCE_TITLE = "🧠 Model Confidence Levels"
    
    # Metrics Labels
    CURRENT_PRICE_LABEL = "Current Price"
    MARKET_VOLATILITY_LABEL = "Market Volatility"
    MARKET_VOLATILITY_HELP = "20-day rolling volatility based on price changes"
    MODEL_ACCURACY_LABEL = "Model Accuracy"
    MODEL_ACCURACY_HELP = "Accuracy on historical test data"
    CONFIDENCE_HELP = "AI confidence level for this outcome"
    
    # Strategy Info
    STRATEGY_MODE_ACTIVE = "{mode_emoji} **{mode} Strategy Mode** is active"
    
    # Prediction Map
    PREDICTION_MAP = {
        1.0: "TAKE PROFIT HIT 🟢", 
        -1.0: "STOP LOSS HIT 🔴", 
        0.0: "TIME LIMIT HIT ⚪️"
    }
    
    # Signal Analysis Messages
    BULLISH_SIGNAL = """
    **📈 BULLISH SIGNAL - RECOMMENDED ACTION: BUY**
    
    **AI Analysis:**
    - Current market volatility: {volatility:.2f}%
    - Take Profit target: ${tp_price:.4f} (+{tp_percentage:.2f}%)
    - Stop Loss level: ${sl_price:.4f} (-{sl_percentage:.2f}%)
    - Analysis period: {time_barrier_days} days
    - Expected outcome: Price likely to reach profit target before stop loss
    
    **Risk/Reward Ratio:** {risk_reward:.2f}:1
    """
    
    BEARISH_SIGNAL = """
    **📉 BEARISH SIGNAL - RECOMMENDED ACTION: SELL**
    
    **AI Analysis:**
    - Current market volatility: {volatility:.2f}%
    - Stop Loss target: ${sl_price:.4f} (-{sl_percentage:.2f}%)
    - Take Profit level: ${tp_price:.4f} (+{tp_percentage:.2f}%)
    - Analysis period: {time_barrier_days} days
    - Expected outcome: Price likely to hit stop loss before take profit
    
    **Risk/Reward Ratio:** {risk_reward:.2f}:1
    """
    
    NEUTRAL_SIGNAL = """
    **⚪ NEUTRAL SIGNAL - RECOMMENDED ACTION: HOLD**
    
    **AI Analysis:**
    - Current market volatility: {volatility:.2f}%
    - Take Profit target: ${tp_price:.4f} (+{tp_percentage:.2f}%)
    - Stop Loss level: ${sl_price:.4f} (-{sl_percentage:.2f}%)
    - Analysis period: {time_barrier_days} days
    - Expected outcome: Neither target likely to be reached within time limit
    
    **Market Assessment:** Sideways/consolidation movement expected
    """
    
    # Trading Risk Disclaimer
    TRADING_RISK_DISCLAIMER = """
    ⚠️ **TRADING RISK DISCLAIMER**
    
    This prediction is based on historical data analysis and is NOT guaranteed to be accurate. 
    Cryptocurrency markets are highly volatile and unpredictable. Always:
    
    - Use proper risk management
    - Never invest more than you can afford to lose
    - Consider this as ONE factor in your trading decision
    - Do your own research (DYOR)
    - Past performance does not guarantee future results
    """
    
    # Model Performance Section
    MODEL_PERFORMANCE_TITLE = "🔬 Detailed Model Performance"
    PERFORMANCE_METRICS_TITLE = "📊 Performance Metrics by Signal Type"
    
    # Performance Metrics Explanation
    METRICS_EXPLANATION = """
    **📖 Metrics Explanation:**
    
    **Precision** - Signal Quality
    - How often the AI is correct when it gives this signal
    - Higher = fewer false signals
    
    **Recall** - Opportunity Detection
    - How often the AI catches real opportunities
    - Higher = catches more opportunities
    
    **F1-Score** - Overall Balance
    - Balanced measure of precision and recall
    - Range: 0.0 to 1.0 (higher is better)
    
    **Support** - Data Points
    - Number of historical instances of this outcome
    - More support = more reliable statistics
    """
    
    # Performance Summary
    OVERALL_ACCURACY_MSG = "🎯 **Overall Model Accuracy: {accuracy:.1%}**"
    PERFORMANCE_NOTE = "**Note:** These metrics are calculated on historical test data and may not reflect future performance."
    
    # Error Visualization Section
    ERROR_VISUALIZATION_TITLE = "📊 Prediction Analysis on Test Data"
    NO_TEST_DATA_MSG = "No test data available to visualize."
    
    # Chart Labels
    CHART_TITLE = "AI Prediction Performance on Historical Test Data"
    CHART_PRICE_LABEL = "Price"
    CHART_DATE_LABEL = "Date"
    CHART_PRICE_AXIS = "Price (USD)"
    
    # Prediction Labels
    CORRECT_BUY_LABEL = "Correct BUY ✅"
    INCORRECT_BUY_LABEL = "Incorrect BUY ❌"
    CORRECT_SELL_LABEL = "Correct SELL ✅"
    INCORRECT_SELL_LABEL = "Incorrect SELL ❌"
    
    # Hover Templates
    CORRECT_BUY_HOVER = '<b>Correct BUY Signal</b><br>Date: %{x}<br>Price: $%{y:.4f}<extra></extra>'
    INCORRECT_BUY_HOVER = '<b>Incorrect BUY Signal</b><br>Date: %{x}<br>Price: $%{y:.4f}<extra></extra>'
    CORRECT_SELL_HOVER = '<b>Correct SELL Signal</b><br>Date: %{x}<br>Price: $%{y:.4f}<extra></extra>'
    INCORRECT_SELL_HOVER = '<b>Incorrect SELL Signal</b><br>Date: %{x}<br>Price: $%{y:.4f}<extra></extra>'
    
    # Test Data Summary
    TEST_DATA_SUMMARY = "📈 **Test Data Summary:** {correct}/{total} correct predictions ({accuracy:.1%} accuracy)"
    
    # Signal Class Labels
    SIGNAL_CLASS_LABELS = {
        '-1.0': 'STOP LOSS 🔴', 
        '0.0': 'TIME OUT ⚪️', 
        '1.0': 'TAKE PROFIT 🟢'
    }
    
    # Sidebar Configuration Messages
    SIDEBAR_TITLE = "🛠️ Configuration"
    DATA_SETTINGS_TITLE = "Data Settings"
    STRATEGY_MODE_TITLE = "🎯 Strategy Mode"
    DYNAMIC_STRATEGY_TITLE = "🔄 Dynamic Strategy Settings"
    FIXED_STRATEGY_TITLE = "🎯 Fixed Strategy Settings"
    TIME_SETTINGS_TITLE = "⏰ Time Settings"
    ANALYZE_NOW_BUTTON = "🚀 Analyze Now"
    
    # Form Labels
    CRYPTOCURRENCY_LABEL = "Cryptocurrency:"
    DATE_RANGE_LABEL = "Date Range (Days):"
    QUOTE_CURRENCY_LABEL = "Quote Currency:"
    STRATEGY_MODE_LABEL = "Select Strategy Mode:"
    TAKE_PROFIT_MULTIPLIER_LABEL = "Take Profit Multiplier:"
    STOP_LOSS_MULTIPLIER_LABEL = "Stop Loss Multiplier:"
    TIME_LIMIT_LABEL = "Time Limit (Days)"
    TAKE_PROFIT_TARGET_LABEL = "Take Profit Target:"
    STOP_LOSS_TARGET_LABEL = "Stop Loss Target:"
    PRICE_LABEL = "Price:"
    PERCENTAGE_LABEL = "Percentage:"
    CURRENT_PRICE_LABEL_SIDEBAR = "Current Price: {price:.4f} {currency}"
    
    # Help Messages
    STRATEGY_MODE_HELP = "Dynamic: Based on market volatility | Fixed: Based on your targets"
    TAKE_PROFIT_MULTIPLIER_HELP = "Higher values = more aggressive profit targets"
    STOP_LOSS_MULTIPLIER_HELP = "Higher values = more conservative stop losses"
    TIME_LIMIT_HELP = "Maximum number of days to wait for Take Profit or Stop Loss targets to be hit. If neither target is reached within this time, the signal becomes 'TIME LIMIT HIT' (neutral/hold signal)."
    
    # Strategy Mode Info
    DYNAMIC_STRATEGY_INFO = "Targets are calculated based on market volatility"
    FIXED_STRATEGY_INFO = "Current Price: {price:.4f} {currency}"
    
    # Expandable Help Section
    HOW_DOES_THIS_WORK = "ℹ️ How does this work?"
    DYNAMIC_STRATEGY_HELP_TEXT = """**🔄 Dynamic Strategy:**
• Targets based on market volatility
• Higher volatility = Higher targets
• Adapts to market conditions automatically
• Formula: `Target = Price × (1 ± Volatility × Multiplier)`

**Example:** If volatility is 2% and multiplier is 2.0:
- Take Profit: +4% from current price
- Stop Loss: -2% from current price"""
    
    FIXED_STRATEGY_HELP_TEXT = """**🎯 Fixed Strategy:**
• Fixed percentage targets set by user
• Consistent risk/reward ratio
• Predictable entry/exit points
• Formula: `Target = Price × (1 ± Percentage/100)`

**Example:** If you set 3% profit and 1.5% loss:
- Take Profit: +3% from current price
- Stop Loss: -1.5% from current price"""

    # ==========================================
    # RAW DATA DISPLAY MESSAGES
    # ==========================================
    
    # Raw Data Display Section
    RAW_DATA_EXPANDER_TITLE = "Show Raw Data & Indicators"
    
    # Tab Titles
    TAB_ALL_DATA = "📊 All Data"
    TAB_TEST_DATA = "🧪 Test Data Only"
    TAB_RECENT_DATA = "📈 Recent 50 Days"
    
    # Tab Headers
    COMPLETE_DATASET_TITLE = "Complete Dataset"
    TEST_DATASET_TITLE = "Test Dataset with Predictions"
    RECENT_DATASET_TITLE = "Recent 50 Days"
    
    # Status Messages
    NO_TEST_DATA_AVAILABLE = "No test data available"
    TEST_ACCURACY_METRIC = "Test Accuracy"
    TEST_ACCURACY_FORMAT = "{correct}/{total} ({accuracy:.1%})"
    
    # Data Explanation Section
    DATA_EXPLANATION_TITLE = "📖 Data Explanation"
    
    # Basic Data Column Group
    BASIC_DATA_GROUP_TITLE = "**📊 Basic Data:**"
    BASIC_DATA_PRICE_DESC = "- `price`: Current market price"
    BASIC_DATA_VOLUME_DESC = "- `volume`: Trading volume"
    
    # Target Prices Column Group
    TARGET_PRICES_GROUP_TITLE = "**🎯 Target Prices:**"
    TARGET_PRICES_TP_PRICE_DESC = "- `TP_Price`: Take Profit target price"
    TARGET_PRICES_TP_PERCENTAGE_DESC = "- `TP_Percentage`: Take Profit percentage"
    TARGET_PRICES_SL_PRICE_DESC = "- `SL_Price`: Stop Loss target price"
    TARGET_PRICES_SL_PERCENTAGE_DESC = "- `SL_Percentage`: Stop Loss percentage"
    
    # Status & Predictions Column Group
    STATUS_PREDICTIONS_GROUP_TITLE = "**📊 Status & Predictions:**"
    STATUS_PREDICTIONS_DATA_TYPE_DESC = "- `Data_Type`: Training 📚 or Test 🧪"
    STATUS_PREDICTIONS_AI_PREDICTION_DESC = "- `AI_Prediction`: Raw prediction (-1, 0, 1)"
    STATUS_PREDICTIONS_TARGET_ACTUAL_DESC = "- `Target_Actual`: Actual outcome"
    STATUS_PREDICTIONS_CORRECTNESS_DESC = "- `Prediction_Correctness`: AI accuracy"
    
    # Lagged Features Column Group
    LAGGED_FEATURES_GROUP_TITLE = "**📈 Lagged Features:**"
    LAGGED_FEATURES_LAG1_DESC = "- `_lag_1`: 1 day ago values"
    LAGGED_FEATURES_LAG2_DESC = "- `_lag_2`: 2 days ago values"
    LAGGED_FEATURES_LAG3_DESC = "- `_lag_3`: 3 days ago values"
    LAGGED_FEATURES_LAG5_DESC = "- `_lag_5`: 5 days ago values"
    
    # Signal Meanings Section
    SIGNAL_MEANINGS_TITLE = "🏷️ Signal Meanings"
    
    # Take Profit Signal
    TAKE_PROFIT_SIGNAL_TITLE = "**🟢 TAKE PROFIT:**"
    TAKE_PROFIT_SIGNAL_PRICE_DESC = "- Price reached profit target"
    TAKE_PROFIT_SIGNAL_AI_DESC = "- AI suggests buying"
    TAKE_PROFIT_SIGNAL_TYPE_DESC = "- Bullish signal"
    
    # Stop Loss Signal
    STOP_LOSS_SIGNAL_TITLE = "**🔴 STOP LOSS:**"
    STOP_LOSS_SIGNAL_PRICE_DESC = "- Price hit stop loss level"
    STOP_LOSS_SIGNAL_AI_DESC = "- AI suggests selling"
    STOP_LOSS_SIGNAL_TYPE_DESC = "- Bearish signal"
    
    # Time Limit Signal
    TIME_LIMIT_SIGNAL_TITLE = "**⚪️ TIME LIMIT:**"
    TIME_LIMIT_SIGNAL_PRICE_DESC = "- Neither target reached"
    TIME_LIMIT_SIGNAL_BARRIER_DESC = "- Time barrier hit"
    TIME_LIMIT_SIGNAL_TYPE_DESC = "- Neutral/Hold signal"
    
    # Strategy Explanation Section
    STRATEGY_EXPLANATION_TITLE = "🧠 Strategy Explanation"
    
    # Dynamic Strategy Explanation
    DYNAMIC_STRATEGY_EXPLANATION_TITLE = "**🔄 Dynamic Strategy:**"
    DYNAMIC_STRATEGY_EXPLANATION_TEXT = """
    - Targets calculated based on market volatility
    - Higher volatility = Higher targets
    - Adapts to market conditions automatically
    - Formula: `Target = Price × (1 ± Volatility × Multiplier)`
    """
    
    # Fixed Strategy Explanation
    FIXED_STRATEGY_EXPLANATION_TITLE = "**🎯 Fixed Strategy:**"
    FIXED_STRATEGY_EXPLANATION_TEXT = """
    - Fixed percentage targets set by user
    - Consistent targets regardless of market conditions
    - More predictable risk/reward ratio
    - Formula: `Target = Price × (1 ± Percentage/100)`
    """
    
    # Parameters Summary Section
    PARAMETERS_SUMMARY_TITLE = "🔧 Parameters Summary"
    STRATEGY_PARAMETERS_TITLE = "**Strategy Parameters:**"
    MODEL_PARAMETERS_TITLE = "**Model Parameters:**"
    TECHNICAL_INDICATOR_PARAMETERS_TITLE = "**Technical Indicator Parameters:**"