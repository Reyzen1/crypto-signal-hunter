# بهبودهای اضافی برای افزایش دقت

# 1. اضافه کردن به config.py
"""
تنظیمات جدید برای بهبود عملکرد
"""

# Advanced Technical Indicators
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
ATR_PERIOD = 14
CCI_PERIOD = 14
WILLIAMS_R_PERIOD = 14

# Feature Engineering
PRICE_CHANGE_PERIODS = [1, 3, 5, 7]
VOLUME_CHANGE_PERIODS = [1, 3, 5, 7]
ROLLING_CORRELATION_PERIOD = 10

# Model Parameters
RANDOM_FOREST_ESTIMATORS = 200
RANDOM_FOREST_MAX_DEPTH = 10
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 5

# 2. بهبود Dashboard برای نمایش بهتر
def _display_enhanced_metrics(self, prediction, accuracy, probabilities, strategy_params, feature_importance=None):
    """نمایش معیارهای بهبود یافته"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("دقت مدل", f"{accuracy:.1%}")
    
    with col2:
        confidence = max(probabilities.values()) if probabilities else 0
        st.metric("اطمینان", f"{confidence:.1%}")
    
    with col3:
        # محاسبه Risk-Reward Ratio
        if strategy_params['mode'] == 'Fixed':
            rr_ratio = strategy_params['take_profit_pct'] / strategy_params['stop_loss_pct']
            st.metric("Risk/Reward", f"{rr_ratio:.2f}")
    
    # نمایش مهم‌ترین features
    if feature_importance is not None:
        st.subheader("مهم‌ترین عوامل تصمیم‌گیری")
        importance_df = pd.DataFrame({
            'Feature': feature_importance.index,
            'Importance': feature_importance.values
        }).sort_values('Importance', ascending=False).head(10)
        
        st.bar_chart(importance_df.set_index('Feature'))

# 3. اضافه کردن تحلیل Risk Management
def analyze_risk_metrics(self, full_df, strategy_params):
    """تحلیل معیارهای ریسک"""
    
    # محاسبه Sharpe Ratio
    returns = full_df['price'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    
    # محاسبه Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # محاسبه Win Rate
    predictions = full_df['AI_Prediction']
    actual = full_df['target']
    
    buy_signals = predictions == 1
    buy_success = (predictions == 1) & (actual == 1)
    win_rate = buy_success.sum() / buy_signals.sum() if buy_signals.sum() > 0 else 0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_signals': buy_signals.sum()
    }

# 4. بهبود پیش‌پردازش داده‌ها
def preprocess_data_advanced(self):
    """پیش‌پردازش پیشرفته‌تر"""
    
    # حذف outliers
    Q1 = self.df['price'].quantile(0.25)
    Q3 = self.df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    self.df = self.df[(self.df['price'] >= lower_bound) & (self.df['price'] <= upper_bound)]
    
    # نرمال‌سازی features
    from sklearn.preprocessing import StandardScaler
    
    numeric_columns = self.df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
    
    return scaler

# 5. اضافه کردن Ensemble Method
def create_ensemble_model(self):
    """ایجاد مدل ترکیبی"""
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(random_state=42)
    svm = SVC(probability=True, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr), ('svm', svm)],
        voting='soft'
    )
    
    return ensemble

# 6. بهبود نمایش نتایج
def display_advanced_results(self, results):
    """نمایش نتایج پیشرفته"""
    
    st.subheader("📊 تحلیل جامع عملکرد")
    
    # Risk Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    with col2:
        st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
    with col3:
        st.metric("Win Rate", f"{results['win_rate']:.2%}")
    with col4:
        st.metric("Total Signals", results['total_signals'])
    
    # Trading Performance
    if results['win_rate'] > 0.6:
        st.success("🎯 استراتژی عملکرد خوبی دارد")
    elif results['win_rate'] > 0.45:
        st.warning("⚠️ استراتژی متوسط است")
    else:
        st.error("❌ استراتژی نیاز به بهبود دارد")

# 7. تنظیمات بهینه پیشنهادی
OPTIMIZED_CONFIG = {
    'Dynamic': {
        'tp_multiplier': 2.5,
        'sl_multiplier': 1.2,
        'time_barrier_days': 7
    },
    'Fixed': {
        'take_profit_pct': 4.0,
        'stop_loss_pct': 2.0,
        'time_barrier_days': 8
    }
}