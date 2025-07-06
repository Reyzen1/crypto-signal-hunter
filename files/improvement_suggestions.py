# بهبودهای پیشنهادی برای technical_analyzer.py

# 1. اضافه کردن Feature Engineering بهتر
def add_enhanced_features(self):
    """اضافه کردن features پیشرفته‌تر"""
    
    # Price-based features
    self.df['price_change'] = self.df['price'].pct_change()
    self.df['price_change_ma'] = self.df['price_change'].rolling(5).mean()
    self.df['high_low_ratio'] = self.df['price'].rolling(14).max() / self.df['price'].rolling(14).min()
    
    # Volume-based features
    self.df['volume_change'] = self.df['volume'].pct_change()
    self.df['volume_price_trend'] = self.df['volume'] * self.df['price_change']
    
    # Technical indicator combinations
    self.df['rsi_ma'] = self.df['RSI_14'].rolling(3).mean()
    self.df['macd_signal_diff'] = self.df['MACD_12_26_9'] - self.df['MACDs_12_26_9']
    
    # Volatility features
    self.df['volatility_rank'] = self.df['volatility'].rolling(30).rank(pct=True)
    
    # Time-based features
    self.df['day_of_week'] = pd.to_datetime(self.df.index).dayofweek
    self.df['month'] = pd.to_datetime(self.df.index).month

# 2. بهبود مدل ML
def create_enhanced_model(self):
    """استفاده از مدل‌های پیشرفته‌تر"""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    
    # Grid search for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    base_model = GradientBoostingClassifier(random_state=42)
    model = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy')
    
    return model

# 3. بهبود استراتژی Triple Barrier
def _get_adaptive_triple_barrier_labels(self, strategy_params):
    """Triple barrier با تنظیمات تطبیقی"""
    prices = self.df['price']
    labels = pd.Series(index=prices.index, data=0.0)
    
    # استفاده از ATR برای تعیین حد ضرر و سود
    self.df['atr'] = ta.atr(high=prices, low=prices, close=prices, length=14)
    
    for i in range(len(prices) - strategy_params['time_barrier_days']):
        entry_price = prices.iloc[i]
        atr_value = self.df['atr'].iloc[i]
        
        if pd.isna(atr_value):
            continue
            
        # استفاده از ATR برای تعیین سطوح
        take_profit_level = entry_price + (atr_value * 2)
        stop_loss_level = entry_price - (atr_value * 1.5)
        
        # ادامه کد مشابه قبلی...

# 4. اضافه کردن Feature Selection
def select_best_features(self, X, y):
    """انتخاب بهترین features"""
    from sklearn.feature_selection import SelectKBest, f_classif
    
    selector = SelectKBest(score_func=f_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    
    # گرفتن نام features منتخب
    selected_features = X.columns[selector.get_support()]
    
    return X[selected_features], selected_features

# 5. اضافه کردن Walk-Forward Validation
def walk_forward_validation(self, X, y, n_splits=5):
    """اعتبارسنجی پیش‌رونده برای time series"""
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

# 6. بهبود تنظیمات استراتژی
def optimize_strategy_parameters(self, base_params):
    """بهینه‌سازی پارامترهای استراتژی"""
    best_score = 0
    best_params = base_params.copy()
    
    # تست پارامترهای مختلف
    tp_multipliers = [1.5, 2.0, 2.5, 3.0]
    sl_multipliers = [0.5, 1.0, 1.5, 2.0]
    time_barriers = [5, 7, 10, 14]
    
    for tp in tp_multipliers:
        for sl in sl_multipliers:
            for tb in time_barriers:
                test_params = base_params.copy()
                test_params.update({
                    'tp_multiplier': tp,
                    'sl_multiplier': sl,
                    'time_barrier_days': tb
                })
                
                X, y, _ = self.prepare_ml_data(test_params)
                if X is not None and len(y.unique()) > 1:
                    score, _ = self.walk_forward_validation(X, y)
                    if score > best_score:
                        best_score = score
                        best_params = test_params
    
    return best_params, best_score