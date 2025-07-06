# analysis/technical_analyzer.py
"""
This module provides the core analysis engine.
Version 7.0: Enhanced with advanced features, better ML models, and improved strategies.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import config

class CryptoAnalyzer:
    """
    The core engine for performing technical analysis and training the ML model.
    Enhanced with advanced features and better ML capabilities.
    """
    def __init__(self, market_data):
        """
        Initializes the analyzer with market data from the API.
        """
        self.df = self._create_dataframe(market_data)
        self.scaler = None

    def _create_dataframe(self, market_data):
        """
        Processes the raw market data JSON into a clean pandas DataFrame.
        """
        df_price = pd.DataFrame(market_data['prices'], columns=['timestamp', 'price'])
        df_price['date'] = pd.to_datetime(df_price['timestamp'], unit='ms').dt.date
        df_price = df_price.drop_duplicates(subset='date').set_index('date')

        df_volume = pd.DataFrame(market_data['total_volumes'], columns=['timestamp', 'volume'])
        df_volume['date'] = pd.to_datetime(df_volume['timestamp'], unit='ms').dt.date
        df_volume = df_volume.drop_duplicates(subset='date').set_index('date')
        
        return pd.merge(df_price.drop('timestamp', axis=1), df_volume.drop('timestamp', axis=1), on='date', how='inner')

    def preprocess_data_advanced(self):
        """پیش‌پردازش پیشرفته‌تر"""
        # حذف outliers
        Q1 = self.df['price'].quantile(0.25)
        Q3 = self.df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        original_length = len(self.df)
        self.df = self.df[(self.df['price'] >= lower_bound) & (self.df['price'] <= upper_bound)]
        
        if len(self.df) < original_length * 0.8:  # اگر بیش از 20% داده حذف شد
            # بازگشت به داده‌های اصلی
            self.df = self._create_dataframe(self.df)
        
        return self

    def add_enhanced_features(self):
        """اضافه کردن features پیشرفته‌تر"""
        if self.df.empty:
            return
            
        close_price = self.df['price']
        volume = self.df['volume']
        
        # Price-based features
        self.df['price_change'] = self.df['price'].pct_change()
        self.df['price_change_ma'] = self.df['price_change'].rolling(5).mean()
        self.df['high_low_ratio'] = self.df['price'].rolling(14).max() / self.df['price'].rolling(14).min()
        
        # Volume-based features
        self.df['volume_change'] = self.df['volume'].pct_change()
        self.df['volume_price_trend'] = self.df['volume'] * self.df['price_change']
        
        # Time-based features
        self.df['day_of_week'] = pd.to_datetime(self.df.index).dayofweek
        self.df['month'] = pd.to_datetime(self.df.index).month
        
        # Multiple period price changes
        for period in [1, 3, 5, 7]:
            self.df[f'price_change_{period}d'] = self.df['price'].pct_change(periods=period)
            self.df[f'volume_change_{period}d'] = self.df['volume'].pct_change(periods=period)

    def add_all_indicators(self):
        """
        Calculates all technical indicators using the pandas_ta library.
        Enhanced with more indicators.
        """
        if self.df.empty:
            return

        close_price = self.df['price']
        volume = self.df['volume']

        # Basic trend and momentum indicators
        self.df.ta.sma(close=close_price, length=config.SMA_EXTRA_SHORT_PERIOD, append=True)
        self.df.ta.sma(close=close_price, length=config.SMA_SHORT_PERIOD, append=True)
        self.df.ta.sma(close=close_price, length=config.SMA_LONG_PERIOD, append=True)
        self.df.ta.rsi(close=close_price, length=config.RSI_PERIOD, append=True)
        self.df.ta.macd(close=close_price, fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL, append=True)
        
        # Volatility indicators
        self.df.ta.bbands(close=close_price, length=config.BBANDS_LENGTH, std=config.BBANDS_STD, append=True)
        self.df['volatility'] = self.df['price'].pct_change().rolling(window=config.VOLATILITY_WINDOW).std()
        
        # Advanced technical indicators
        self.df.ta.stoch(high=close_price, low=close_price, close=close_price, k=14, d=3, append=True)
        self.df.ta.atr(high=close_price, low=close_price, close=close_price, length=14, append=True)
        self.df.ta.cci(high=close_price, low=close_price, close=close_price, length=14, append=True)
        self.df.ta.willr(high=close_price, low=close_price, close=close_price, length=14, append=True)
        
        # Volume indicators
        vol_sma_name = f'VOL_SMA_{config.VOLUME_SMA_PERIOD}'
        self.df.ta.sma(close=volume, length=config.VOLUME_SMA_PERIOD, append=True, col_names=(vol_sma_name,))
        
        epsilon = 1e-10
        self.df['VOL_RATIO'] = self.df['volume'] / (self.df[vol_sma_name] + epsilon)
        self.df.ta.obv(close=close_price, volume=volume, append=True)
        
        # Enhanced features
        self.add_enhanced_features()
        
        # Technical indicator combinations
        if f'RSI_{config.RSI_PERIOD}' in self.df.columns:
            self.df['rsi_ma'] = self.df[f'RSI_{config.RSI_PERIOD}'].rolling(3).mean()
        
        if f'MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}' in self.df.columns:
            macd_col = f'MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}'
            signal_col = f'MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}'
            if signal_col in self.df.columns:
                self.df['macd_signal_diff'] = self.df[macd_col] - self.df[signal_col]
        
        # Volatility features
        self.df['volatility_rank'] = self.df['volatility'].rolling(30).rank(pct=True)
        
        # Clean up any potential duplicate columns
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        self.df.dropna(inplace=True)

    def _get_adaptive_triple_barrier_labels(self, strategy_params):
        """
        Triple barrier با تنظیمات تطبیقی و استفاده از ATR
        """
        prices = self.df['price']
        labels = pd.Series(index=prices.index, data=0.0)
        mode = strategy_params['mode']
        time_barrier_days = strategy_params['time_barrier_days']
        
        # محاسبه ATR اگر موجود نباشد
        if 'ATRr_14' not in self.df.columns:
            self.df['ATRr_14'] = prices.rolling(14).apply(lambda x: x.max() - x.min())

        for i in range(len(prices) - time_barrier_days):
            entry_price = prices.iloc[i]
            
            if mode == 'Dynamic':
                # Dynamic mode: استفاده از ATR
                atr_value = self.df['ATRr_14'].iloc[i]
                if pd.isna(atr_value) or atr_value == 0:
                    continue
                
                tp_multiplier = strategy_params['tp_multiplier']
                sl_multiplier = strategy_params['sl_multiplier']
                
                take_profit_level = entry_price + (atr_value * tp_multiplier)
                stop_loss_level = entry_price - (atr_value * sl_multiplier)
                
            elif mode == 'Fixed':
                # Fixed mode: استفاده از درصد ثابت
                take_profit_pct = strategy_params['take_profit_pct']
                stop_loss_pct = strategy_params['stop_loss_pct']
                
                take_profit_level = entry_price * (1 + take_profit_pct / 100)
                stop_loss_level = entry_price * (1 - stop_loss_pct / 100)
            
            else:
                raise ValueError(f"Unknown strategy mode: {mode}")

            future_prices = prices.iloc[i+1 : i+1+time_barrier_days]

            for price in future_prices:
                if price >= take_profit_level:
                    labels.iloc[i] = 1.0
                    break 
                if price <= stop_loss_level:
                    labels.iloc[i] = -1.0
                    break
        
        return labels

    def _get_triple_barrier_labels(self, strategy_params):
        """
        Labels the data based on the triple-barrier method with dual strategy support.
        """
        return self._get_adaptive_triple_barrier_labels(strategy_params)

    def select_best_features(self, X, y, k=50):
        """انتخاب بهترین features"""
        if len(X.columns) <= k:
            return X, X.columns
            
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()]
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

    def walk_forward_validation(self, X, y, n_splits=5):
        """اعتبارسنجی پیش‌رونده برای time series"""
        if len(X) < n_splits * 2:
            return None, None
            
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if len(y_train.unique()) < 2:
                continue
                
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        if not scores:
            return None, None
            
        return np.mean(scores), np.std(scores)

    def create_ensemble_model(self):
        """ایجاد مدل ترکیبی"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        
        # استفاده از احتمالات برای SVM
        svm = SVC(probability=True, random_state=42, class_weight='balanced')
        
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('lr', lr), ('svm', svm)],
            voting='soft'
        )
        
        return ensemble

    def create_enhanced_model(self):
        """استفاده از مدل‌های پیشرفته‌تر با Grid Search"""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
        model = GridSearchCV(
            base_model, 
            param_grid, 
            cv=3, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        return model

    def prepare_ml_data(self, strategy_params):
        """
        Prepares the data for machine learning by adding labels and lagged features.
        Enhanced with better feature engineering.
        """
        # Step 1: Create labels using the triple-barrier method
        self.df['target'] = self._get_triple_barrier_labels(strategy_params)
        
        # Step 2: Identify base features to create lags from
        features_to_exclude = ['price', 'volume', 'target', 'timestamp']
        base_features = [col for col in self.df.columns if col not in features_to_exclude]
        
        original_features_df = self.df[base_features].copy()
        lagged_features_list = [original_features_df]
        lags_to_create = [1, 2, 3, 5, 7]

        # Step 3: Create lagged features
        for lag in lags_to_create:
            shifted_df = original_features_df.shift(lag)
            shifted_df.columns = [f'{col}_lag_{lag}' for col in original_features_df.columns]
            lagged_features_list.append(shifted_df)

        # Combine original and lagged features
        X_with_lags = pd.concat(lagged_features_list, axis=1)

        # Step 4: Manage NaN values and finalize data
        y = self.df['target']
        combined = pd.concat([X_with_lags, y], axis=1)
        combined.dropna(inplace=True)
        
        if combined.empty: 
            return None, None, None

        # Separate features and target
        X = combined.drop('target', axis=1)
        y = combined['target']

        # Feature selection
        try:
            X, selected_features = self.select_best_features(X, y)
        except:
            selected_features = X.columns

        # Full DataFrame for plotting
        full_df = self.df.loc[X.index].copy()

        return X, y, full_df

    def optimize_strategy_parameters(self, base_params):
        """بهینه‌سازی پارامترهای استراتژی"""
        best_score = 0
        best_params = base_params.copy()
        
        if base_params['mode'] == 'Dynamic':
            tp_multipliers = [1.5, 2.0, 2.5, 3.0]
            sl_multipliers = [0.5, 1.0, 1.5, 2.0]
            
            for tp in tp_multipliers:
                for sl in sl_multipliers:
                    test_params = base_params.copy()
                    test_params.update({
                        'tp_multiplier': tp,
                        'sl_multiplier': sl
                    })
                    
                    X, y, _ = self.prepare_ml_data(test_params)
                    if X is not None and len(y.unique()) > 1:
                        score, _ = self.walk_forward_validation(X, y)
                        if score and score > best_score:
                            best_score = score
                            best_params = test_params
        
        return best_params, best_score

    def analyze_risk_metrics(self, full_df):
        """تحلیل معیارهای ریسک"""
        if 'price' not in full_df.columns:
            return {}
            
        # محاسبه بازده
        returns = full_df['price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Sharpe Ratio
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win Rate
        if 'AI_Prediction' in full_df.columns and 'target' in full_df.columns:
            predictions = full_df['AI_Prediction']
            actual = full_df['target']
            
            buy_signals = predictions == 1
            buy_success = (predictions == 1) & (actual == 1)
            win_rate = buy_success.sum() / buy_signals.sum() if buy_signals.sum() > 0 else 0
            total_signals = buy_signals.sum()
        else:
            win_rate = 0
            total_signals = 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_signals': total_signals,
            'volatility': returns.std(),
            'avg_return': returns.mean()
        }

    def train_and_predict(self, strategy_params):
        """
        Trains the enhanced model and makes predictions.
        Enhanced with better models and feature selection.
        """
        # Prepare features
        X, y, full_df = self.prepare_ml_data(strategy_params)

        if X is None or X.empty or len(y.unique()) < 2:
            return None, None, None, None, None, None, None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        if len(X_train) == 0 or len(X_test) == 0:
            return None, None, None, None, None, None, None

        # Latest features for prediction
        latest_features = X.iloc[[-1]]

        # Create and train enhanced model
        try:
            model = self.create_enhanced_model()
            model.fit(X_train, y_train)
            
            # Get feature importance
            if hasattr(model, 'best_estimator_'):
                feature_importance = pd.Series(
                    model.best_estimator_.feature_importances_, 
                    index=X.columns
                ).sort_values(ascending=False)
            else:
                feature_importance = None
                
        except Exception as e:
            # Fallback to simple model
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            feature_importance = pd.Series(
                model.feature_importances_, 
                index=X.columns
            ).sort_values(ascending=False)
        
        # Evaluate model
        y_pred_on_test = model.predict(X_test)
        report = classification_report(
            y_test, y_pred_on_test, output_dict=True, 
            zero_division=0, labels=[1.0, -1.0, 0.0]
        )
        accuracy = model.score(X_test, y_test)

        # Test results for visualization
        test_results_df = pd.DataFrame(index=X_test.index)
        test_results_df['actual'] = y_test
        test_results_df['predicted'] = y_pred_on_test
        
        # Predict on all data
        all_predictions = model.predict(X)
        
        # Add predictions to full DataFrame
        full_df['AI_Prediction'] = all_predictions
        full_df['AI_Prediction_Label'] = full_df['AI_Prediction'].map({
            1.0: 'TAKE PROFIT 🟢',
            -1.0: 'STOP LOSS 🔴',
            0.0: 'TIME LIMIT ⚪️'
        })
        
        # Final prediction
        prediction = model.predict(latest_features)
        probabilities = model.predict_proba(latest_features)
        prob_dict = {model.classes_[i]: probabilities[0][i] for i in range(len(model.classes_))}

        return prediction[0], accuracy, prob_dict, report, test_results_df, full_df, feature_importance