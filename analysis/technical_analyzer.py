# analysis/technical_analyzer.py
"""
Optimized crypto technical analyzer with improved code structure and efficiency.
Version 6.3: Refactored with better constants management, flexible lagged features,
and reduced complexity.
"""

import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report  # تصحیح شده
import config

class CryptoAnalyzer:
    """Optimized crypto technical analyzer with ML capabilities."""
    
    # Class constants
    LABEL_MAP = {
        1.0: 'TAKE PROFIT 🟢',
        -1.0: 'STOP LOSS 🔴',
        0.0: 'TIME LIMIT ⚪️'
    }
    
    DEFAULT_LAGS = [1, 2, 3, 5]
    MODEL_PARAMS = {
        'n_estimators': 100,
        'random_state': 42,
        'class_weight': 'balanced'
    }
    
    SPLIT_PARAMS = {
        'test_size': 0.2,
        'random_state': 42,
        'shuffle': False
    }
    
    def __init__(self, market_data, lags=None):
        """Initialize analyzer with market data and optional lag configuration."""
        self.df = self._create_dataframe(market_data)
        self.lags = lags or self.DEFAULT_LAGS
        self.technical_params = self._get_technical_params()
        self.model_params = self._get_model_params()
    
    def _create_dataframe(self, market_data):
        """Process raw market data into clean DataFrame."""
        # Process price data
        df_price = pd.DataFrame(market_data['prices'], columns=['timestamp', 'price'])
        df_price['date'] = pd.to_datetime(df_price['timestamp'], unit='ms').dt.date
        df_price = df_price.drop_duplicates(subset='date').set_index('date')
        
        # Process volume data
        df_volume = pd.DataFrame(market_data['total_volumes'], columns=['timestamp', 'volume'])
        df_volume['date'] = pd.to_datetime(df_volume['timestamp'], unit='ms').dt.date
        df_volume = df_volume.drop_duplicates(subset='date').set_index('date')
        
        # Merge and clean
        return pd.merge(
            df_price.drop('timestamp', axis=1),
            df_volume.drop('timestamp', axis=1),
            on='date', how='inner'
        )
    
    def _get_technical_params(self):
        """Get technical analysis parameters from config."""
        return {
            'SMA_EXTRA_SHORT_PERIOD': config.SMA_EXTRA_SHORT_PERIOD,
            'SMA_SHORT_PERIOD': config.SMA_SHORT_PERIOD,
            'SMA_LONG_PERIOD': config.SMA_LONG_PERIOD,
            'RSI_PERIOD': config.RSI_PERIOD,
            'MACD_FAST': config.MACD_FAST,
            'MACD_SLOW': config.MACD_SLOW,
            'MACD_SIGNAL': config.MACD_SIGNAL,
            'BBANDS_LENGTH': config.BBANDS_LENGTH,
            'BBANDS_STD': config.BBANDS_STD,
            'VOLUME_SMA_PERIOD': config.VOLUME_SMA_PERIOD,
            'VOLATILITY_WINDOW': config.VOLATILITY_WINDOW
        }
    
    def _get_model_params(self):
        """Get model parameters with current configuration."""
        return {
            'algorithm': 'RandomForestClassifier',
            'lagged_features': self.lags,
            **self.MODEL_PARAMS,
            **self.SPLIT_PARAMS
        }
    
    def add_all_indicators(self):
        """Add all technical indicators to DataFrame."""
        if self.df.empty:
            return
        
        close_price = self.df['price']
        volume = self.df['volume']
        
        # Add all indicators in one go
        self._add_trend_indicators(close_price)
        self._add_volatility_indicators(close_price)
        self._add_volume_indicators(close_price, volume)
        
        # Clean up and handle duplicates
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        self.df.dropna(inplace=True)
    
    def _add_trend_indicators(self, close_price):
        """Add trend and momentum indicators."""
        # SMA indicators
        for period in [config.SMA_EXTRA_SHORT_PERIOD, config.SMA_SHORT_PERIOD, config.SMA_LONG_PERIOD]:
            self.df.ta.sma(close=close_price, length=period, append=True)
        
        # RSI and MACD
        self.df.ta.rsi(close=close_price, length=config.RSI_PERIOD, append=True)
        self.df.ta.macd(close=close_price, fast=config.MACD_FAST, 
                        slow=config.MACD_SLOW, signal=config.MACD_SIGNAL, append=True)
    
    def _add_volatility_indicators(self, close_price):
        """Add volatility indicators."""
        # Bollinger Bands
        self.df.ta.bbands(close=close_price, length=config.BBANDS_LENGTH, 
                         std=config.BBANDS_STD, append=True)
        
        # Add volatility column
        self.df['volatility'] = (self.df['price'].pct_change()
                                .rolling(window=config.VOLATILITY_WINDOW).std())
    
    def _add_volume_indicators(self, close_price, volume):
        """Add volume indicators."""
        # Volume SMA
        vol_sma_name = f'VOL_SMA_{config.VOLUME_SMA_PERIOD}'
        self.df.ta.sma(close=volume, length=config.VOLUME_SMA_PERIOD, 
                      append=True, col_names=(vol_sma_name,))
        
        # Volume ratio (with epsilon for stability)
        self.df['VOL_RATIO'] = volume / (self.df[vol_sma_name] + 1e-10)
        
        # On-Balance Volume
        self.df.ta.obv(close=close_price, volume=volume, append=True)
    
    def _get_triple_barrier_labels(self, strategy_params):
        """Create labels using triple-barrier method."""
        prices = self.df['price']
        labels = pd.Series(index=prices.index, data=0.0)
        
        mode = strategy_params['mode']
        time_barrier_days = strategy_params['time_barrier_days']
        
        for i in range(len(prices) - time_barrier_days):
            levels = self._calculate_barrier_levels(i, strategy_params)
            if levels is None:
                continue
            
            take_profit_level, stop_loss_level = levels
            future_prices = prices.iloc[i+1 : i+1+time_barrier_days]
            
            for price in future_prices:
                if price >= take_profit_level:
                    labels.iloc[i] = 1.0
                    break
                elif price <= stop_loss_level:
                    labels.iloc[i] = -1.0
                    break
        
        return labels
    
    def _calculate_barrier_levels(self, idx, strategy_params):
        """Calculate take profit and stop loss levels."""
        entry_price = self.df['price'].iloc[idx]
        mode = strategy_params['mode']
        
        if mode == 'Dynamic':
            volatility = self.df['volatility'].iloc[idx]
            if pd.isna(volatility) or volatility == 0:
                return None
            
            tp_level = entry_price * (1 + volatility * strategy_params['tp_multiplier'])
            sl_level = entry_price * (1 - volatility * strategy_params['sl_multiplier'])
        
        elif mode == 'Fixed':
            tp_level = entry_price * (1 + strategy_params['take_profit_pct'] / 100)
            sl_level = entry_price * (1 - strategy_params['stop_loss_pct'] / 100)
        
        else:
            raise ValueError(f"Unknown strategy mode: {mode}")
        
        return tp_level, sl_level
    
    def prepare_ml_data(self, strategy_params):
        """Prepare ML data with labels and lagged features."""
        # Create labels
        self.df['target'] = self._get_triple_barrier_labels(strategy_params)
        
        # Create lagged features efficiently
        features_to_exclude = ['price', 'volume', 'target']
        base_features = [col for col in self.df.columns if col not in features_to_exclude]
        
        # Use list comprehension for efficient lagged feature creation
        lagged_features_list = [self.df[base_features]]
        lagged_features_list.extend([
            self.df[base_features].shift(lag).add_suffix(f'_lag_{lag}')
            for lag in self.lags
        ])
        
        # Combine all features
        X_with_lags = pd.concat(lagged_features_list, axis=1)
        
        # Create final dataset
        combined = pd.concat([X_with_lags, self.df['target']], axis=1)
        combined.dropna(inplace=True)
        
        if combined.empty:
            return None, None, None
        
        X = combined.drop('target', axis=1)
        y = combined['target']
        full_df = self.df.loc[X.index].copy()
        
        return X, y, full_df
    
    def train_and_predict(self, strategy_params):
        """Train model and make predictions."""
        X, y, full_df = self.prepare_ml_data(strategy_params)
        
        if X is None or X.empty or len(y.unique()) < 2:
            return (None,) * 6
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, **self.SPLIT_PARAMS)
        
        if len(X_train) == 0 or len(X_test) == 0:
            return (None,) * 6
        
        # Train model
        model = RandomForestClassifier(**self.MODEL_PARAMS)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_on_test = model.predict(X_test)
        all_predictions = model.predict(X)
        
        # Calculate metrics
        accuracy = model.score(X_test, y_test)
        report = classification_report(y_test, y_pred_on_test, output_dict=True,
                                     zero_division=0, labels=[1.0, -1.0, 0.0])
        
        # Create result datasets
        test_results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred_on_test
        }, index=X_test.index)
        
        enhanced_full_df = self._create_enhanced_display_df(
            full_df, X, y, X_train, X_test, y_train, y_test, 
            all_predictions, y_pred_on_test
        )
        
        # Make final prediction
        latest_features = X.iloc[[-1]]
        prediction = model.predict(latest_features)[0]
        probabilities = model.predict_proba(latest_features)
        prob_dict = {model.classes_[i]: probabilities[0][i] 
                    for i in range(len(model.classes_))}
        
        return prediction, accuracy, prob_dict, report, test_results_df, enhanced_full_df
    
    def _create_enhanced_display_df(self, full_df, X, y, X_train, X_test, 
                                  y_train, y_test, all_predictions, y_pred_on_test):
        """Create enhanced DataFrame for display with all information."""
        enhanced_df = pd.concat([full_df, X], axis=1)
        
        # Add prediction columns
        enhanced_df['AI_Prediction'] = all_predictions
        enhanced_df['AI_Prediction_Label'] = enhanced_df['AI_Prediction'].map(self.LABEL_MAP)
        
        # Add data type information
        enhanced_df['Data_Type'] = 'Unknown'
        enhanced_df.loc[X_train.index, 'Data_Type'] = 'Training 📚'
        enhanced_df.loc[X_test.index, 'Data_Type'] = 'Test 🧪'
        
        # Add prediction correctness for test data
        enhanced_df['Prediction_Correctness'] = ''
        test_correct = y_test == y_pred_on_test
        enhanced_df.loc[X_test.index, 'Prediction_Correctness'] = test_correct.map({
            True: 'Correct ✅',
            False: 'Incorrect ❌'
        })
        
        # Add actual target information
        enhanced_df['Target_Actual'] = y
        enhanced_df['Target_Actual_Label'] = enhanced_df['Target_Actual'].map(self.LABEL_MAP)
        
        return enhanced_df