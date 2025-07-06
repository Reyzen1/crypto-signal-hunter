# analysis/technical_analyzer.py
"""
This module provides the core analysis engine.
Version 6.1: Added prediction column to raw data display.
"""

import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import config

class CryptoAnalyzer:
    """
    The core engine for performing technical analysis and training the ML model.
    """
    def __init__(self, market_data):
        """
        Initializes the analyzer with market data from the API.
        """
        self.df = self._create_dataframe(market_data)

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

    def add_all_indicators(self):
        """
        Calculates all technical indicators using the pandas_ta library.
        """
        if self.df.empty:
            return

        close_price = self.df['price']
        volume = self.df['volume']

        # Trend and Momentum Indicators
        self.df.ta.sma(close=close_price, length=config.SMA_EXTRA_SHORT_PERIOD, append=True)
        self.df.ta.sma(close=close_price, length=config.SMA_SHORT_PERIOD, append=True)
        self.df.ta.sma(close=close_price, length=config.SMA_LONG_PERIOD, append=True)
        self.df.ta.rsi(close=close_price, length=config.RSI_PERIOD, append=True)
        self.df.ta.macd(close=close_price, fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL, append=True)
        
        # Volatility Indicator
        self.df.ta.bbands(close=close_price, length=config.BBANDS_LENGTH, std=config.BBANDS_STD, append=True)
        
        # Add volatility column for dynamic strategy
        self.df['volatility'] = self.df['price'].pct_change().rolling(window=config.VOLATILITY_WINDOW).std()
        
        # Volume Indicators
        vol_sma_name = f'VOL_SMA_{config.VOLUME_SMA_PERIOD}'
        self.df.ta.sma(close=volume, length=config.VOLUME_SMA_PERIOD, append=True, col_names=(vol_sma_name,))
        
        epsilon = 1e-10
        self.df['VOL_RATIO'] = self.df['volume'] / (self.df[vol_sma_name] + epsilon)
        self.df.ta.obv(close=close_price, volume=volume, append=True)
        
        # Clean up any potential duplicate columns that might be generated
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        self.df.dropna(inplace=True)

    def _get_triple_barrier_labels(self, strategy_params):
        """
        Labels the data based on the triple-barrier method with dual strategy support.
        - 1: Take Profit hit
        - -1: Stop Loss hit
        - 0: Time Barrier hit
        
        Args:
            strategy_params (dict): Dictionary containing strategy parameters
                - mode: 'Dynamic' or 'Fixed'
                - For Dynamic mode: tp_multiplier, sl_multiplier, time_barrier_days
                - For Fixed mode: take_profit_pct, stop_loss_pct, time_barrier_days
        """
        prices = self.df['price']
        labels = pd.Series(index=prices.index, data=0.0)
        mode = strategy_params['mode']
        time_barrier_days = strategy_params['time_barrier_days']

        for i in range(len(prices) - time_barrier_days):
            entry_price = prices.iloc[i]
            
            if mode == 'Dynamic':
                # Dynamic mode: calculate thresholds based on volatility
                volatility = self.df['volatility'].iloc[i]
                if pd.isna(volatility) or volatility == 0:
                    continue
                
                tp_multiplier = strategy_params['tp_multiplier']
                sl_multiplier = strategy_params['sl_multiplier']
                
                take_profit_level = entry_price * (1 + volatility * tp_multiplier)
                stop_loss_level = entry_price * (1 - volatility * sl_multiplier)
                
            elif mode == 'Fixed':
                # Fixed mode: use fixed percentages
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

    def prepare_ml_data(self, strategy_params):
        """
        Prepares the data for machine learning by adding labels and lagged features.
        
        Args:
            strategy_params (dict): Dictionary containing strategy parameters
        """
        # Step 1: Create labels using the triple-barrier method
        self.df['target'] = self._get_triple_barrier_labels(strategy_params)
        
        # Step 2: Identify base features to create lags from
        features_to_exclude = ['price', 'volume', 'target']
        base_features = [col for col in self.df.columns if col not in features_to_exclude]
        
        original_features_df = self.df[base_features].copy()
        lagged_features_list = [original_features_df]
        lags_to_create = [1, 2, 3, 5] # Define which past days to use as features

        # Step 3: Create lagged features in a loop
        for lag in lags_to_create:
            shifted_df = original_features_df.shift(lag)
            shifted_df.columns = [f'{col}_lag_{lag}' for col in original_features_df.columns]
            lagged_features_list.append(shifted_df)

        # Combine original and lagged features into one DataFrame
        X_with_lags = pd.concat(lagged_features_list, axis=1)

        # Step 4 & 5: Manage NaN values and finalize data
        y = self.df['target']
        combined = pd.concat([X_with_lags, y], axis=1)
        combined.dropna(inplace=True) # Drop rows with any NaN values (from shifting)
        
        if combined.empty: 
            return None, None, None

        # Separate the final features (X) and target (y)
        X = combined.drop('target', axis=1)
        y = combined['target']

        # The full_df should align with the cleaned data's index for plotting
        full_df = self.df.loc[X.index].copy()

        return X, y, full_df

    def train_and_predict(self, strategy_params):
        """
        Trains the RandomForest model and makes a prediction for the latest data point.
        
        Args:
            strategy_params (dict): Dictionary containing strategy parameters
        """
        # Prepare all features, including lagged ones
        X, y, full_df = self.prepare_ml_data(strategy_params)

        if X is None or X.empty or len(y.unique()) < 2:
            return None, None, None, None, None, None

        # Split data without shuffling to respect the time-series nature
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        if len(X_train) == 0 or len(X_test) == 0:
            return None, None, None, None, None, None

        # The last row of features is used for the final real-time prediction
        latest_features = X.iloc[[-1]]

        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Evaluate the model on the test set
        y_pred_on_test = model.predict(X_test)
        report = classification_report(y_test, y_pred_on_test, output_dict=True, zero_division=0, labels=[1.0, -1.0, 0.0])
        accuracy = model.score(X_test, y_test)

        # Create a DataFrame for error analysis visualization
        test_results_df = pd.DataFrame(index=X_test.index)
        test_results_df['actual'] = y_test
        test_results_df['predicted'] = y_pred_on_test
        
        # Make predictions for all data points (including the latest one)
        all_predictions = model.predict(X)
        
        # Add predictions to the full DataFrame for display
        full_df['AI_Prediction'] = all_predictions
        full_df['AI_Prediction_Label'] = full_df['AI_Prediction'].map({
            1.0: 'TAKE PROFIT 🟢',
            -1.0: 'STOP LOSS 🔴',
            0.0: 'TIME LIMIT ⚪️'
        })
        
        # Make the final prediction on the most recent data
        prediction = model.predict(latest_features)
        probabilities = model.predict_proba(latest_features)
        prob_dict = {model.classes_[i]: probabilities[0][i] for i in range(len(model.classes_))}

        return prediction[0], accuracy, prob_dict, report, test_results_df, full_df