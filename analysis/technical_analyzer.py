# analysis/technical_analyzer.py
"""
This module provides the core analysis engine for the cryptocurrency data.
Version 2.8: Added detailed classification report for model evaluation.
"""

import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Import advanced evaluation metrics
from sklearn.metrics import classification_report

class CryptoAnalyzer:
    """
    Performs all technical analysis, signal generation, and ML model operations.
    """
    def __init__(self, market_data):
        """
        Initializes the analyzer with raw market data from the API.
        """
        if not market_data or 'prices' not in market_data or 'total_volumes' not in market_data:
            raise ValueError("Invalid or incomplete market data provided.")
        self.market_data = market_data
        self.df = self._create_dataframe()

    def _create_dataframe(self):
        """
        Converts raw market data into a clean, merged pandas DataFrame.
        """
        df_price = pd.DataFrame(self.market_data['prices'], columns=['timestamp', 'price'])
        df_price['date'] = pd.to_datetime(df_price['timestamp'], unit='ms')
        df_price = df_price.set_index('date').drop('timestamp', axis=1)

        df_volume = pd.DataFrame(self.market_data['total_volumes'], columns=['timestamp', 'volume'])
        df_volume['date'] = pd.to_datetime(df_volume['timestamp'], unit='ms')
        df_volume = df_volume.set_index('date').drop('timestamp', axis=1)

        return pd.merge(df_price, df_volume, on='date', how='inner')

    def add_all_indicators(self):
        """
        Calculates and appends all technical indicators. This version also
        handles potential duplicate columns to ensure data integrity.
        """
        if self.df.empty: return self.df

        self.df['SMA_10'] = self.df['price'].rolling(window=10).mean()
        self.df['SMA_30'] = self.df['price'].rolling(window=30).mean()
        self.df.ta.macd(close=self.df['price'], append=True)
        self.df.ta.rsi(close=self.df['price'], append=True)
        
        bbands = self.df.ta.bbands(length=20, std=2, append=False)
        if bbands is not None and not bbands.empty:
            self.df = pd.concat([self.df, bbands], axis=1)

        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        self.df.dropna(inplace=True)
        return self.df

    def generate_signals(self):
        """
        Interprets indicator values to generate actionable trading signals.
        (Currently not used in the main display but kept for future use).
        """
        signals = []
        if len(self.df) < 2: return signals
        latest = self.df.iloc[-1]
        previous = self.df.iloc[-2]
        rsi_value = latest['RSI_14']
        if rsi_value > 70: signals.append(('warning', f"RSI is at {rsi_value:.2f}, indicating **Overbought**."))
        elif rsi_value < 30: signals.append(('success', f"RSI is at {rsi_value:.2f}, indicating **Oversold**."))
        else: signals.append(('info', f"RSI is at {rsi_value:.2f}, in the **Neutral Zone**."))
        macd_line, signal_line = 'MACD_12_26_9', 'MACDs_12_26_9'
        is_bullish_crossover = previous[macd_line] < previous[signal_line] and latest[macd_line] > latest[signal_line]
        is_bearish_crossover = previous[macd_line] > previous[signal_line] and latest[macd_line] < latest[signal_line]
        if is_bullish_crossover: signals.append(('success', "**🔥 Bullish MACD Crossover** detected!"))
        elif is_bearish_crossover: signals.append(('error', "**🚨 Bearish MACD Crossover** detected!"))
        else:
            if latest[macd_line] > latest[signal_line]: signals.append(('info', f"**MACD Bullish Trend**"))
            else: signals.append(('warning', f"**MACD Bearish Trend**"))
        return signals

    def prepare_ml_data(self, threshold=0.01):
        """
        Prepares data for a 3-class classification problem (Buy/Sell/Hold).
        """
        if self.df.empty:
            return None, None

        ml_df = self.df.copy()
        ml_df['price_change'] = ml_df['price'].shift(-1) / ml_df['price'] - 1

        ml_df['target'] = 0 # Default to Hold
        ml_df.loc[ml_df['price_change'] > threshold, 'target'] = 1  # Buy
        ml_df.loc[ml_df['price_change'] < -threshold, 'target'] = -1 # Sell

        base_features = [
            'SMA_10', 'SMA_30', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9',
            'MACDs_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'
        ]
        
        for feature in base_features:
            if feature in ml_df.columns:
                for i in range(1, 4):
                    ml_df[f'{feature}_lag_{i}'] = ml_df[feature].shift(i)
        
        ml_df.dropna(inplace=True)
        
        features_to_exclude = ['price', 'volume', 'price_change', 'target']
        all_features = [col for col in ml_df.columns if col not in features_to_exclude]
        
        X = ml_df[all_features]
        y = ml_df['target']

        return X, y

    def train_and_predict(self):
        """
        Trains the model and returns a detailed classification report along with the prediction.
        """
        X, y = self.prepare_ml_data()
        if X is None or X.empty or len(X) < 20:
            return None, None, None, None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        if len(X_train) == 0 or len(X_test) == 0:
            return None, None, None, None

        latest_features = X.iloc[[-1]]

        # Use class_weight='balanced' to give more importance to the minority classes (Buy/Sell)
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        y_pred_on_test = model.predict(X_test)
        
        # Generate the detailed classification report as a dictionary
        report = classification_report(y_test, y_pred_on_test, output_dict=True, zero_division=0)
        
        prediction = model.predict(latest_features)
        probabilities = model.predict_proba(latest_features)
        accuracy = model.score(X_test, y_test)
        
        prob_dict = {model.classes_[i]: probabilities[0][i] for i in range(len(model.classes_))}

        return prediction[0], accuracy, prob_dict, report