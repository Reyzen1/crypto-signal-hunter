# analysis/technical_analyzer.py
"""
This module provides the core analysis engine for the cryptocurrency data.
Version 3.1: Methods now accept strategy parameters from the UI.
"""

import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import config

class CryptoAnalyzer:
    def __init__(self, market_data):
        if not market_data or 'prices' not in market_data or 'total_volumes' not in market_data:
            raise ValueError("Invalid or incomplete market data provided.")
        self.market_data = market_data
        self.df = self._create_dataframe()

    def _create_dataframe(self):
        # ... (unchanged)
        return pd.merge(pd.DataFrame(self.market_data['prices'], columns=['timestamp', 'price']).assign(date=lambda x: pd.to_datetime(x['timestamp'], unit='ms')).set_index('date').drop('timestamp', axis=1),
                        pd.DataFrame(self.market_data['total_volumes'], columns=['timestamp', 'volume']).assign(date=lambda x: pd.to_datetime(x['timestamp'], unit='ms')).set_index('date').drop('timestamp', axis=1),
                        on='date', how='inner')

    def add_all_indicators(self):
        # ... (unchanged, but now uses config)
        if self.df.empty: return self.df
        self.df['SMA_10'] = self.df['price'].rolling(window=config.SMA_SHORT_PERIOD).mean()
        self.df['SMA_30'] = self.df['price'].rolling(window=config.SMA_LONG_PERIOD).mean()
        self.df.ta.macd(close=self.df['price'], fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL, append=True)
        self.df.ta.rsi(length=config.RSI_PERIOD, append=True)
        bbands = self.df.ta.bbands(length=config.BBANDS_LENGTH, std=config.BBANDS_STD, append=False)
        if bbands is not None and not bbands.empty: self.df = pd.concat([self.df, bbands], axis=1)
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        self.df.dropna(inplace=True)
        return self.df

    def _get_triple_barrier_labels(self, take_profit_pct, stop_loss_pct, time_barrier_days):
        # ... (unchanged)
        prices = self.df['price']
        labels = pd.Series(index=prices.index, data=0.0)
        for i in range(len(prices) - time_barrier_days):
            entry_price = prices.iloc[i]
            take_profit_level = entry_price * (1 + take_profit_pct / 100)
            stop_loss_level = entry_price * (1 - stop_loss_pct / 100)
            future_prices = prices.iloc[i+1 : i+1+time_barrier_days]
            for price in future_prices:
                if price >= take_profit_level:
                    labels.iloc[i] = 1; break
                if price <= stop_loss_level:
                    labels.iloc[i] = -1; break
        return labels

    # --- MODIFIED: Now accepts strategy parameters ---
    def prepare_ml_data(self, take_profit_pct, stop_loss_pct, time_barrier_days):
        if self.df.empty: return None, None
            
        self.df['target'] = self._get_triple_barrier_labels(
            take_profit_pct, stop_loss_pct, time_barrier_days
        )
        # Using simple features for now
        base_features = [col for col in self.df.columns if col.startswith(('SMA', 'RSI', 'MACD', 'BB'))]
        X = self.df[base_features]
        y = self.df['target']
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[base_features]
        y = combined['target']
        return X, y

    # --- MODIFIED: Now accepts and passes down strategy parameters ---
    def train_and_predict(self, take_profit_pct, stop_loss_pct, time_barrier_days):
        X, y = self.prepare_ml_data(take_profit_pct, stop_loss_pct, time_barrier_days)
        if X is None or X.empty or len(y.unique()) < 2:
            return None, None, None, None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        if len(X_train) == 0 or len(X_test) == 0:
            return None, None, None, None

        latest_features = X.iloc[[-1]]
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred_on_test = model.predict(X_test)
        report = classification_report(y_test, y_pred_on_test, output_dict=True, zero_division=0, labels=[1, -1, 0])
        prediction = model.predict(latest_features)
        probabilities = model.predict_proba(latest_features)
        accuracy = model.score(X_test, y_test)
        prob_dict = {model.classes_[i]: probabilities[0][i] for i in range(len(model.classes_))}
        return prediction[0], accuracy, prob_dict, report