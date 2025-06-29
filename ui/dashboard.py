# ui/dashboard.py
"""
This module defines the user interface for the Crypto Signal Hunter dashboard.
Version 2.9: Added a detailed classification report for model evaluation.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis.technical_analyzer import CryptoAnalyzer
from api.coingecko import get_all_coins, get_market_data

class Dashboard:
    """
    Manages the creation and orchestration of all UI components for the dashboard.
    """
    def __init__(self):
        """Initializes the dashboard and sets the Streamlit page configuration."""
        st.set_page_config(
            page_title="Crypto Signal Hunter",
            page_icon="🏹",
            layout="wide"
        )

    def _build_sidebar(self):
        """
        Creates the sidebar for user inputs and configuration.
        """
        st.sidebar.title("🛠️ Configuration")

        def reset_analysis_state():
            st.session_state.analysis_requested = False

        all_coins = get_all_coins()
        if not all_coins:
            st.error("Could not fetch coin list.")
            st.stop()

        selected_coin_name = st.sidebar.selectbox(
            "Select Cryptocurrency:",
            options=list(all_coins.keys()),
            index=list(all_coins.keys()).index("Bitcoin"),
            on_change=reset_analysis_state
        )
        
        days_to_fetch = st.sidebar.slider(
            "Select Date Range (Days):",
            min_value=100, max_value=365, value=180,
            on_change=reset_analysis_state
        )
        
        vs_currency = st.sidebar.selectbox(
            "Select Quote Currency:",
            options=["usd", "eur", "jpy", "btc"],
            index=0,
            on_change=reset_analysis_state
        )
        
        st.sidebar.markdown("---")
        
        st.sidebar.button(
            "🚀 Analyze Now",
            on_click=lambda: st.session_state.update(analysis_requested=True),
            use_container_width=True
        )
        
        st.sidebar.info("Data sourced from CoinGecko API.")
        return all_coins[selected_coin_name], selected_coin_name, days_to_fetch, vs_currency

    def _display_ml_prediction(self, prediction, accuracy, probabilities):
        """
        Displays the AI model's 3-class prediction and its confidence probabilities.
        """
        st.subheader("🔮 AI Price Direction Prediction")
        
        if prediction is None or accuracy is None or probabilities is None:
            st.warning("Could not generate an AI prediction. More historical data may be needed.")
            return

        prediction_map = {1: "BUY 🟢", -1: "SELL 🔴", 0: "HOLD ⚪️"}
        
        st.metric(label="AI Signal for Next Day", value=prediction_map.get(prediction, "Unknown"))

        st.write("Model Confidence:")
        
        for class_val, prob in probabilities.items():
            label = prediction_map.get(class_val, f"Class {class_val}")
            st.text(f"{label}:")
            st.progress(prob)

        st.info(f"The model's overall accuracy is **{accuracy:.2%}** on historical test data.", icon="🧠")
        
    def _display_model_performance(self, report):
        """Displays the detailed model performance in a structured table."""
        st.subheader("🔬 Detailed Model Performance")

        if report is None:
            st.warning("Could not generate a performance report.")
            return

        df_report = pd.DataFrame(report).transpose()
        df_report.rename(index={'-1.0': 'SELL 🔴', '0.0': 'HOLD ⚪️', '1.0': 'BUY 🟢'}, inplace=True)
        
        st.dataframe(df_report)
        st.caption("""
        **Precision:** Of all predictions for a class, how many were correct? (e.g., How trustworthy are the 'BUY' signals?)
        **Recall:** Of all actual instances of a class, how many did the model find? (e.g., How many of the real 'BUY' opportunities were found?)
        """)

    def run(self):
        """The main execution method that orchestrates the entire dashboard."""
        st.title("🏹 Crypto Signal Hunter")

        if 'analysis_requested' not in st.session_state:
            st.session_state.analysis_requested = False

        coin_id, name, days, currency = self._build_sidebar()
        
        if not st.session_state.analysis_requested:
            st.info("Configure your analysis in the sidebar and click 'Analyze Now'.")
            st.stop()

        st.header(f"Analysis for: {name} ({currency.upper()})")
        
        with st.spinner("Fetching data and training AI model... This may take a moment."):
            market_data = get_market_data(coin_id, currency, days)
            if market_data:
                try:
                    analyzer = CryptoAnalyzer(market_data)
                    analyzer.add_all_indicators()

                    prediction, accuracy, probabilities, report = analyzer.train_and_predict()
                    
                    self._display_ml_prediction(prediction, accuracy, probabilities)
                    st.markdown("---")
                    self._display_model_performance(report)
                    
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Could not retrieve market data for the selected options.")