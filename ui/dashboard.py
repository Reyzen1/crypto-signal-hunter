# ui/dashboard.py
"""
This module defines the UI for the Crypto Signal Hunter dashboard.
Version 3.0: Implemented UI for the Triple Barrier Method strategy.
"""

import streamlit as st
import pandas as pd
from analysis.technical_analyzer import CryptoAnalyzer
from api.coingecko import get_all_coins, get_market_data
# Import the config file to use default values
import config

class Dashboard:
    def __init__(self):
        st.set_page_config(page_title="Crypto Co-pilot AI", page_icon="🤖", layout="wide")

    def _build_sidebar(self):
        """
        Creates the sidebar, now including widgets for strategy parameters.
        """
        st.sidebar.title("🛠️ Configuration")
        def reset_analysis_state(): st.session_state.analysis_requested = False
        
        # --- Section 1: Data Configuration ---
        st.sidebar.subheader("Data Settings")
        all_coins = get_all_coins()
        if not all_coins: st.error("Could not fetch coin list."); st.stop()
        
        selected_coin_name = st.sidebar.selectbox("Cryptocurrency:", options=list(all_coins.keys()), index=list(all_coins.keys()).index("Bitcoin"), on_change=reset_analysis_state)
        days_to_fetch = st.sidebar.slider("Date Range (Days):", min_value=100, max_value=365, value=180, on_change=reset_analysis_state)
        vs_currency = st.sidebar.selectbox("Quote Currency:", options=["usd", "eur", "jpy", "btc"], index=0, on_change=reset_analysis_state)
        
        # --- Section 2: Strategy Configuration (Triple Barrier) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Trading Strategy Settings")
        
        tp_pct = st.sidebar.number_input("Take Profit (%)", min_value=0.5, value=config.TAKE_PROFIT_PCT, step=0.5, on_change=reset_analysis_state)
        sl_pct = st.sidebar.number_input("Stop Loss (%)", min_value=0.5, value=config.STOP_LOSS_PCT, step=0.5, on_change=reset_analysis_state)
        time_limit = st.sidebar.slider("Time Limit (Days)", min_value=3, max_value=30, value=config.TIME_BARRIER_DAYS, on_change=reset_analysis_state)
        
        st.sidebar.markdown("---")
        st.sidebar.button("🚀 Analyze Now", on_click=lambda: st.session_state.update(analysis_requested=True), use_container_width=True)
        st.sidebar.info("Data sourced from CoinGecko API.")
        
        return all_coins[selected_coin_name], selected_coin_name, days_to_fetch, vs_currency, tp_pct, sl_pct, time_limit

    def _display_ml_prediction(self, prediction, accuracy, probabilities):
        """Displays the AI model's prediction for the defined strategy."""
        st.subheader("🔮 AI Strategy Prediction")
        if prediction is None:
            st.warning("Could not generate an AI prediction. More data may be needed.")
            return

        prediction_map = {1: "TAKE PROFIT HIT 🟢", -1: "STOP LOSS HIT 🔴", 0: "TIME LIMIT HIT ⚪️"}
        st.metric(label="Predicted Outcome of Your Strategy", value=prediction_map.get(prediction, "Unknown"))
        st.write("Model Confidence:")
        for class_val, prob in probabilities.items():
            label = prediction_map.get(class_val, f"Class {class_val}")
            st.text(f"{label}:")
            st.progress(prob)
        st.info(f"The model's overall accuracy is **{accuracy:.2%}** on historical test data.", icon="🧠")
        st.caption("Disclaimer: This is an experimental feature and not financial advice.")

    def _display_model_performance(self, report):
        """Displays the detailed model performance in a structured table."""
        # ... (unchanged)
        st.subheader("🔬 Detailed Model Performance")
        if report is None: st.warning("Could not generate performance report."); return
        df_report = pd.DataFrame(report).transpose()
        df_report.rename(index={'-1.0': 'STOP LOSS 🔴', '0.0': 'TIME OUT ⚪️', '1.0': 'TAKE PROFIT 🟢'}, inplace=True)
        st.dataframe(df_report)
        st.caption("**Precision:** How trustworthy are the signals? | **Recall:** How many of the opportunities were found?")

    def run(self):
        """The main execution method for the Streamlit app."""
        st.title("🤖 Crypto Co-pilot AI")

        if 'analysis_requested' not in st.session_state:
            st.session_state.analysis_requested = False

        coin_id, name, days, currency, tp, sl, time_limit = self._build_sidebar()
        
        if not st.session_state.analysis_requested:
            st.info("Configure your analysis in the sidebar and click 'Analyze Now'.")
            st.stop()

        st.header(f"Analysis for: {name} ({currency.upper()})")
        
        with st.spinner("Fetching data and training AI model based on your strategy..."):
            market_data = get_market_data(coin_id, currency, days)
            if market_data:
                try:
                    analyzer = CryptoAnalyzer(market_data)
                    analyzer.add_all_indicators()
                    
                    # --- KEY CHANGE: Pass user-defined strategy parameters to the model ---
                    prediction, accuracy, probabilities, report = analyzer.train_and_predict(
                        take_profit_pct=tp,
                        stop_loss_pct=sl,
                        time_barrier_days=time_limit
                    )
                    
                    self._display_ml_prediction(prediction, accuracy, probabilities)
                    st.markdown("---")
                    self._display_model_performance(report)

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Could not retrieve market data.")