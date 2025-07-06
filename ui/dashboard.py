# ui/dashboard.py
"""
This module defines the UI for the Crypto Signal Hunter dashboard.
Version 6.0: Added dual strategy support (Dynamic and Fixed modes).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from analysis.technical_analyzer import CryptoAnalyzer
from api.coingecko import get_all_coins, get_market_data
import config

class Dashboard:
    def __init__(self):
        st.set_page_config(page_title="Crypto Co-pilot AI", page_icon="🤖", layout="wide")

    def _calculate_percentage_from_price(self, current_price, target_price):
        """Calculate percentage change from current price to target price."""
        if current_price <= 0:
            return 0.0
        return ((target_price - current_price) / current_price) * 100

    def _calculate_price_from_percentage(self, current_price, percentage):
        """Calculate target price from current price and percentage change."""
        return current_price * (1 + percentage / 100)

    def _build_sidebar(self):
        st.sidebar.title("🛠️ Configuration")
        def reset_analysis_state(): 
            st.session_state.analysis_requested = False
            if 'current_price' in st.session_state:
                del st.session_state.current_price
        
        st.sidebar.subheader("Data Settings")
        all_coins = get_all_coins()
        if not all_coins: 
            st.error("Could not fetch coin list.")
            st.stop()
        
        selected_coin_name = st.sidebar.selectbox(
            "Cryptocurrency:", 
            options=list(all_coins.keys()), 
            index=list(all_coins.keys()).index(config.DEFAULT_COIN_NAME), 
            on_change=reset_analysis_state
        )
        
        days_to_fetch = st.sidebar.slider(
            "Date Range (Days):", 
            min_value=100, 
            max_value=365, 
            value=config.DEFAULT_DAYS_TO_FETCH, 
            on_change=reset_analysis_state
        )
        
        vs_currency = st.sidebar.selectbox(
            "Quote Currency:", 
            options=["usd", "eur", "jpy", "btc"], 
            index=["usd", "eur", "jpy", "btc"].index(config.DEFAULT_CURRENCY), 
            on_change=reset_analysis_state
        )
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Strategy Mode")
        
        # Strategy mode selection
        strategy_mode = st.sidebar.radio(
            "Select Strategy Mode:",
            options=['Dynamic', 'Fixed'],
            index=0 if config.DEFAULT_STRATEGY_MODE == 'Dynamic' else 1,
            on_change=reset_analysis_state,
            help="Dynamic: Based on market volatility | Fixed: Based on your targets"
        )
        
        st.sidebar.markdown("---")
        
        # Get current price for Fixed mode calculations
        if strategy_mode == 'Fixed':
            if 'current_price' not in st.session_state:
                # Fetch current price
                market_data = get_market_data(all_coins[selected_coin_name], vs_currency, 1)
                if market_data and 'prices' in market_data and len(market_data['prices']) > 0:
                    st.session_state.current_price = market_data['prices'][-1][1]
                else:
                    st.session_state.current_price = 0.0
            
            current_price = st.session_state.current_price
        
        # Strategy-specific settings
        if strategy_mode == 'Dynamic':
            st.sidebar.subheader("🔄 Dynamic Strategy Settings")
            st.sidebar.info("Targets are calculated based on market volatility")
            
            tp_multiplier = st.sidebar.number_input(
                "Take Profit Multiplier:", 
                min_value=0.5, 
                max_value=5.0,
                value=config.VOLATILITY_TAKE_PROFIT_MULTIPLIER, 
                step=0.1, 
                on_change=reset_analysis_state,
                help="Higher values = more aggressive profit targets"
            )
            
            sl_multiplier = st.sidebar.number_input(
                "Stop Loss Multiplier:", 
                min_value=0.5, 
                max_value=5.0,
                value=config.VOLATILITY_STOP_LOSS_MULTIPLIER, 
                step=0.1, 
                on_change=reset_analysis_state,
                help="Higher values = more conservative stop losses"
            )
            
            strategy_params = {
                'mode': 'Dynamic',
                'tp_multiplier': tp_multiplier,
                'sl_multiplier': sl_multiplier
            }
            
        else:  # Fixed mode
            st.sidebar.subheader("🎯 Fixed Strategy Settings")
            st.sidebar.info(f"Current Price: {current_price:.4f} {vs_currency.upper()}")
            
            # Take Profit Section
            st.sidebar.write("**Take Profit Target:**")
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                tp_price = st.number_input(
                    "Price:",
                    min_value=0.0,
                    value=self._calculate_price_from_percentage(current_price, config.TAKE_PROFIT_PCT),
                    step=current_price * 0.001,
                    format="%.6f",
                    key="tp_price",
                    on_change=reset_analysis_state
                )
            
            with col2:
                tp_pct = st.number_input(
                    "Percentage:",
                    value=self._calculate_percentage_from_price(current_price, tp_price),
                    step=0.1,
                    format="%.2f",
                    key="tp_pct",
                    on_change=reset_analysis_state
                )
            
            # Sync price and percentage for Take Profit
            if st.session_state.get('tp_price') != tp_price:
                tp_pct = self._calculate_percentage_from_price(current_price, tp_price)
                st.session_state.tp_pct = tp_pct
            elif st.session_state.get('tp_pct') != tp_pct:
                tp_price = self._calculate_price_from_percentage(current_price, tp_pct)
                st.session_state.tp_price = tp_price
            
            # Stop Loss Section
            st.sidebar.write("**Stop Loss Target:**")
            col3, col4 = st.sidebar.columns(2)
            
            with col3:
                sl_price = st.number_input(
                    "Price:",
                    min_value=0.0,
                    value=self._calculate_price_from_percentage(current_price, -config.STOP_LOSS_PCT),
                    step=current_price * 0.001,
                    format="%.6f",
                    key="sl_price",
                    on_change=reset_analysis_state
                )
            
            with col4:
                sl_pct = st.number_input(
                    "Percentage:",
                    value=abs(self._calculate_percentage_from_price(current_price, sl_price)),
                    step=0.1,
                    format="%.2f",
                    key="sl_pct",
                    on_change=reset_analysis_state
                )
            
            # Sync price and percentage for Stop Loss
            if st.session_state.get('sl_price') != sl_price:
                sl_pct = abs(self._calculate_percentage_from_price(current_price, sl_price))
                st.session_state.sl_pct = sl_pct
            elif st.session_state.get('sl_pct') != sl_pct:
                sl_price = self._calculate_price_from_percentage(current_price, -sl_pct)
                st.session_state.sl_price = sl_price
            
            strategy_params = {
                'mode': 'Fixed',
                'take_profit_pct': tp_pct,
                'stop_loss_pct': sl_pct
            }
        
        # Common settings
        st.sidebar.markdown("---")
        st.sidebar.subheader("⏰ Time Settings")
        time_limit = st.sidebar.slider(
            "Time Limit (Days)", 
            min_value=3, 
            max_value=30, 
            value=config.TIME_BARRIER_DAYS, 
            on_change=reset_analysis_state
        )
        
        strategy_params['time_barrier_days'] = time_limit
        
        st.sidebar.markdown("---")
        st.sidebar.button(
            "🚀 Analyze Now", 
            on_click=lambda: st.session_state.update(analysis_requested=True), 
            use_container_width=True
        )
        
        return all_coins[selected_coin_name], selected_coin_name, days_to_fetch, vs_currency, strategy_params

    def _display_ml_prediction(self, prediction, accuracy, probabilities, strategy_params):
        st.subheader("🔮 AI Strategy Prediction")
        
        # Display strategy mode
        mode_emoji = "🔄" if strategy_params['mode'] == 'Dynamic' else "🎯"
        st.info(f"{mode_emoji} **{strategy_params['mode']} Strategy Mode** is active", icon="ℹ️")
        
        prediction_map = {1.0: "TAKE PROFIT HIT 🟢", -1.0: "STOP LOSS HIT 🔴", 0.0: "TIME LIMIT HIT ⚪️"}
        st.metric(label="AI Signal for the Last Day", value=prediction_map.get(prediction, "Unknown"))
        
        st.write("Model Confidence:")
        if probabilities:
            for class_val, prob in probabilities.items():
                label = prediction_map.get(class_val, f"Class {class_val}")
                st.text(f"{label}:")
                st.progress(prob)
        
        st.info(f"Model's overall accuracy on test data: **{accuracy:.2%}**.", icon="🧠")

    def _display_model_performance(self, report):
        st.subheader("🔬 Detailed Model Performance")
        if report is None: 
            return
        
        df_report = pd.DataFrame(report).transpose()
        df_report.index = df_report.index.astype(str)
        df_report.rename(index={
            '-1.0': 'STOP LOSS 🔴', 
            '0.0': 'TIME OUT ⚪️', 
            '1.0': 'TAKE PROFIT 🟢'
        }, inplace=True)
        st.dataframe(df_report)
        st.caption("**Precision:** Signal Quality | **Recall:** Opportunity Finding")

    def _display_error_visualization(self, main_df, test_results_df):
        st.subheader("📊 Prediction Analysis on Test Data")
        if test_results_df is None or test_results_df.empty:
            st.warning("No test data available to visualize.")
            return
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=main_df.index, 
            y=main_df['price'], 
            mode='lines', 
            name='Price', 
            line=dict(color='deepskyblue')
        ))
        
        correct_buys = test_results_df[(test_results_df['predicted'] == 1) & (test_results_df['actual'] == 1)]
        fig.add_trace(go.Scatter(
            x=correct_buys.index, 
            y=main_df.loc[correct_buys.index]['price'], 
            mode='markers', 
            name='Correct BUY', 
            marker=dict(color='limegreen', size=10, symbol='triangle-up')
        ))
        
        incorrect_buys = test_results_df[(test_results_df['predicted'] == 1) & (test_results_df['actual'] != 1)]
        fig.add_trace(go.Scatter(
            x=incorrect_buys.index, 
            y=main_df.loc[incorrect_buys.index]['price'], 
            mode='markers', 
            name='Incorrect BUY', 
            marker=dict(color='yellow', size=10, symbol='triangle-up', line=dict(width=1, color='black'))
        ))
        
        correct_sells = test_results_df[(test_results_df['predicted'] == -1) & (test_results_df['actual'] == -1)]
        fig.add_trace(go.Scatter(
            x=correct_sells.index, 
            y=main_df.loc[correct_sells.index]['price'], 
            mode='markers', 
            name='Correct SELL', 
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
        
        incorrect_sells = test_results_df[(test_results_df['predicted'] == -1) & (test_results_df['actual'] != -1)]
        fig.add_trace(go.Scatter(
            x=incorrect_sells.index, 
            y=main_df.loc[incorrect_sells.index]['price'], 
            mode='markers', 
            name='Incorrect SELL', 
            marker=dict(color='orange', size=10, symbol='triangle-down', line=dict(width=1, color='black'))
        ))
        
        fig.update_layout(
            template="plotly_dark", 
            height=500, 
            margin=dict(t=30, b=20, l=20, r=20), 
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    def run(self):
        st.title("🤖 Crypto Co-pilot AI")
        
        if 'analysis_requested' not in st.session_state: 
            st.session_state.analysis_requested = False
        
        coin_id, name, days, currency, strategy_params = self._build_sidebar()
        
        if not st.session_state.analysis_requested:
            st.info("Configure your analysis in the sidebar and click 'Analyze Now'.")
            st.stop()
        
        st.header(f"Analysis for: {name} ({currency.upper()})")
        
        with st.spinner("Fetching data and training AI model..."):
            market_data = get_market_data(coin_id, currency, days)
            
            if market_data:
                try:
                    analyzer = CryptoAnalyzer(market_data)
                    analyzer.add_all_indicators()
                    
                    prediction, accuracy, probabilities, report, test_results, full_df = analyzer.train_and_predict(
                        strategy_params
                    )
                    
                    if prediction is None:
                        st.warning("🚨 AI Model could not be trained.")
                        st.info("Not enough data remained or all outcomes were the same. Try adjusting parameters.")
                        _, y, _ = analyzer.prepare_ml_data(strategy_params)
                        if y is not None:
                            st.dataframe(y.value_counts())
                    else:
                        self._display_ml_prediction(prediction, accuracy, probabilities, strategy_params)
                        st.markdown("---")
                        self._display_model_performance(report)
                        st.markdown("---")
                        self._display_error_visualization(full_df, test_results)
                        
                        with st.expander("Show Raw Data & Indicators"):
                            #display_df = full_df.tail(100).copy()
                            display_df = full_df.copy()
                            important_cols = ['price', 'volume', 'AI_Prediction', 'AI_Prediction_Label', 'target']
                            other_cols = [col for col in display_df.columns if col not in important_cols]
                            reordered_cols = important_cols + other_cols
                            st.dataframe(display_df[reordered_cols])
                            
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.exception(e)
            else:
                st.warning("Could not retrieve market data.")