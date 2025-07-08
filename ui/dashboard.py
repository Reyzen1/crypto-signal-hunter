# ui/dashboard.py
"""
This module defines the UI for the Crypto Signal Hunter dashboard.
Version 6.2: Refactored to use separate components for better code organization.
"""

import streamlit as st
from analysis.technical_analyzer import CryptoAnalyzer
from api.coingecko import get_market_data
from ui.sidebar_config import SidebarConfig
from ui.display_components import DisplayComponents
from ui.raw_data_display import RawDataDisplay

class Dashboard:
    def __init__(self):
        st.set_page_config(page_title="Crypto Co-pilot AI", page_icon="🤖", layout="wide")
        self.sidebar_config = SidebarConfig()
        self.display_components = DisplayComponents()
        self.raw_data_display = RawDataDisplay()

    def run(self):
        st.title("🤖 Crypto Co-pilot AI")
        
        if 'analysis_requested' not in st.session_state: 
            st.session_state.analysis_requested = False
        
        coin_id, name, days, currency, strategy_params = self.sidebar_config.build_sidebar()
        
        if not st.session_state.analysis_requested:
            st.info("Configure your analysis in the sidebar and click 'Analyze Now'.")
            st.stop()
        
        st.header(f"Analysis for: {name} ({currency.upper()})")

        strategy_mode = strategy_params['mode']
        mode_emoji = "🔄" if strategy_mode == 'Dynamic' else "🎯"

        if strategy_mode == 'Dynamic':
            st.success(f"""{mode_emoji} **Dynamic Strategy Active** | TP: {strategy_params['tp_multiplier']}x volatility | SL: {strategy_params['sl_multiplier']}x volatility | Time: {strategy_params['time_barrier_days']} days""")
        else:
            st.success(f"""{mode_emoji} **Fixed Strategy Active** | TP: {strategy_params['take_profit_pct']:.1f}% | SL: {strategy_params['stop_loss_pct']:.1f}% | Time: {strategy_params['time_barrier_days']} days""")

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
                        self.display_components.display_ml_prediction(prediction, accuracy, probabilities, strategy_params)
                        st.markdown("---")
                        self.display_components.display_model_performance(report)
                        st.markdown("---")
                        self.display_components.display_error_visualization(full_df, test_results)
                        self.raw_data_display.display_raw_data(full_df, strategy_params, analyzer)

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.exception(e)
            else:
                st.warning("Could not retrieve market data.")