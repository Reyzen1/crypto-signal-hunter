# ui/dashboard.py
"""
This module defines the UI for the Crypto Signal Hunter dashboard.
Version 6.3: Enhanced with better information display and risk disclaimers.
"""

import streamlit as st
from analysis.technical_analyzer import CryptoAnalyzer
from api.coingecko import get_market_data
from ui.sidebar_config import SidebarConfig
from ui.display_components import DisplayComponents
from ui.raw_data_display import RawDataDisplay
from ui.messages import Messages

class Dashboard:
    def __init__(self):
        st.set_page_config(page_title="Crypto Co-pilot AI", page_icon="🤖", layout="wide")
        self.sidebar_config = SidebarConfig()
        self.display_components = DisplayComponents()
        self.raw_data_display = RawDataDisplay()
        self.messages = Messages()

    def show_welcome_section(self):
        """Display welcome section with system information when analysis is not requested."""
        st.markdown(self.messages.WELCOME_SECTION)
        
        # Add disclaimer section
        st.error(self.messages.RISK_DISCLAIMER)
        
        st.info(self.messages.GETTING_STARTED_MSG)

    def run(self):
        if 'analysis_requested' not in st.session_state: 
            st.session_state.analysis_requested = False
        
        coin_id, name, days, currency, strategy_params = self.sidebar_config.build_sidebar()
        
        if not st.session_state.analysis_requested:
            self.show_welcome_section()
            st.stop()
        
        st.title(self.messages.DASHBOARD_TITLE)
        st.header(self.messages.ANALYSIS_HEADER.format(name=name, currency=currency.upper()))

        strategy_mode = strategy_params['mode']
        mode_emoji = "🔄" if strategy_mode == 'Dynamic' else "🎯"

        if strategy_mode == 'Dynamic':
            st.success(self.messages.DYNAMIC_STRATEGY_MSG.format(
                tp_multiplier=strategy_params['tp_multiplier'],
                sl_multiplier=strategy_params['sl_multiplier'],
                time_barrier_days=strategy_params['time_barrier_days']
            ))
        else:
            st.success(self.messages.FIXED_STRATEGY_MSG.format(
                take_profit_pct=strategy_params['take_profit_pct'],
                stop_loss_pct=strategy_params['stop_loss_pct'],
                time_barrier_days=strategy_params['time_barrier_days']
            ))

        with st.spinner(self.messages.FETCHING_DATA_MSG):
            market_data = get_market_data(coin_id, currency, days)
            
            if market_data:
                try:
                    analyzer = CryptoAnalyzer(market_data)
                    analyzer.add_all_indicators()
                    
                    prediction, accuracy, probabilities, report, test_results, full_df = analyzer.train_and_predict(
                        strategy_params
                    )
                    
                    if prediction is None:
                        st.warning(self.messages.MODEL_TRAINING_ERROR)
                        st.info(self.messages.INSUFFICIENT_DATA_MSG)
                        _, y, _ = analyzer.prepare_ml_data(strategy_params)
                        if y is not None:
                            st.dataframe(y.value_counts())
                    else:
                        self.display_components.display_ml_prediction(prediction, accuracy, probabilities, strategy_params, full_df)
                        st.markdown("---")
                        self.display_components.display_model_performance(report)
                        st.markdown("---")
                        self.display_components.display_error_visualization(full_df, test_results)
                        self.raw_data_display.display_raw_data(full_df, strategy_params, analyzer)

                except Exception as e:
                    st.error(self.messages.UNEXPECTED_ERROR_MSG.format(error=e))
                    st.exception(e)
            else:
                st.warning(self.messages.DATA_RETRIEVAL_ERROR)