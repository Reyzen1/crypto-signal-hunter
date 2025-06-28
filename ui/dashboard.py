# ui/dashboard.py
"""
This module defines the user interface for the Crypto Signal Hunter dashboard.

It uses Streamlit to build a web-based UI where users can select
cryptocurrencies, view interactive charts, and receive automated technical
analysis signals. This version uses Streamlit's Session State to provide a
robust and stateful user experience, preventing UI resets on reruns.
"""

import streamlit as st
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

        This method builds the widgets for selecting coin, days, and currency.
        It uses callbacks to reset the analysis state when inputs change,
        and a main button to trigger the analysis.

        Returns:
            A tuple containing user selections: (coin_name, coin_id, days, currency)
        """
        st.sidebar.title("🛠️ Configuration")

        # A callback function to reset the analysis state.
        # This forces the user to click "Analyze Now" again after changing parameters,
        # ensuring the displayed data matches the selected inputs.
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
            min_value=30, max_value=365, value=90,
            on_change=reset_analysis_state
        )
        
        vs_currency = st.sidebar.selectbox(
            "Select Quote Currency:",
            options=["usd", "eur", "jpy", "btc"],
            index=0,
            on_change=reset_analysis_state
        )
        
        st.sidebar.markdown("---")
        
        # This button sets the 'analysis_requested' flag in session state to True.
        # This state persists across reruns until reset.
        st.sidebar.button(
            "🚀 Analyze Now",
            on_click=lambda: st.session_state.update(analysis_requested=True),
            use_container_width=True
        )
        
        st.sidebar.info("Data sourced from CoinGecko API.")
        return all_coins[selected_coin_name], selected_coin_name, days_to_fetch, vs_currency

    def _display_signals(self, signals):
        """Displays the generated signals in a two-column layout."""
        st.subheader("🤖 Automated Signal Analysis")
        if not signals:
            st.info("No specific trading signals detected for the latest data point.")
            return

        rsi_signals = [s for s in signals if 'RSI' in s[1]]
        macd_signals = [s for s in signals if 'MACD' in s[1]]

        col1, col2 = st.columns(2)
        with col1:
            for signal_type, message in rsi_signals:
                if signal_type == 'success': st.success(message, icon="📈")
                elif signal_type == 'warning': st.warning(message, icon="⚠️")
                elif signal_type == 'info': st.info(message, icon="ℹ️")
        with col2:
            for signal_type, message in macd_signals:
                if signal_type == 'success': st.success(message, icon="🔥")
                elif signal_type == 'error': st.error(message, icon="🚨")
                elif signal_type == 'info': st.info(message, icon="📈")
                elif signal_type == 'warning': st.warning(message, icon="📉")

    def _display_charts(self, df, coin_name):
        """Renders the main price/volume chart and the indicator charts."""
        st.subheader(f"Charts for {coin_name}")
        
        # Price and Volume Chart
        fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])
        fig_price.add_trace(go.Scatter(x=df.index, y=df['price'], name="Price", line=dict(color='deepskyblue', width=2)), row=1, col=1)
        fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_10'], name="SMA 10", line=dict(color='orange', width=1, dash='dash')), row=1, col=1)
        fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_30'], name="SMA 30", line=dict(color='magenta', width=1, dash='dot')), row=1, col=1)
        fig_price.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color='rgba(128, 128, 128, 0.5)'), row=2, col=1)
        fig_price.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=500, margin=dict(t=0, b=20, l=20, r=20))
        st.plotly_chart(fig_price, use_container_width=True)

        # Indicator Charts (RSI and MACD)
        col1, col2 = st.columns(2)
        with col1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name="RSI", line=dict(color='rgb(255, 102, 204)', width=1.5)))
            fig_rsi.add_shape(type='line', x0=df.index.min(), y0=70, x1=df.index.max(), y1=70, line=dict(color='red', width=2, dash='dash'))
            fig_rsi.add_shape(type='line', x0=df.index.min(), y0=30, x1=df.index.max(), y1=30, line=dict(color='green', width=2, dash='dash'))
            fig_rsi.update_layout(title="RSI", template="plotly_dark", height=300, yaxis_range=[0,100], margin=dict(t=30, b=20, l=20, r=20))
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            macd_line_col, signal_line_col, histogram_col = 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=df.index, y=df[histogram_col], name="Histogram", marker_color='rgba(128, 128, 128, 0.5)'))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df[macd_line_col], name="MACD", line=dict(color='rgb(0, 191, 255)', width=1.5)))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df[signal_line_col], name="Signal", line=dict(color='orange', width=1.5, dash='dot')))
            fig_macd.update_layout(title="MACD", template="plotly_dark", height=300, margin=dict(t=30, b=20, l=20, r=20))
            st.plotly_chart(fig_macd, use_container_width=True)

    def run(self):
        """The main execution method that orchestrates the entire dashboard."""
        st.title("🏹 Crypto Signal Hunter")

        # Initialize session state if it doesn't exist. This is the app's "memory".
        if 'analysis_requested' not in st.session_state:
            st.session_state.analysis_requested = False

        coin_id, name, days, currency = self._build_sidebar()
        
        # The main logic only runs if the 'analysis_requested' state is True.
        # This prevents the app from resetting after the initial run.
        if not st.session_state.analysis_requested:
            st.info("Configure your analysis in the sidebar and click 'Analyze Now'.")
            st.stop() # Halts the script until the state is changed by the button click.

        # --- Main analysis and display block ---
        st.header(f"Analysis for: {name} ({currency.upper()})")
        market_data = get_market_data(coin_id, currency, days)
        if market_data:
            try:
                analyzer = CryptoAnalyzer(market_data)
                processed_df = analyzer.add_all_indicators()

                if processed_df.empty:
                    st.warning("Not enough data for the selected period to calculate all indicators. Please select a longer date range.")
                    st.stop()
                    
                signals = analyzer.generate_signals()
                
                self._display_signals(signals)
                st.markdown("---")
                self._display_charts(processed_df, name)

                with st.expander("Show Raw Data & Indicators"):
                    st.dataframe(processed_df.tail(10))

            except (ValueError, IndexError) as e:
                st.error(f"Analysis Error: {e}. Try selecting a longer date range.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Could not retrieve market data for the selected options.")