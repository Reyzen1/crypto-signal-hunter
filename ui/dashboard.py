# ui/dashboard.py
"""
This module defines the user interface for the Crypto Signal Hunter dashboard.

It uses the Streamlit library to build a web-based UI where users can
select cryptocurrencies, view interactive charts, and receive automated
technical analysis signals.
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

    def _build_sidebar(self, all_coins):
        """
        Creates the sidebar for user inputs and configuration.

        Returns:
            A tuple containing user selections:
            (coin_name, coin_id, days, currency, run_button_pressed)
        """
        st.sidebar.title("🛠️ Configuration")

        selected_coin_name = st.sidebar.selectbox(
            "Select Cryptocurrency:",
            options=list(all_coins.keys()),
            index=list(all_coins.keys()).index("Bitcoin")
        )
        selected_coin_id = all_coins[selected_coin_name]

        days_to_fetch = st.sidebar.slider(
            "Select Date Range (Days):",
            min_value=30, max_value=365, value=90
        )
        vs_currency = st.sidebar.selectbox(
            "Select Quote Currency:",
            options=["usd", "eur", "jpy", "btc"],
            index=0
        )
        st.sidebar.markdown("---")
        
        run_button = st.sidebar.button("🚀 Analyze Now")
        
        st.sidebar.info("Data sourced from CoinGecko API.")
        return selected_coin_name, selected_coin_id, days_to_fetch, vs_currency, run_button

    def _build_price_chart(self, df, coin_name, currency):
        """Builds the main interactive price and volume chart."""
        st.subheader(f"Price & Volume Chart for {coin_name}")
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.8, 0.2]
        )
        
        fig.add_trace(go.Scatter(x=df.index, y=df['price'], name="Price", line=dict(color='deepskyblue', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_10'], name="SMA 10", line=dict(color='orange', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_30'], name="SMA 30", line=dict(color='magenta', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color='rgba(128, 128, 128, 0.5)'), row=2, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=500, margin=dict(t=20, b=20, l=20, r=20))
        fig.update_yaxes(title_text="<b>Price</b>", row=1, col=1)
        fig.update_yaxes(title_text="<b>Volume</b>", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    def _build_indicator_charts(self, df):
        """Creates a two-column layout for RSI and MACD charts."""
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Relative Strength Index (RSI)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name="RSI", line=dict(color='rgb(255, 102, 204)', width=1.5)))
            fig.add_shape(type='line', x0=df.index.min(), y0=70, x1=df.index.max(), y1=70, line=dict(color='red', width=2, dash='dash'))
            fig.add_shape(type='line', x0=df.index.min(), y0=30, x1=df.index.max(), y1=30, line=dict(color='green', width=2, dash='dash'))
            fig.update_layout(template="plotly_dark", height=300, yaxis_range=[0,100], margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("MACD")
            macd_line_col, signal_line_col, histogram_col = 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df.index, y=df[histogram_col], name="Histogram", marker_color='rgba(128, 128, 128, 0.5)'))
            fig.add_trace(go.Scatter(x=df.index, y=df[macd_line_col], name="MACD", line=dict(color='rgb(0, 191, 255)', width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df[signal_line_col], name="Signal", line=dict(color='orange', width=1.5, dash='dot')))
            fig.update_layout(template="plotly_dark", height=300, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

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
    
    def run(self):
        """The main execution method that orchestrates the entire dashboard."""
        st.title("🏹 Crypto Signal Hunter")

        all_coins = get_all_coins()
        if not all_coins:
            st.error("Could not fetch coin list. Please check the API status or your internet connection.")
            return

        name, coin_id, days, currency, run_button = self._build_sidebar(all_coins)
        
        # This is a key UX improvement. The main analysis only runs when the user
        # explicitly clicks the button, preventing excessive API calls.
        if not run_button:
            st.info("Configure your analysis in the sidebar and click 'Analyze Now'.")
            st.stop() # Halts the script execution until the button is pressed.

        # --- Main analysis and display block ---
        market_data = get_market_data(coin_id, currency, days)
        if market_data:
            try:
                analyzer = CryptoAnalyzer(market_data)
                processed_df = analyzer.add_all_indicators()
                signals = analyzer.generate_signals()
                
                self._display_signals(signals)
                st.markdown("---")
                self._build_price_chart(processed_df, name, currency)
                self._build_indicator_charts(processed_df)

                with st.expander("Show Raw Data"):
                    st.dataframe(processed_df.tail(10))

            except (ValueError, IndexError) as e:
                st.error(f"Analysis Error: {e}. Try selecting a longer date range.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Could not retrieve market data for the selected options.")