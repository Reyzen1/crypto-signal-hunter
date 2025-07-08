# ui/sidebar_config.py
"""
This module handles the sidebar configuration for the Crypto Signal Hunter dashboard.
Extracted from dashboard.py for better code organization.
"""

import streamlit as st
from api.coingecko import get_all_coins, get_market_data
import config
from ui.messages import Messages

class SidebarConfig:
    def __init__(self):
        pass

    def _calculate_percentage_from_price(self, current_price, target_price):
        """Calculate percentage change from current price to target price."""
        if current_price <= 0:
            return 0.0
        return ((target_price - current_price) / current_price) * 100

    def _calculate_price_from_percentage(self, current_price, percentage):
        """Calculate target price from current price and percentage change."""
        return current_price * (1 + percentage / 100)

    def build_sidebar(self):
        st.sidebar.title(Messages.SIDEBAR_TITLE)
        def reset_analysis_state(): 
            st.session_state.analysis_requested = False
            if 'current_price' in st.session_state:
                del st.session_state.current_price
        
        st.sidebar.subheader(Messages.DATA_SETTINGS_TITLE)
        all_coins = get_all_coins()
        if not all_coins: 
            st.error(Messages.COIN_LIST_ERROR)
            st.stop()
        
        selected_coin_name = st.sidebar.selectbox(
            Messages.CRYPTOCURRENCY_LABEL, 
            options=list(all_coins.keys()), 
            index=list(all_coins.keys()).index(config.DEFAULT_COIN_NAME), 
            on_change=reset_analysis_state
        )
        
        days_to_fetch = st.sidebar.slider(
            Messages.DATE_RANGE_LABEL, 
            min_value=100, 
            max_value=365, 
            value=config.DEFAULT_DAYS_TO_FETCH, 
            on_change=reset_analysis_state
        )
        
        vs_currency = st.sidebar.selectbox(
            Messages.QUOTE_CURRENCY_LABEL, 
            options=["usd", "eur", "jpy", "btc"], 
            index=["usd", "eur", "jpy", "btc"].index(config.DEFAULT_CURRENCY), 
            on_change=reset_analysis_state
        )
        
        st.sidebar.markdown("---")
        st.sidebar.subheader(Messages.STRATEGY_MODE_TITLE)
        
        # Strategy mode selection
        strategy_mode = st.sidebar.radio(
            Messages.STRATEGY_MODE_LABEL,
            options=['Dynamic', 'Fixed'],
            index=0 if config.DEFAULT_STRATEGY_MODE == 'Dynamic' else 1,
            on_change=reset_analysis_state,
            help=Messages.STRATEGY_MODE_HELP
        )

        with st.sidebar.expander(Messages.HOW_DOES_THIS_WORK, expanded=False):
            if strategy_mode == 'Dynamic':
                st.markdown(Messages.DYNAMIC_STRATEGY_HELP_TEXT)
            else:
                st.markdown(Messages.FIXED_STRATEGY_HELP_TEXT)

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
            st.sidebar.subheader(Messages.DYNAMIC_STRATEGY_TITLE)
            st.sidebar.info(Messages.DYNAMIC_STRATEGY_INFO)
            
            tp_multiplier = st.sidebar.number_input(
                Messages.TAKE_PROFIT_MULTIPLIER_LABEL, 
                min_value=0.5, 
                max_value=5.0,
                value=config.VOLATILITY_TAKE_PROFIT_MULTIPLIER, 
                step=0.1, 
                on_change=reset_analysis_state,
                help=Messages.TAKE_PROFIT_MULTIPLIER_HELP
            )
            
            sl_multiplier = st.sidebar.number_input(
                Messages.STOP_LOSS_MULTIPLIER_LABEL, 
                min_value=0.5, 
                max_value=5.0,
                value=config.VOLATILITY_STOP_LOSS_MULTIPLIER, 
                step=0.1, 
                on_change=reset_analysis_state,
                help=Messages.STOP_LOSS_MULTIPLIER_HELP
            )
            
            strategy_params = {
                'mode': 'Dynamic',
                'tp_multiplier': tp_multiplier,
                'sl_multiplier': sl_multiplier
            }
            
        else:  # Fixed mode
            st.sidebar.subheader(Messages.FIXED_STRATEGY_TITLE)
            st.sidebar.info(Messages.CURRENT_PRICE_LABEL_SIDEBAR.format(price=current_price, currency=vs_currency.upper()))
            
            # Take Profit Section
            st.sidebar.write(f"**{Messages.TAKE_PROFIT_TARGET_LABEL}**")
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                tp_price = st.number_input(
                    f"{Messages.PRICE_LABEL}",
                    min_value=0.0,
                    value=self._calculate_price_from_percentage(current_price, config.TAKE_PROFIT_PCT),
                    step=current_price * 0.001,
                    format="%.6f",
                    key="tp_price",
                    on_change=reset_analysis_state
                )
            
            with col2:
                tp_pct = st.number_input(
                    f"{Messages.PERCENTAGE_LABEL}",
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
            st.sidebar.write(f"**{Messages.STOP_LOSS_TARGET_LABEL}**")
            col3, col4 = st.sidebar.columns(2)
            
            with col3:
                sl_price = st.number_input(
                    f"{Messages.PRICE_LABEL}",
                    min_value=0.0,
                    value=self._calculate_price_from_percentage(current_price, -config.STOP_LOSS_PCT),
                    step=current_price * 0.001,
                    format="%.6f",
                    key="sl_price",
                    on_change=reset_analysis_state
                )
            
            with col4:
                sl_pct = st.number_input(
                    f"{Messages.PERCENTAGE_LABEL}",
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
        st.sidebar.subheader(Messages.TIME_SETTINGS_TITLE)
        time_limit = st.sidebar.slider(
            Messages.TIME_LIMIT_LABEL, 
            min_value=3, 
            max_value=30, 
            value=config.TIME_BARRIER_DAYS, 
            on_change=reset_analysis_state,
            help=Messages.TIME_LIMIT_HELP
        )
        
        strategy_params['time_barrier_days'] = time_limit
        
        st.sidebar.markdown("---")
        st.sidebar.button(
            Messages.ANALYZE_NOW_BUTTON, 
            on_click=lambda: st.session_state.update(analysis_requested=True), 
            use_container_width=True
        )
        
        return all_coins[selected_coin_name], selected_coin_name, days_to_fetch, vs_currency, strategy_params