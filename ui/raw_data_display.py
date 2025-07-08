# ui/raw_data_display.py
"""
This module handles the raw data display for the Crypto Signal Hunter dashboard.
Extracted from dashboard.py for better code organization.
"""

import streamlit as st
import pandas as pd

class RawDataDisplay:
    def __init__(self):
        pass

    def _add_target_prices_to_dataframe(self, display_df, strategy_params):
        """Add Take Profit and Stop Loss target prices and percentages to the dataframe."""
        if strategy_params['mode'] == 'Dynamic':
            # Dynamic mode: calculate based on volatility
            display_df['TP_Price'] = display_df.apply(
                lambda row: row['price'] * (1 + row['volatility'] * strategy_params['tp_multiplier']) 
                if pd.notna(row['volatility']) and row['volatility'] > 0 else None, 
                axis=1
            )
            display_df['SL_Price'] = display_df.apply(
                lambda row: row['price'] * (1 - row['volatility'] * strategy_params['sl_multiplier']) 
                if pd.notna(row['volatility']) and row['volatility'] > 0 else None, 
                axis=1
            )
            display_df['TP_Percentage'] = display_df.apply(
                lambda row: row['volatility'] * strategy_params['tp_multiplier'] * 100 
                if pd.notna(row['volatility']) and row['volatility'] > 0 else None, 
                axis=1
            )
            display_df['SL_Percentage'] = display_df.apply(
                lambda row: row['volatility'] * strategy_params['sl_multiplier'] * 100 
                if pd.notna(row['volatility']) and row['volatility'] > 0 else None, 
                axis=1
            )
        else:
            # Fixed mode: use fixed percentages
            tp_pct = strategy_params['take_profit_pct']
            sl_pct = strategy_params['stop_loss_pct']
            
            display_df['TP_Price'] = display_df['price'] * (1 + tp_pct / 100)
            display_df['SL_Price'] = display_df['price'] * (1 - sl_pct / 100)
            display_df['TP_Percentage'] = tp_pct
            display_df['SL_Percentage'] = sl_pct
        
        return display_df

    def display_raw_data(self, full_df, strategy_params, analyzer):
        """Display raw data with tabs and explanations."""
        with st.expander("Show Raw Data & Indicators"):
            display_df = full_df.copy()
            
            # Add target prices and percentages
            display_df = self._add_target_prices_to_dataframe(display_df, strategy_params)
            
            # Define column groups for better organization
            basic_cols = ['price', 'volume']
            target_cols = ['TP_Price', 'TP_Percentage', 'SL_Price', 'SL_Percentage']
            status_cols = ['Data_Type', 'Prediction_Correctness']
            prediction_cols = ['AI_Prediction', 'AI_Prediction_Label', 'Target_Actual', 'Target_Actual_Label']
            
            # Get all indicator columns (original technical indicators)
            indicator_cols = [col for col in display_df.columns 
                            if col not in basic_cols + target_cols + status_cols + prediction_cols
                            and not col.endswith('_lag_1') 
                            and not col.endswith('_lag_2')
                            and not col.endswith('_lag_3')
                            and not col.endswith('_lag_5')
                            and col != 'target']
            
            # Get all lagged feature columns
            lagged_cols = [col for col in display_df.columns 
                         if col.endswith('_lag_1') or col.endswith('_lag_2') 
                         or col.endswith('_lag_3') or col.endswith('_lag_5')]
            
            # Organize columns in a logical order
            reordered_cols = (basic_cols + target_cols + status_cols + prediction_cols + 
                            indicator_cols + lagged_cols)
            
            # Ensure all columns exist in the dataframe
            available_cols = [col for col in reordered_cols if col in display_df.columns]
            
            # Create tabs for different data views
            tab1, tab2, tab3 = st.tabs(["📊 All Data", "🧪 Test Data Only", "📈 Recent 50 Days"])
            
            with tab1:
                st.subheader("Complete Dataset")
                st.dataframe(display_df[available_cols], use_container_width=True)
                
            with tab2:
                st.subheader("Test Dataset with Predictions")
                test_data = display_df[display_df['Data_Type'] == 'Test 🧪']
                if not test_data.empty:
                    st.dataframe(test_data[available_cols], use_container_width=True)
                    
                    # Show test statistics
                    if 'Prediction_Correctness' in test_data.columns:
                        correct_predictions = (test_data['Prediction_Correctness'] == 'Correct ✅').sum()
                        total_predictions = len(test_data)
                        st.metric("Test Accuracy", f"{correct_predictions}/{total_predictions} ({correct_predictions/total_predictions:.1%})")
                else:
                    st.info("No test data available")
                    
            with tab3:
                st.subheader("Recent 50 Days")
                recent_data = display_df.tail(50)
                st.dataframe(recent_data[available_cols], use_container_width=True)
            
            # Add data explanation
            self._show_data_explanation()
            
            # Show parameters summary
            self._show_parameters_summary(strategy_params, analyzer)

    def _show_data_explanation(self):
        """Display explanation of data columns and signals."""
        st.markdown("---")
        st.subheader("📖 Data Explanation")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**📊 Basic Data:**")
            st.markdown("- `price`: Current market price")
            st.markdown("- `volume`: Trading volume")

        with col2:
            st.markdown("**🎯 Target Prices:**")
            st.markdown("- `TP_Price`: Take Profit target price")
            st.markdown("- `TP_Percentage`: Take Profit percentage")
            st.markdown("- `SL_Price`: Stop Loss target price")
            st.markdown("- `SL_Percentage`: Stop Loss percentage")

        with col3:
            st.markdown("**📊 Status & Predictions:**")
            st.markdown("- `Data_Type`: Training 📚 or Test 🧪")
            st.markdown("- `AI_Prediction`: Raw prediction (-1, 0, 1)")
            st.markdown("- `Target_Actual`: Actual outcome")
            st.markdown("- `Prediction_Correctness`: AI accuracy")

        with col4:
            st.markdown("**📈 Lagged Features:**")
            st.markdown("- `_lag_1`: 1 day ago values")
            st.markdown("- `_lag_2`: 2 days ago values")
            st.markdown("- `_lag_3`: 3 days ago values")
            st.markdown("- `_lag_5`: 5 days ago values")

        # Add signal meanings
        st.markdown("---")
        st.subheader("🏷️ Signal Meanings")

        label_col1, label_col2, label_col3 = st.columns(3)

        with label_col1:
            st.markdown("**🟢 TAKE PROFIT:**")
            st.markdown("- Price reached profit target")
            st.markdown("- AI suggests buying")
            st.markdown("- Bullish signal")

        with label_col2:
            st.markdown("**🔴 STOP LOSS:**")
            st.markdown("- Price hit stop loss level")
            st.markdown("- AI suggests selling")
            st.markdown("- Bearish signal")

        with label_col3:
            st.markdown("**⚪️ TIME LIMIT:**")
            st.markdown("- Neither target reached")
            st.markdown("- Time barrier hit")
            st.markdown("- Neutral/Hold signal")

        # Add strategy explanations
        st.markdown("---")
        st.subheader("🧠 Strategy Explanation")

        strategy_col1, strategy_col2 = st.columns(2)

        with strategy_col1:
            st.markdown("**🔄 Dynamic Strategy:**")
            st.markdown("""
            - Targets calculated based on market volatility
            - Higher volatility = Higher targets
            - Adapts to market conditions automatically
            - Formula: `Target = Price × (1 ± Volatility × Multiplier)`
            """)

        with strategy_col2:
            st.markdown("**🎯 Fixed Strategy:**")
            st.markdown("""
            - Fixed percentage targets set by user
            - Consistent targets regardless of market conditions
            - More predictable risk/reward ratio
            - Formula: `Target = Price × (1 ± Percentage/100)`
            """)

    def _show_parameters_summary(self, strategy_params, analyzer):
        """Display parameters summary."""
        st.markdown("---")
        st.subheader("🔧 Parameters Summary")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            st.write("**Strategy Parameters:**")
            strategy_summary = {k: v for k, v in strategy_params.items()}
            st.json(strategy_summary)
        
        with param_col2:
            st.write("**Model Parameters:**")
            model_summary = analyzer.model_params
            st.json(model_summary)
        
        with param_col3:
            st.write("**Technical Indicator Parameters:**")
            tech_summary = analyzer.technical_params
            st.json(tech_summary)