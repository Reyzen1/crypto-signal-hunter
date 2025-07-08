# ui/raw_data_display.py
"""
This module handles the raw data display for the Crypto Signal Hunter dashboard.
Extracted from dashboard.py for better code organization.
"""

import streamlit as st
import pandas as pd
from .messages import Messages

class RawDataDisplay:
    def __init__(self):
        self.messages = Messages()

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
        with st.expander(self.messages.RAW_DATA_EXPANDER_TITLE):
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
            tab1, tab2, tab3 = st.tabs([
                self.messages.TAB_ALL_DATA, 
                self.messages.TAB_TEST_DATA, 
                self.messages.TAB_RECENT_DATA
            ])
            
            with tab1:
                st.subheader(self.messages.COMPLETE_DATASET_TITLE)
                st.dataframe(display_df[available_cols], use_container_width=True)
                
            with tab2:
                st.subheader(self.messages.TEST_DATASET_TITLE)
                test_data = display_df[display_df['Data_Type'] == 'Test 🧪']
                if not test_data.empty:
                    st.dataframe(test_data[available_cols], use_container_width=True)
                    
                    # Show test statistics
                    if 'Prediction_Correctness' in test_data.columns:
                        correct_predictions = (test_data['Prediction_Correctness'] == 'Correct ✅').sum()
                        total_predictions = len(test_data)
                        st.metric(
                            self.messages.TEST_ACCURACY_METRIC,
                            self.messages.TEST_ACCURACY_FORMAT.format(
                                correct=correct_predictions,
                                total=total_predictions,
                                accuracy=correct_predictions/total_predictions
                            )
                        )
                else:
                    st.info(self.messages.NO_TEST_DATA_AVAILABLE)
                    
            with tab3:
                st.subheader(self.messages.RECENT_DATASET_TITLE)
                recent_data = display_df.tail(50)
                st.dataframe(recent_data[available_cols], use_container_width=True)
            
            # Add data explanation
            self._show_data_explanation()
            
            # Show parameters summary
            self._show_parameters_summary(strategy_params, analyzer)

    def _show_data_explanation(self):
        """Display explanation of data columns and signals."""
        st.markdown("---")
        st.subheader(self.messages.DATA_EXPLANATION_TITLE)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(self.messages.BASIC_DATA_GROUP_TITLE)
            st.markdown(self.messages.BASIC_DATA_PRICE_DESC)
            st.markdown(self.messages.BASIC_DATA_VOLUME_DESC)

        with col2:
            st.markdown(self.messages.TARGET_PRICES_GROUP_TITLE)
            st.markdown(self.messages.TARGET_PRICES_TP_PRICE_DESC)
            st.markdown(self.messages.TARGET_PRICES_TP_PERCENTAGE_DESC)
            st.markdown(self.messages.TARGET_PRICES_SL_PRICE_DESC)
            st.markdown(self.messages.TARGET_PRICES_SL_PERCENTAGE_DESC)

        with col3:
            st.markdown(self.messages.STATUS_PREDICTIONS_GROUP_TITLE)
            st.markdown(self.messages.STATUS_PREDICTIONS_DATA_TYPE_DESC)
            st.markdown(self.messages.STATUS_PREDICTIONS_AI_PREDICTION_DESC)
            st.markdown(self.messages.STATUS_PREDICTIONS_TARGET_ACTUAL_DESC)
            st.markdown(self.messages.STATUS_PREDICTIONS_CORRECTNESS_DESC)

        with col4:
            st.markdown(self.messages.LAGGED_FEATURES_GROUP_TITLE)
            st.markdown(self.messages.LAGGED_FEATURES_LAG1_DESC)
            st.markdown(self.messages.LAGGED_FEATURES_LAG2_DESC)
            st.markdown(self.messages.LAGGED_FEATURES_LAG3_DESC)
            st.markdown(self.messages.LAGGED_FEATURES_LAG5_DESC)

        # Add signal meanings
        st.markdown("---")
        st.subheader(self.messages.SIGNAL_MEANINGS_TITLE)

        label_col1, label_col2, label_col3 = st.columns(3)

        with label_col1:
            st.markdown(self.messages.TAKE_PROFIT_SIGNAL_TITLE)
            st.markdown(self.messages.TAKE_PROFIT_SIGNAL_PRICE_DESC)
            st.markdown(self.messages.TAKE_PROFIT_SIGNAL_AI_DESC)
            st.markdown(self.messages.TAKE_PROFIT_SIGNAL_TYPE_DESC)

        with label_col2:
            st.markdown(self.messages.STOP_LOSS_SIGNAL_TITLE)
            st.markdown(self.messages.STOP_LOSS_SIGNAL_PRICE_DESC)
            st.markdown(self.messages.STOP_LOSS_SIGNAL_AI_DESC)
            st.markdown(self.messages.STOP_LOSS_SIGNAL_TYPE_DESC)

        with label_col3:
            st.markdown(self.messages.TIME_LIMIT_SIGNAL_TITLE)
            st.markdown(self.messages.TIME_LIMIT_SIGNAL_PRICE_DESC)
            st.markdown(self.messages.TIME_LIMIT_SIGNAL_BARRIER_DESC)
            st.markdown(self.messages.TIME_LIMIT_SIGNAL_TYPE_DESC)

        # Add strategy explanations
        st.markdown("---")
        st.subheader(self.messages.STRATEGY_EXPLANATION_TITLE)

        strategy_col1, strategy_col2 = st.columns(2)

        with strategy_col1:
            st.markdown(self.messages.DYNAMIC_STRATEGY_EXPLANATION_TITLE)
            st.markdown(self.messages.DYNAMIC_STRATEGY_EXPLANATION_TEXT)

        with strategy_col2:
            st.markdown(self.messages.FIXED_STRATEGY_EXPLANATION_TITLE)
            st.markdown(self.messages.FIXED_STRATEGY_EXPLANATION_TEXT)

    def _show_parameters_summary(self, strategy_params, analyzer):
        """Display parameters summary."""
        st.markdown("---")
        st.subheader(self.messages.PARAMETERS_SUMMARY_TITLE)
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            st.write(self.messages.STRATEGY_PARAMETERS_TITLE)
            strategy_summary = {k: v for k, v in strategy_params.items()}
            st.json(strategy_summary)
        
        with param_col2:
            st.write(self.messages.MODEL_PARAMETERS_TITLE)
            model_summary = analyzer.model_params
            st.json(model_summary)
        
        with param_col3:
            st.write(self.messages.TECHNICAL_INDICATOR_PARAMETERS_TITLE)
            tech_summary = analyzer.technical_params
            st.json(tech_summary)