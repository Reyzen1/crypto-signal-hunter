# ui/display_components.py
"""
This module handles the display components for the Crypto Signal Hunter dashboard.
Version 6.3: Enhanced with more detailed prediction information and market insights.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from ui.messages import Messages

class DisplayComponents:
    def __init__(self):
        self.messages = Messages()

    def display_ml_prediction(self, prediction, accuracy, probabilities, strategy_params, full_df):
        st.subheader(self.messages.AI_PREDICTION_TITLE)
        
        # Calculate market volatility and other metrics
        current_price = full_df['price'].iloc[-1]
        price_change_7d = ((current_price - full_df['price'].iloc[-8]) / full_df['price'].iloc[-8]) * 100 if len(full_df) >= 8 else 0
        current_volatility = full_df['volatility'].iloc[-1] * 100 if 'volatility' in full_df.columns else 0
        
        # Display current market conditions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=self.messages.CURRENT_PRICE_LABEL,
                value=f"${current_price:.4f}",
                delta=f"{price_change_7d:.2f}% (7d)"
            )
        
        with col2:
            st.metric(
                label=self.messages.MARKET_VOLATILITY_LABEL,
                value=f"{current_volatility:.2f}%",
                help=self.messages.MARKET_VOLATILITY_HELP
            )
        
        with col3:
            st.metric(
                label=self.messages.MODEL_ACCURACY_LABEL,
                value=f"{accuracy:.1%}",
                help=self.messages.MODEL_ACCURACY_HELP
            )
        
        # Display strategy mode and parameters
        mode_emoji = "🔄" if strategy_params['mode'] == 'Dynamic' else "🎯"
        st.info(self.messages.STRATEGY_MODE_ACTIVE.format(
            mode_emoji=mode_emoji, 
            mode=strategy_params['mode']
        ), icon="ℹ️")
        
        # Calculate actual profit/loss targets
        if strategy_params['mode'] == 'Dynamic':
            if current_volatility > 0:
                tp_percentage = current_volatility * strategy_params['tp_multiplier']
                sl_percentage = current_volatility * strategy_params['sl_multiplier']
                tp_price = current_price * (1 + tp_percentage / 100)
                sl_price = current_price * (1 - sl_percentage / 100)
            else:
                tp_percentage = sl_percentage = 0
                tp_price = sl_price = current_price
        else:
            tp_percentage = strategy_params['take_profit_pct']
            sl_percentage = strategy_params['stop_loss_pct']
            tp_price = current_price * (1 + tp_percentage / 100)
            sl_price = current_price * (1 - sl_percentage / 100)
        
        # Enhanced prediction display
        st.markdown(f"### {self.messages.AI_SIGNAL_TITLE}")
        
        prediction_text = self.messages.PREDICTION_MAP.get(prediction, "Unknown")
        
        # Calculate risk/reward ratio
        risk_reward = tp_percentage / sl_percentage if sl_percentage != 0 else 0
        
        if prediction == 1.0:  # Take Profit
            st.success(f"**{prediction_text}**")
            st.info(self.messages.BULLISH_SIGNAL.format(
                volatility=current_volatility,
                tp_price=tp_price,
                tp_percentage=tp_percentage,
                sl_price=sl_price,
                sl_percentage=sl_percentage,
                time_barrier_days=strategy_params['time_barrier_days'],
                risk_reward=risk_reward
            ))
        
        elif prediction == -1.0:  # Stop Loss
            st.error(f"**{prediction_text}**")
            st.warning(self.messages.BEARISH_SIGNAL.format(
                volatility=current_volatility,
                sl_price=sl_price,
                sl_percentage=sl_percentage,
                tp_price=tp_price,
                tp_percentage=tp_percentage,
                time_barrier_days=strategy_params['time_barrier_days'],
                risk_reward=risk_reward
            ))
        
        else:  # Time Limit
            st.info(f"**{prediction_text}**")
            st.info(self.messages.NEUTRAL_SIGNAL.format(
                volatility=current_volatility,
                tp_price=tp_price,
                tp_percentage=tp_percentage,
                sl_price=sl_price,
                sl_percentage=sl_percentage,
                time_barrier_days=strategy_params['time_barrier_days']
            ))
        
        # Display model confidence
        st.markdown(f"### {self.messages.MODEL_CONFIDENCE_TITLE}")
        if probabilities:
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            
            for i, (class_val, prob) in enumerate(probabilities.items()):
                label = self.messages.PREDICTION_MAP.get(class_val, f"Class {class_val}")
                
                if i == 0:
                    with conf_col1:
                        st.metric(
                            label=label,
                            value=f"{prob:.1%}",
                            help=self.messages.CONFIDENCE_HELP
                        )
                        st.progress(prob)
                elif i == 1:
                    with conf_col2:
                        st.metric(
                            label=label,
                            value=f"{prob:.1%}",
                            help=self.messages.CONFIDENCE_HELP
                        )
                        st.progress(prob)
                else:
                    with conf_col3:
                        st.metric(
                            label=label,
                            value=f"{prob:.1%}",
                            help=self.messages.CONFIDENCE_HELP
                        )
                        st.progress(prob)
        
        # Add risk disclaimer
        st.error(self.messages.TRADING_RISK_DISCLAIMER)

    def display_model_performance(self, report):
        st.subheader(self.messages.MODEL_PERFORMANCE_TITLE)
        if report is None: 
            return
        
        df_report = pd.DataFrame(report).transpose()
        df_report.index = df_report.index.astype(str)
        df_report.rename(index=self.messages.SIGNAL_CLASS_LABELS, inplace=True)
        
        # Display metrics in a more user-friendly way
        st.markdown(f"### {self.messages.PERFORMANCE_METRICS_TITLE}")
        
        # Create columns for better layout
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.dataframe(df_report.round(3), use_container_width=True)
        
        with perf_col2:
            st.markdown(self.messages.METRICS_EXPLANATION)
        
        # Add overall performance summary
        if 'accuracy' in df_report.index:
            overall_accuracy = df_report.loc['accuracy', 'precision']
            st.success(self.messages.OVERALL_ACCURACY_MSG.format(accuracy=overall_accuracy))
        
        st.caption(self.messages.PERFORMANCE_NOTE)

    def display_error_visualization(self, main_df, test_results_df):
        st.subheader(self.messages.ERROR_VISUALIZATION_TITLE)
        if test_results_df is None or test_results_df.empty:
            st.warning(self.messages.NO_TEST_DATA_MSG)
            return
        
        # Create more detailed visualization
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=main_df.index, 
            y=main_df['price'], 
            mode='lines', 
            name=self.messages.CHART_PRICE_LABEL, 
            line=dict(color='deepskyblue', width=2)
        ))
        
        # Add prediction markers with better styling
        correct_buys = test_results_df[(test_results_df['predicted'] == 1) & (test_results_df['actual'] == 1)]
        if not correct_buys.empty:
            fig.add_trace(go.Scatter(
                x=correct_buys.index, 
                y=main_df.loc[correct_buys.index]['price'], 
                mode='markers', 
                name=self.messages.CORRECT_BUY_LABEL, 
                marker=dict(color='limegreen', size=12, symbol='triangle-up'),
                hovertemplate=self.messages.CORRECT_BUY_HOVER
            ))
        
        incorrect_buys = test_results_df[(test_results_df['predicted'] == 1) & (test_results_df['actual'] != 1)]
        if not incorrect_buys.empty:
            fig.add_trace(go.Scatter(
                x=incorrect_buys.index, 
                y=main_df.loc[incorrect_buys.index]['price'], 
                mode='markers', 
                name=self.messages.INCORRECT_BUY_LABEL, 
                marker=dict(color='yellow', size=12, symbol='triangle-up', line=dict(width=2, color='black')),
                hovertemplate=self.messages.INCORRECT_BUY_HOVER
            ))
        
        correct_sells = test_results_df[(test_results_df['predicted'] == -1) & (test_results_df['actual'] == -1)]
        if not correct_sells.empty:
            fig.add_trace(go.Scatter(
                x=correct_sells.index, 
                y=main_df.loc[correct_sells.index]['price'], 
                mode='markers', 
                name=self.messages.CORRECT_SELL_LABEL, 
                marker=dict(color='red', size=12, symbol='triangle-down'),
                hovertemplate=self.messages.CORRECT_SELL_HOVER
            ))
        
        incorrect_sells = test_results_df[(test_results_df['predicted'] == -1) & (test_results_df['actual'] != -1)]
        if not incorrect_sells.empty:
            fig.add_trace(go.Scatter(
                x=incorrect_sells.index, 
                y=main_df.loc[incorrect_sells.index]['price'], 
                mode='markers', 
                name=self.messages.INCORRECT_SELL_LABEL, 
                marker=dict(color='orange', size=12, symbol='triangle-down', line=dict(width=2, color='black')),
                hovertemplate=self.messages.INCORRECT_SELL_HOVER
            ))
        
        # Update layout with better styling
        fig.update_layout(
            title=self.messages.CHART_TITLE,
            xaxis_title=self.messages.CHART_DATE_LABEL,
            yaxis_title=self.messages.CHART_PRICE_AXIS,
            template="plotly_dark", 
            height=600, 
            margin=dict(t=50, b=50, l=50, r=50), 
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics
        total_predictions = len(test_results_df)
        correct_predictions = (test_results_df['predicted'] == test_results_df['actual']).sum()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        st.info(self.messages.TEST_DATA_SUMMARY.format(
            correct=correct_predictions,
            total=total_predictions,
            accuracy=accuracy
        ))