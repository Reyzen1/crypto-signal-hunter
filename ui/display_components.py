# ui/display_components.py
"""
This module handles the display components for the Crypto Signal Hunter dashboard.
Extracted from dashboard.py for better code organization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

class DisplayComponents:
    def __init__(self):
        pass

    def display_ml_prediction(self, prediction, accuracy, probabilities, strategy_params):
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

    def display_model_performance(self, report):
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

    def display_error_visualization(self, main_df, test_results_df):
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