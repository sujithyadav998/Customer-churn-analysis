import streamlit as st
import json
import pandas as pd
from datetime import datetime
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Insights", layout="wide")
st.title("Stock Insights Dashboard")

def load_all_predictions():
    predictions_dir = 'data/predictions'
    all_predictions = {}
    
    if not os.path.exists(predictions_dir):
        return all_predictions
    
    for file in os.listdir(predictions_dir):
        if file.endswith('_predictions.json'):
            symbol = file.replace('_predictions.json', '')
            with open(os.path.join(predictions_dir, file), 'r') as f:
                all_predictions[symbol] = json.load(f)
    
    return all_predictions

def create_summary_table(predictions):
    summary_data = []
    
    for symbol, data in predictions.items():
        current_price = data['current_price']
        pred_dates = list(data['predictions'].keys())
        last_pred = float(data['predictions'][pred_dates[-1]]['predicted'])
        
        # Calculate expected return
        expected_return = ((last_pred - current_price) / current_price) * 100
        
        # Determine recommendation
        if expected_return > 15:
            recommendation = "Strong Buy"
        elif expected_return > 5:
            recommendation = "Buy"
        elif expected_return > -5:
            recommendation = "Hold"
        else:
            recommendation = "Sell"
            
        summary_data.append({
            'Symbol': symbol,
            'Current Price': f"${current_price:.2f}",
            'Predicted Price (90d)': f"${last_pred:.2f}",
            'Expected Return': f"{expected_return:.1f}%",
            'Recommendation': recommendation,
            'Last Updated': data['timestamp']
        })
    
    return pd.DataFrame(summary_data)

# Load all predictions
all_predictions = load_all_predictions()

if not all_predictions:
    st.warning("No prediction data available. Please run predictions first.")
else:
    # Create summary table
    summary_df = create_summary_table(all_predictions)
    
    # Display summary table
    st.subheader("Stock Predictions Summary")
    st.dataframe(summary_df, use_container_width=True)
    
    # Create sections for different recommendations
    strong_buy = summary_df[summary_df['Recommendation'] == 'Strong Buy']
    buy = summary_df[summary_df['Recommendation'] == 'Buy']
    hold = summary_df[summary_df['Recommendation'] == 'Hold']
    sell = summary_df[summary_df['Recommendation'] == 'Sell']
    
    # Display recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strong Buy / Buy Recommendations")
        if len(strong_buy) + len(buy) > 0:
            combined_buy = pd.concat([strong_buy, buy])
            st.dataframe(combined_buy[['Symbol', 'Expected Return', 'Recommendation']])
        else:
            st.write("No buy recommendations at this time.")
    
    with col2:
        st.subheader("Hold / Sell Recommendations")
        if len(hold) + len(sell) > 0:
            combined_sell = pd.concat([hold, sell])
            st.dataframe(combined_sell[['Symbol', 'Expected Return', 'Recommendation']])
        else:
            st.write("No hold/sell recommendations at this time.")
    
    # Add detailed analysis for each stock
    st.subheader("Detailed Stock Analysis")
    selected_symbol = st.selectbox("Select Stock", list(all_predictions.keys()))
    
    if selected_symbol:
        stock_data = all_predictions[selected_symbol]
        
        # Create prediction chart
        dates = list(stock_data['predictions'].keys())
        predicted_prices = [data['predicted'] for data in stock_data['predictions'].values()]
        lower_bounds = [data['lower_bound'] for data in stock_data['predictions'].values()]
        upper_bounds = [data['upper_bound'] for data in stock_data['predictions'].values()]
        
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=dates,
            y=predicted_prices,
            name='Predicted Price',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=upper_bounds + lower_bounds[::-1],
            fill='toself',
            fillcolor='rgba(255,127,14,0.2)',
            line=dict(color='rgba(255,127,14,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title="insights")