import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os

# Page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("Stock Price Predictor")

# Define the stock symbols
stocks = {
    'Meta': 'META', 'Apple': 'AAPL', 'Amazon': 'AMZN',
    'Netflix': 'NFLX', 'Google': 'GOOGL', 'Microsoft': 'MSFT',
    'NVIDIA': 'NVDA', 'AMD': 'AMD', 'Intel': 'INTC',
    'Salesforce': 'CRM'
}

# Create sidebar for stock selection
selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
stock_symbol = stocks[selected_stock]

def load_predictions(symbol):
    prediction_file = f'data/predictions/{symbol}_predictions.json'
    if not os.path.exists(prediction_file):
        st.error(f"No predictions found for {symbol}. Please run model.py first.")
        return None
        
    with open(prediction_file, 'r') as f:
        data = json.load(f)
    return data

# Load predictions
predictions_data = load_predictions(stock_symbol)

if predictions_data:
    # Convert predictions to DataFrame
    pred_data = []
    for date, values in predictions_data['predictions'].items():
        pred_data.append({
            'Date': date,
            'Predicted Price': values['predicted'],
            'Lower Bound': values['lower_bound'],
            'Upper Bound': values['upper_bound']
        })
    
    pred_df = pd.DataFrame(pred_data)
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
    pred_df = pred_df.set_index('Date')
    
    # Display predictions table
    st.subheader("Predicted Prices")
    st.dataframe(pred_df.round(2))
    
    # ... rest of the display code (Model Information, Insights, etc.) ... 