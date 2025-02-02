import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Define the stock symbols
stocks = {
    'Meta': 'META',
    'Apple': 'AAPL',
    'Amazon': 'AMZN',
    'Netflix': 'NFLX',
    'Google': 'GOOGL',
    'Microsoft': 'MSFT',
    'NVIDIA': 'NVDA',
    'AMD': 'AMD',
    'Intel': 'INTC',
    'Salesforce': 'CRM'
}

# Set up the Streamlit page
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("Tech Stocks Dashboard")

# Create sidebar for stock selection
selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
stock_symbol = stocks[selected_stock]

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Fetch stock data
@st.cache_data
def load_data(symbol, start, end):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start, end=end)
    return data

# Get stock info separately (not cached)
def get_stock_info(symbol):
    stock = yf.Ticker(symbol)
    return stock.info

# Update the data loading
data = load_data(stock_symbol, start_date, end_date)
info = get_stock_info(stock_symbol)

# Display company info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
with col2:
    st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
with col3:
    st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Volume Analysis", "Returns Distribution", "Technical Indicators"])

with tab1:
    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'])])
    fig.update_layout(title=f"{selected_stock} Stock Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Volume chart
    fig = px.bar(data, x=data.index, y='Volume', title=f"{selected_stock} Trading Volume")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Daily returns distribution
    daily_returns = data['Close'].pct_change()
    fig = px.histogram(daily_returns, title=f"{selected_stock} Daily Returns Distribution")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Technical indicators
    # Calculate moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='20-day MA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='50-day MA'))
    fig.update_layout(title=f"{selected_stock} Technical Analysis", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# Display key statistics
st.subheader("Key Statistics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
with col2:
    st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")
with col3:
    st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
with col4:
    st.metric("Beta", f"{info.get('beta', 0):.2f}")

# Company description
st.subheader("Company Description")
st.write(info.get('longBusinessSummary', 'No description available')) 