import streamlit as st
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="Stock Price Analyzer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
        background: linear-gradient(135deg, #ffffff 0%, #87CEEB 100%);
    }
    .title-text {
        font-size: 3.5rem !important;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #0f4c81, #2c7fb8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1.5rem 0;
        margin-bottom: 0.5rem;
    }
    .subtitle-text {
        font-size: 1.5rem !important;
        text-align: center;
        color: #4a4a4a;
        font-weight: 400;
        margin-bottom: 3rem;
    }
    .feature-box {
        padding: 2rem;
        border-radius: 15px;
        border: none;
        margin: 1rem;
        text-align: center;
        background: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .feature-box:hover {
        transform: translateY(-5px);
    }
    .feature-title {
        font-size: 1.5rem !important;
        font-weight: 600;
        color: #0f4c81;
        margin-bottom: 1rem;
    }
    .feature-desc {
        font-size: 1rem;
        color: #666;
        line-height: 1.6;
    }
    .company-button {
        background-color: white;
        color: #0f4c81;
        border: 2px solid #0f4c81;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    .company-button:hover {
        background-color: #0f4c81;
        color: white;
    }
    .section-header {
        color: #0f4c81;
        font-size: 2rem !important;
        font-weight: 600;
        margin: 2rem 0;
        padding-left: 1rem;
        border-left: 5px solid #0f4c81;
    }
    .how-to-use {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin: 2rem 0;
    }
    .footer {
        background: #0f4c81;
        color: white !important;
        padding: 2rem;
        border-radius: 15px 15px 0 0;
        margin-top: 3rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title Section
st.markdown('<p class="title-text">Tech Stock Price Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Analyze and predict stock prices of top 10 tech companies</p>', unsafe_allow_html=True)

# Featured Companies Section
st.markdown('<h2 class="section-header">Featured Companies</h2>', unsafe_allow_html=True)
cols = st.columns(5)
companies = {
    "Meta": "META",
    "Apple": "AAPL",
    "Amazon": "AMZN",
    "Netflix": "NFLX",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "AMD": "AMD",
    "Intel": "INTC",
    "Salesforce": "CRM"
}

for idx, (company, symbol) in enumerate(companies.items()):
    with cols[idx % 5]:
        st.markdown(f"""
            <button class="company-button">
                {company} ({symbol})
            </button>
        """, unsafe_allow_html=True)

# Features Section
st.markdown('<h2 class="section-header">Our Features</h2>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <p class="feature-title">📊 Dashboard</p>
        <p class="feature-desc">Interactive dashboard showing real-time stock data, technical indicators, and company information.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <p class="feature-title">🔮 Stock Predictor</p>
        <p class="feature-desc">Advanced machine learning models to predict future stock prices and trends.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <p class="feature-title">📈 Insights</p>
        <p class="feature-desc">Detailed analysis and insights about stock performance and market trends.</p>
    </div>
    """, unsafe_allow_html=True)

# How to Use Section
st.markdown('<h2 class="section-header">How to Use</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="how-to-use">
    <ol style="list-style-type: none; padding: 0;">
        <li style="margin: 1rem 0; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <strong style="color: #0f4c81;">1. Navigate</strong> - Use the sidebar menu to access different features
        </li>
        <li style="margin: 1rem 0; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <strong style="color: #0f4c81;">2. Select</strong> - Choose a stock from our curated list of top tech companies
        </li>
        <li style="margin: 1rem 0; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <strong style="color: #0f4c81;">3. Analyze</strong> - Explore real-time data and predictions
        </li>
        <li style="margin: 1rem 0; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <strong style="color: #0f4c81;">4. Track</strong> - Monitor your favorite stocks and get insights
        </li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">Built with ❤️ using Streamlit and Python</p>
    <p style="font-size: 1rem; opacity: 0.8;">Data provided by Yahoo Finance</p>
</div>
""", unsafe_allow_html=True) 