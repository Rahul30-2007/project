import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from utils.data_processor import DataProcessor
from utils.model_trainer import ModelTrainer
from utils.technical_indicators import TechnicalIndicators
from utils.visualizations import Visualizations

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

def main():
    st.title("📈 Stock Price Predictor")
    st.markdown("Predict future stock trends using ARIMA/SARIMA models with technical indicators")
    
    # Initialize utility classes
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    technical_indicators = TechnicalIndicators()
    visualizations = Visualizations()
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Stock ticker input
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
        ).upper()
        
        # Date range selector
        st.subheader("📅 Date Range")
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365*2),
                max_value=datetime.now() - timedelta(days=1)
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now() - timedelta(days=1),
                max_value=datetime.now()
            )
        
        # Model parameters
        st.subheader("🤖 Model Parameters")
        forecast_days = st.slider(
            "Forecast Days",
            min_value=1,
            max_value=30,
            value=10,
            help="Number of days to forecast into the future"
        )
        
        model_type = st.selectbox(
            "Model Type",
            ["ARIMA", "SARIMA"],
            help="Choose between ARIMA and SARIMA models"
        )
        
        # Technical indicators selection
        st.subheader("📊 Technical Indicators")
        include_sma = st.checkbox("Simple Moving Average (SMA)", value=True)
        include_ema = st.checkbox("Exponential Moving Average (EMA)", value=True)
        include_rsi = st.checkbox("RSI", value=True)
        include_macd = st.checkbox("MACD", value=True)
        include_bollinger = st.checkbox("Bollinger Bands", value=True)
        
        # Fetch data button
        if st.button("📥 Fetch Data & Train Model", type="primary"):
            if not ticker:
                st.error("Please enter a stock ticker symbol")
                return
            
            if start_date >= end_date:
                st.error("Start date must be before end date")
                return
            
            # Fetch and process data
            with st.spinner("Fetching stock data..."):
                try:
                    data = data_processor.fetch_stock_data(ticker, start_date, end_date)
                    if data is None or data.empty:
                        st.error(f"No data found for ticker {ticker}")
                        return
                    
                    st.session_state.data = data
                    st.success(f"✅ Successfully fetched {len(data)} days of data for {ticker}")
                    
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
                    return
            
            # Calculate technical indicators
            with st.spinner("Calculating technical indicators..."):
                try:
                    indicators = {}
                    if include_sma:
                        indicators['SMA'] = technical_indicators.calculate_sma(data['Close'])
                    if include_ema:
                        indicators['EMA'] = technical_indicators.calculate_ema(data['Close'])
                    if include_rsi:
                        indicators['RSI'] = technical_indicators.calculate_rsi(data['Close'])
                    if include_macd:
                        macd_data = technical_indicators.calculate_macd(data['Close'])
                        indicators.update(macd_data)
                    if include_bollinger:
                        bollinger_data = technical_indicators.calculate_bollinger_bands(data['Close'])
                        indicators.update(bollinger_data)
                    
                    st.session_state.data = pd.concat([data, pd.DataFrame(indicators)], axis=1)
                    
                except Exception as e:
                    st.error(f"Error calculating technical indicators: {str(e)}")
                    return
            
            # Train model
            with st.spinner(f"Training {model_type} model..."):
                try:
                    if model_type == "ARIMA":
                        predictions, metrics = model_trainer.train_arima(
                            data['Close'], forecast_days
                        )
                    else:  # SARIMA
                        predictions, metrics = model_trainer.train_sarima(
                            data['Close'], forecast_days
                        )
                    
                    st.session_state.predictions = predictions
                    st.session_state.model_metrics = metrics
                    st.success(f"✅ {model_type} model trained successfully!")
                    
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    return
    
    # Main content area
    if st.session_state.data is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📊 {ticker} Stock Analysis")
            
            # Display basic statistics
            latest_price = st.session_state.data['Close'].iloc[-1]
            price_change = st.session_state.data['Close'].iloc[-1] - st.session_state.data['Close'].iloc[-2]
            price_change_pct = (price_change / st.session_state.data['Close'].iloc[-2]) * 100
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Latest Price", f"${latest_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
            with metric_col2:
                st.metric("High", f"${st.session_state.data['High'].max():.2f}")
            with metric_col3:
                st.metric("Low", f"${st.session_state.data['Low'].min():.2f}")
            with metric_col4:
                st.metric("Volume", f"{st.session_state.data['Volume'].mean():.0f}")
            
            # Historical price chart
            st.subheader("📈 Historical Price & Technical Indicators")
            price_chart = visualizations.create_price_chart(st.session_state.data, ticker)
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Technical indicators chart
            if any([include_rsi, include_macd]):
                st.subheader("📊 Technical Indicators")
                
                if include_rsi and 'RSI' in st.session_state.data.columns:
                    rsi_chart = visualizations.create_rsi_chart(st.session_state.data)
                    st.plotly_chart(rsi_chart, use_container_width=True)
                
                if include_macd and 'MACD' in st.session_state.data.columns:
                    macd_chart = visualizations.create_macd_chart(st.session_state.data)
                    st.plotly_chart(macd_chart, use_container_width=True)
        
        with col2:
            st.subheader("📋 Data Summary")
            
            # Display data info
            st.write(f"**Data Range:** {st.session_state.data.index[0].strftime('%Y-%m-%d')} to {st.session_state.data.index[-1].strftime('%Y-%m-%d')}")
            st.write(f"**Total Records:** {len(st.session_state.data)}")
            
            # Model metrics
            if st.session_state.model_metrics is not None:
                st.subheader("🎯 Model Performance")
                for metric, value in st.session_state.model_metrics.items():
                    st.metric(metric, f"{value:.4f}")
            
            # Predictions summary
            if st.session_state.predictions is not None:
                st.subheader("🔮 Forecast Summary")
                forecast_data = st.session_state.predictions['forecast']
                st.write(f"**Forecast Period:** {len(forecast_data)} days")
                st.write(f"**Predicted Price Range:** ${forecast_data.min():.2f} - ${forecast_data.max():.2f}")
                
                # Export predictions
                if st.button("📥 Export Predictions"):
                    csv = pd.DataFrame(st.session_state.predictions).to_csv()
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{ticker}_predictions.csv",
                        mime="text/csv"
                    )
            
            # Display recent data
            st.subheader("📊 Recent Data")
            st.dataframe(
                st.session_state.data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10),
                use_container_width=True
            )
    
    else:
        st.info("👆 Please configure the parameters in the sidebar and click 'Fetch Data & Train Model' to get started!")
        
        # Show sample usage
        st.subheader("🚀 How to Use")
        st.markdown("""
        1. **Enter a stock ticker** (e.g., AAPL, GOOGL, MSFT)
        2. **Select date range** for historical data
        3. **Choose model type** (ARIMA or SARIMA)
        4. **Select technical indicators** to include
        5. **Click 'Fetch Data & Train Model'** to analyze
        """)
        
        st.subheader("📚 About the Models")
        st.markdown("""
        **ARIMA (AutoRegressive Integrated Moving Average)**
        - Best for stationary time series
        - Captures trends and patterns in historical data
        - Suitable for short to medium-term forecasting
        
        **SARIMA (Seasonal ARIMA)**
        - Extension of ARIMA with seasonal components
        - Captures seasonal patterns in stock prices
        - Better for stocks with clear seasonal behavior
        """)

if __name__ == "__main__":
    main()
