import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataProcessor:
    def __init__(self):
        pass
    
    def fetch_stock_data(self, ticker, start_date, end_date):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Create yfinance ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch historical data
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"No data available for ticker {ticker}")
                return None
            
            # Clean the data
            data = data.dropna()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                st.error(f"Missing required columns for ticker {ticker}")
                return None
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def validate_ticker(self, ticker):
        """
        Validate if a ticker symbol exists
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            bool: True if ticker is valid, False otherwise
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return 'symbol' in info or 'shortName' in info
        except:
            return False
    
    def prepare_data_for_modeling(self, data, target_column='Close'):
        """
        Prepare data for time series modeling
        
        Args:
            data (pd.DataFrame): Stock data
            target_column (str): Column to use for modeling
            
        Returns:
            pd.Series: Prepared time series data
        """
        try:
            # Extract target column
            ts_data = data[target_column].copy()
            
            # Remove any remaining NaN values
            ts_data = ts_data.dropna()
            
            # Ensure the index is datetime
            if not isinstance(ts_data.index, pd.DatetimeIndex):
                ts_data.index = pd.to_datetime(ts_data.index)
            
            return ts_data
            
        except Exception as e:
            st.error(f"Error preparing data for modeling: {str(e)}")
            return None
    
    def get_stock_info(self, ticker):
        """
        Get additional stock information
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD')
            }
            
        except Exception as e:
            st.warning(f"Could not fetch additional info for {ticker}: {str(e)}")
            return {'name': ticker, 'sector': 'Unknown', 'industry': 'Unknown', 'market_cap': 'N/A', 'currency': 'USD'}
    
    def calculate_returns(self, data, column='Close'):
        """
        Calculate daily returns
        
        Args:
            data (pd.DataFrame): Stock data
            column (str): Column to calculate returns for
            
        Returns:
            pd.Series: Daily returns
        """
        try:
            returns = data[column].pct_change().dropna()
            return returns
        except Exception as e:
            st.error(f"Error calculating returns: {str(e)}")
            return None
    
    def calculate_volatility(self, data, window=20, column='Close'):
        """
        Calculate rolling volatility
        
        Args:
            data (pd.DataFrame): Stock data
            window (int): Rolling window size
            column (str): Column to calculate volatility for
            
        Returns:
            pd.Series: Rolling volatility
        """
        try:
            returns = self.calculate_returns(data, column)
            volatility = returns.rolling(window=window).std()
            return volatility
        except Exception as e:
            st.error(f"Error calculating volatility: {str(e)}")
            return None
