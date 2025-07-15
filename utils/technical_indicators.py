import pandas as pd
import numpy as np
import streamlit as st

class TechnicalIndicators:
    def __init__(self):
        pass
    
    def calculate_sma(self, data, window=20):
        """
        Calculate Simple Moving Average
        
        Args:
            data (pd.Series): Price data
            window (int): Moving average window
            
        Returns:
            pd.Series: Simple Moving Average
        """
        try:
            return data.rolling(window=window).mean()
        except Exception as e:
            st.error(f"Error calculating SMA: {str(e)}")
            return pd.Series(index=data.index)
    
    def calculate_ema(self, data, window=20):
        """
        Calculate Exponential Moving Average
        
        Args:
            data (pd.Series): Price data
            window (int): Moving average window
            
        Returns:
            pd.Series: Exponential Moving Average
        """
        try:
            return data.ewm(span=window).mean()
        except Exception as e:
            st.error(f"Error calculating EMA: {str(e)}")
            return pd.Series(index=data.index)
    
    def calculate_rsi(self, data, window=14):
        """
        Calculate Relative Strength Index
        
        Args:
            data (pd.Series): Price data
            window (int): RSI window
            
        Returns:
            pd.Series: RSI values
        """
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            st.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=data.index)
    
    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data (pd.Series): Price data
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line EMA period
            
        Returns:
            dict: MACD line, signal line, and histogram
        """
        try:
            ema_fast = self.calculate_ema(data, fast_period)
            ema_slow = self.calculate_ema(data, slow_period)
            
            macd_line = ema_fast - ema_slow
            signal_line = self.calculate_ema(macd_line, signal_period)
            histogram = macd_line - signal_line
            
            return {
                'MACD': macd_line,
                'MACD_Signal': signal_line,
                'MACD_Histogram': histogram
            }
        except Exception as e:
            st.error(f"Error calculating MACD: {str(e)}")
            return {
                'MACD': pd.Series(index=data.index),
                'MACD_Signal': pd.Series(index=data.index),
                'MACD_Histogram': pd.Series(index=data.index)
            }
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """
        Calculate Bollinger Bands
        
        Args:
            data (pd.Series): Price data
            window (int): Moving average window
            num_std (float): Number of standard deviations
            
        Returns:
            dict: Upper band, middle band (SMA), and lower band
        """
        try:
            sma = self.calculate_sma(data, window)
            std = data.rolling(window=window).std()
            
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            
            return {
                'BB_Upper': upper_band,
                'BB_Middle': sma,
                'BB_Lower': lower_band
            }
        except Exception as e:
            st.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {
                'BB_Upper': pd.Series(index=data.index),
                'BB_Middle': pd.Series(index=data.index),
                'BB_Lower': pd.Series(index=data.index)
            }
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            k_period (int): %K period
            d_period (int): %D period
            
        Returns:
            dict: %K and %D values
        """
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'Stoch_K': k_percent,
                'Stoch_D': d_percent
            }
        except Exception as e:
            st.error(f"Error calculating Stochastic: {str(e)}")
            return {
                'Stoch_K': pd.Series(index=close.index),
                'Stoch_D': pd.Series(index=close.index)
            }
    
    def calculate_atr(self, high, low, close, window=14):
        """
        Calculate Average True Range
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            window (int): ATR window
            
        Returns:
            pd.Series: ATR values
        """
        try:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
            
            return atr
        except Exception as e:
            st.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(index=close.index)
    
    def calculate_williams_r(self, high, low, close, window=14):
        """
        Calculate Williams %R
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            window (int): Williams %R window
            
        Returns:
            pd.Series: Williams %R values
        """
        try:
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            
            return williams_r
        except Exception as e:
            st.error(f"Error calculating Williams %R: {str(e)}")
            return pd.Series(index=close.index)
    
    def calculate_momentum(self, data, window=10):
        """
        Calculate Price Momentum
        
        Args:
            data (pd.Series): Price data
            window (int): Momentum window
            
        Returns:
            pd.Series: Momentum values
        """
        try:
            momentum = data.diff(window)
            return momentum
        except Exception as e:
            st.error(f"Error calculating Momentum: {str(e)}")
            return pd.Series(index=data.index)
    
    def calculate_roc(self, data, window=10):
        """
        Calculate Rate of Change
        
        Args:
            data (pd.Series): Price data
            window (int): ROC window
            
        Returns:
            pd.Series: ROC values
        """
        try:
            roc = ((data - data.shift(window)) / data.shift(window)) * 100
            return roc
        except Exception as e:
            st.error(f"Error calculating ROC: {str(e)}")
            return pd.Series(index=data.index)
    
    def get_trading_signals(self, data):
        """
        Generate basic trading signals based on technical indicators
        
        Args:
            data (pd.DataFrame): Data with technical indicators
            
        Returns:
            pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        try:
            signals = pd.Series(0, index=data.index)
            
            # Simple moving average crossover
            if 'SMA' in data.columns:
                signals[(data['Close'] > data['SMA']) & (data['Close'].shift(1) <= data['SMA'].shift(1))] = 1
                signals[(data['Close'] < data['SMA']) & (data['Close'].shift(1) >= data['SMA'].shift(1))] = -1
            
            # RSI overbought/oversold
            if 'RSI' in data.columns:
                signals[(data['RSI'] < 30) & (data['RSI'].shift(1) >= 30)] = 1
                signals[(data['RSI'] > 70) & (data['RSI'].shift(1) <= 70)] = -1
            
            # MACD crossover
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd_crossover_up = (data['MACD'] > data['MACD_Signal']) & (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))
                macd_crossover_down = (data['MACD'] < data['MACD_Signal']) & (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1))
                
                signals[macd_crossover_up] = 1
                signals[macd_crossover_down] = -1
            
            return signals
            
        except Exception as e:
            st.error(f"Error generating trading signals: {str(e)}")
            return pd.Series(0, index=data.index)
