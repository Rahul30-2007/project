import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

class Visualizations:
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def create_price_chart(self, data, ticker):
        """
        Create interactive price chart with technical indicators
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            ticker (str): Stock ticker symbol
            
        Returns:
            plotly.graph_objects.Figure: Interactive price chart
        """
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{ticker} Price Chart', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color=self.colors['success'],
                    decreasing_line_color=self.colors['danger']
                ),
                row=1, col=1
            )
            
            # Moving averages
            if 'SMA' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA'],
                        mode='lines',
                        name='SMA (20)',
                        line=dict(color=self.colors['primary'], width=2)
                    ),
                    row=1, col=1
                )
            
            if 'EMA' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['EMA'],
                        mode='lines',
                        name='EMA (20)',
                        line=dict(color=self.colors['secondary'], width=2)
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color=self.colors['info'], width=1, dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color=self.colors['info'], width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(23, 162, 184, 0.1)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=self.colors['light'],
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'{ticker} Stock Analysis',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                xaxis2_title='Date',
                yaxis2_title='Volume',
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(rangeslider_visible=False)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating price chart: {str(e)}")
            return go.Figure()
    
    def create_rsi_chart(self, data):
        """
        Create RSI chart
        
        Args:
            data (pd.DataFrame): Stock data with RSI
            
        Returns:
            plotly.graph_objects.Figure: RSI chart
        """
        try:
            fig = go.Figure()
            
            # RSI line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color=self.colors['primary'], width=2)
                )
            )
            
            # Overbought and oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color=self.colors['danger'], annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color=self.colors['success'], annotation_text="Oversold (30)")
            fig.add_hline(y=50, line_dash="dot", line_color=self.colors['dark'], annotation_text="Neutral (50)")
            
            # Shaded regions
            fig.add_hrect(y0=70, y1=100, fillcolor=self.colors['danger'], opacity=0.1, line_width=0)
            fig.add_hrect(y0=0, y1=30, fillcolor=self.colors['success'], opacity=0.1, line_width=0)
            
            fig.update_layout(
                title='RSI (Relative Strength Index)',
                xaxis_title='Date',
                yaxis_title='RSI',
                height=300,
                yaxis=dict(range=[0, 100]),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating RSI chart: {str(e)}")
            return go.Figure()
    
    def create_macd_chart(self, data):
        """
        Create MACD chart
        
        Args:
            data (pd.DataFrame): Stock data with MACD
            
        Returns:
            plotly.graph_objects.Figure: MACD chart
        """
        try:
            fig = go.Figure()
            
            # MACD line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color=self.colors['primary'], width=2)
                )
            )
            
            # Signal line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color=self.colors['secondary'], width=2)
                )
            )
            
            # Histogram
            colors = ['red' if x < 0 else 'green' for x in data['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['MACD_Histogram'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.7
                )
            )
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color=self.colors['dark'])
            
            fig.update_layout(
                title='MACD (Moving Average Convergence Divergence)',
                xaxis_title='Date',
                yaxis_title='MACD',
                height=300,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating MACD chart: {str(e)}")
            return go.Figure()
    
    def create_prediction_chart(self, historical_data, predictions, ticker):
        """
        Create prediction chart with confidence intervals
        
        Args:
            historical_data (pd.DataFrame): Historical stock data
            predictions (dict): Predictions with forecast, upper_ci, lower_ci
            ticker (str): Stock ticker symbol
            
        Returns:
            plotly.graph_objects.Figure: Prediction chart
        """
        try:
            fig = go.Figure()
            
            # Historical prices (last 60 days for context)
            recent_data = historical_data.tail(60)
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color=self.colors['primary'], width=2)
                )
            )
            
            # Add confidence intervals first (so they appear behind the forecast line)
            fig.add_trace(
                go.Scatter(
                    x=predictions['upper_ci'].index,
                    y=predictions['upper_ci'],
                    mode='lines',
                    name='Upper CI (95%)',
                    line=dict(color=self.colors['info'], width=1, dash='dash'),
                    showlegend=True
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=predictions['lower_ci'].index,
                    y=predictions['lower_ci'],
                    mode='lines',
                    name='Lower CI (95%)',
                    line=dict(color=self.colors['info'], width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(23, 162, 184, 0.2)',
                    showlegend=True
                )
            )
            
            # Forecast line
            fig.add_trace(
                go.Scatter(
                    x=predictions['forecast'].index,
                    y=predictions['forecast'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color=self.colors['secondary'], width=3),
                    marker=dict(size=8, color=self.colors['secondary'])
                )
            )
            
            # Add vertical line to separate historical and forecast
            last_date = historical_data.index[-1]
            fig.add_vline(
                x=last_date,
                line_dash="dash",
                line_color=self.colors['dark'],
                annotation_text="Forecast Start",
                annotation_position="top"
            )
            
            # Calculate price change for annotation
            last_price = historical_data['Close'].iloc[-1]
            first_forecast = predictions['forecast'].iloc[0]
            price_change = first_forecast - last_price
            price_change_pct = (price_change / last_price) * 100
            
            fig.update_layout(
                title=f'{ticker} Price Prediction - {len(predictions["forecast"])} Day Forecast',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500,
                showlegend=True,
                hovermode='x unified',
                annotations=[
                    dict(
                        x=0.02,
                        y=0.98,
                        xref='paper',
                        yref='paper',
                        text=f'Price Change: {price_change:+.2f} ({price_change_pct:+.1f}%)',
                        showarrow=False,
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='black',
                        borderwidth=1
                    )
                ]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating prediction chart: {str(e)}")
            return go.Figure()
    
    def create_correlation_heatmap(self, data):
        """
        Create correlation heatmap for technical indicators
        
        Args:
            data (pd.DataFrame): Data with technical indicators
            
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        try:
            # Select numeric columns for correlation
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            correlation_matrix = data[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title='Technical Indicators Correlation Matrix',
                height=400,
                width=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
            return go.Figure()
    
    def create_residual_plots(self, residuals):
        """
        Create residual analysis plots
        
        Args:
            residuals (pd.Series): Model residuals
            
        Returns:
            plotly.graph_objects.Figure: Residual plots
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Residuals vs Time', 'Residuals Distribution', 'Q-Q Plot', 'ACF of Residuals'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Residuals vs Time
            fig.add_trace(
                go.Scatter(
                    x=residuals.index,
                    y=residuals,
                    mode='lines+markers',
                    name='Residuals',
                    line=dict(color=self.colors['primary'])
                ),
                row=1, col=1
            )
            
            # Residuals histogram
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    name='Distribution',
                    marker_color=self.colors['secondary'],
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            # Q-Q plot (simplified)
            sorted_residuals = np.sort(residuals)
            normal_quantiles = np.random.normal(0, 1, len(sorted_residuals))
            normal_quantiles.sort()
            
            fig.add_trace(
                go.Scatter(
                    x=normal_quantiles,
                    y=sorted_residuals,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color=self.colors['success'])
                ),
                row=2, col=1
            )
            
            # ACF of residuals (simplified)
            from statsmodels.tsa.stattools import acf
            acf_values = acf(residuals, nlags=20)
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(acf_values))),
                    y=acf_values,
                    name='ACF',
                    marker_color=self.colors['info']
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Residual Analysis',
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating residual plots: {str(e)}")
            return go.Figure()