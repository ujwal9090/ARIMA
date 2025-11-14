import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# ----------------------------------------------
# Stationarity check function
# ----------------------------------------------
def check_stationarity(series):
    result = adfuller(series.dropna())
    if result[1] < 0.05:
        return "The series is stationary"
    else:
        return "The series is NOT stationary"

# ----------------------------------------------
# Streamlit UI
# ----------------------------------------------
st.title("📈 Stock Price Forecast Dashboard")

# User Inputs
ticker = st.text_input("Enter Stock Ticker (Example: 7203.T)", "7203.T")

start = st.date_input("Start Date", dt.date(2024,1,1))
end = st.date_input("End Date", dt.date(2025,11,1))

steps = st.number_input("Forecast Days", min_value=5, max_value=60, value=10)

# ----------------------------------------------
# Process on button click
# ----------------------------------------------
if st.button("Generate Forecast"):
    
    # Download data
    st.subheader("Downloading stock data...")
    data = yf.download(ticker, start=start, end=end)
    
    if data.empty:
        st.error("No data found! Please check ticker name.")
    else:
        data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        data = data.set_index("Date")
        
        st.write("### Raw Data")
        st.dataframe(data.head())
        
        # Prepare Close & Returns
        df = data[['Close']]
        df['Return'] = df['Close'].pct_change()
        df = df.dropna()
        
        # Check Stationarity
        st.subheader("ADF Stationarity Test")
        stationarity_result = check_stationarity(df['Close'])
        st.write(stationarity_result)
        
        # Differencing
        df['Close_Diff'] = df['Close'].diff()
        df = df.dropna()
        
        # Stationarity after differencing
        st.write("After differencing:", check_stationarity(df['Close_Diff']))
        
        # ----------------------------------------------
        # Fit ARIMA Model
        # ----------------------------------------------
        st.subheader("Training ARIMA Model…")
        model = ARIMA(df['Close'], order=(5,1,0))
        model_fit = model.fit()
        
        # Forecast
        forecast = model_fit.forecast(steps=steps)
        dates = pd.date_range(start=df.index[-1], periods=steps+1, freq='B')[1:]
        
        # ----------------------------------------------
        # Plotting
        # ----------------------------------------------
        st.subheader("📉 Actual vs Predicted Close Prices")
        fig, ax = plt.subplots(figsize=(10,5))
        
        ax.plot(df['Close'], label="Actual Price")
        ax.plot(dates, forecast, label="Predicted Price", linestyle="dashed")
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
        
        st.success("Forecast generated successfully!")

