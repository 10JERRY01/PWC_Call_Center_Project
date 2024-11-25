#Data Preprocessing
#Load Data: Read the dataset and perform basic inspection.

import pandas as pd
import datetime
# Load the dataset
df = pd.read_excel("Call-Center-Dataset.xlsx")

# Inspect the dataset
print(df.info())
print(df.head())

#Handle Missing Values:
#Drop rows with missing Speed of answer in seconds or AvgTalkDuration or Satisfaction rating.

# Drop rows with missing values
df.dropna(subset=['Speed of answer in seconds','AvgTalkDuration', 'Satisfaction rating'], inplace=True)

#Feature Engineering:
# Convert date and time
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
df['Day_of_Week'] = df['Date'].dt.day_name()

# Convert Resolved to binary
df['Is_Resolved'] = df['Resolved'].apply(lambda x: 1 if x == 'Y' else 0)

#Prepare Data for Models:
# Aggregate for time-series models
ts_data = df.groupby(['Date', 'Time'])['Call Id'].count().reset_index()
ts_data.columns = ['ds', 'hour', 'y']

# ML dataset
ml_data = df[['Time', 'Day_of_Week', 'Topic', 'Speed of answer in seconds', 'AvgTalkDuration', 'Satisfaction rating']]
ml_data = pd.get_dummies(ml_data, columns=['Day_of_Week', 'Topic'], drop_first=True)

#Model Training
#ARIMA
from statsmodels.tsa.arima.model import ARIMA

# Train ARIMA
model = ARIMA(ts_data['y'], order=(1, 1, 1))
arima_result = model.fit()

# Forecast
forecast_arima = arima_result.forecast(steps=24)

#Evaluate ARIMA: Use Mean Absolute Error (MAE).
from sklearn.metrics import mean_absolute_error

# Evaluate ARIMA
mae_arima = mean_absolute_error(ts_data['y'][-24:], forecast_arima)
print(f"ARIMA MAE: {mae_arima}")

# Save the model for deployment
import joblib
joblib.dump(arima_result, "arima_model.pkl")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt

# Load the saved ARIMA model
arima_model = joblib.load("arima_model.pkl")

# Streamlit app
st.title("ARIMA Model Forecasting")

# Upload data
uploaded_file = st.file_uploader("Upload your time-series CSV file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file
    data = pd.read_csv(uploaded_file)

    # Ensure date column exists
    if 'Date' not in data.columns:
        st.error("The uploaded file must contain a 'Date' column.")
    else:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        st.write("Uploaded Data Preview:")
        st.write(data.head())

        # Forecast horizon input
        forecast_periods = st.number_input(
            "Enter number of periods to forecast:", min_value=1, value=10, step=1
        )

        if st.button("Generate Forecast"):
            # Extend index for forecast periods
            forecast_index = pd.date_range(
                start=data.index[-1],
                periods=forecast_periods + 1,  # Include last observed date
                freq="D"
            )[1:]

            # Generate forecast
            forecast = arima_model.forecast(steps=forecast_periods)
            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=["Forecast"])

            # Combine actual and forecasted data
            full_data = pd.concat([data, forecast_df])

            # Display forecasted data
            st.write("Forecasted Values:")
            st.write(forecast_df)

            # Plot results
            plt.figure(figsize=(10, 5))
            plt.plot(data.index, data['Satisfaction rating'], label="Actual")
            plt.plot(forecast_df.index, forecast_df["Forecast"], label="Forecast", color="orange")
            plt.legend()
            plt.title("ARIMA Model Forecast")
            plt.xlabel("Date")
            plt.ylabel("Values")
            plt.grid()
            st.pyplot(plt)

