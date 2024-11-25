import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime
# Load dataset
df = pd.read_excel("Call-Center-Dataset.xlsx")

# Inspect dataset
print(df.info())
print(df.describe())
print(df.head())

# Visualize missing data
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Data Heatmap")
plt.show()

# Fill missing values
df['Speed of answer in seconds'].fillna(df['Speed of answer in seconds'].mean(), inplace=True)
df.dropna(subset=['AvgTalkDuration', 'Satisfaction rating'], inplace=True)

# Convert data types
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour  # Extract hour

# Feature engineering
df['Day_of_Week'] = df['Date'].dt.day_name()
df['Is_Resolved'] = df['Resolved'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Agent', 'Topic', 'Day_of_Week'], drop_first=True)

#Exploratory Data Analysis (EDA)
#Understand trends and patterns in the data.
# Call volume by hour
plt.figure(figsize=(10, 5))
sns.countplot(x='Time', data=df, palette='viridis')
plt.title("Call Volume by Hour")
plt.show()

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Aggregate data by time
call_volume = df.groupby('Time')['Call Id'].count()

# Stationarity check
result = adfuller(call_volume)
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# Train ARIMA model
model = ARIMA(call_volume, order=(1, 1, 1))  # Adjust order after testing
arima_result = model.fit()
print(arima_result.summary())

from sklearn.metrics import mean_squared_error

# Example RMSE calculation for ARIMA
arima_forecast = arima_result.forecast(steps=24)
rmse_arima = mean_squared_error(call_volume[-24:], arima_forecast, squared=False)
print(f"ARIMA RMSE: {rmse_arima}")

# Define a function to predict call volume
def predict_call_volume(hour):
    forecast = arima_result.forecast(steps=1)
    return forecast[0]
import streamlit as st
# Streamlit UI
st.title("Call Volume Prediction Using ARIMA")

# User input: Hour of the day
hour = st.number_input("Enter Hour of the Day (0-23):", min_value=0, max_value=23)

# Predict and display results
if hour is not None:
    predicted_volume = predict_call_volume(hour)
    st.write(f"Predicted Call Volume for Hour {hour}: {predicted_volume}")

# Optionally, plot the historical data and forecast
st.subheader("Historical Call Volume")
plt.figure(figsize=(10, 6))
plt.plot(call_volume, label='Historical Call Volume')
plt.title('Call Volume by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Call Volume')
plt.legend()
st.pyplot()
