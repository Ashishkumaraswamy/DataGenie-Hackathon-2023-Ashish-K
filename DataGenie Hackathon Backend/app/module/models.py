import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from tbats import TBATS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime
from .utils import *
from dateutil import parser

def generate_arima_forecast(data, date_from, date_to, frequency, period):
    # Filter data based on the specified date range
    date_from_ts = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
    date_to_ts = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
    date_from_ts = pd.Timestamp(date_from_ts)
    date_to_ts = pd.Timestamp(date_to_ts)
    print('Date From',date_from_ts)
    print('Date To',date_to_ts)
    print("Data",data)
    training_data = data[:date_from_ts]
    print("Triaining Data",training_data)
    print("Frequency:",frequency)
    # Fit ARIMA model to the filtered data
    order = (2, 2, 2)  # Example order, you can optimize this based on your data
    model = ARIMA(training_data, order=order)
    model_fit = model.fit()
    
    # Calculate forecast period based on frequency
    if frequency == 'H':
        forecast_period = (date_to_ts - date_from_ts).total_seconds() / 3600 + period -1 # Difference in hours
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(hours=forecast_period)]
    elif frequency == 'D':
        forecast_period = (date_to_ts - date_from_ts).days + period -1  # Difference in days
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(days=forecast_period)]
    elif frequency == 'W':
        forecast_period = ((date_to_ts - date_from_ts).days + 1) // 7 + period -1 # Difference in weeks
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(weeks=forecast_period)]
    elif frequency == 'M':
        forecast_period = (date_to_ts.year - date_from_ts.year) * 12 + date_to_ts.month - date_from_ts.month + period # Difference in months
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(months=forecast_period)]
    elif frequency == 'Y':
        forecast_period = date_to_ts.year - date_from_ts.year + period -1  # Difference in years
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(years=forecast_period)]
    print("Forecast Period:", forecast_period)
    # Make predictions for the specified period
    forecast = model_fit.forecast(steps=int(forecast_period)+1)
    print("Forecast\n",forecast)
    # Calculate MAPE

    result_json = create_result_json(forecast,actual_values)
    mape = mean_absolute_percentage_error(actual_values, forecast)
    
    return result_json, mape


def generate_ets_forecast(data, date_from, date_to, frequency, period):
    date_from_ts = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
    date_to_ts = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
    date_from_ts = pd.Timestamp(date_from_ts)
    date_to_ts = pd.Timestamp(date_to_ts)
    print('Date From',date_from_ts)
    print('Date To',date_to_ts)
    print("Data",data)
    training_data = data[:date_from_ts]
    print("Triaining Data",training_data)
    print("Frequency:", frequency)
    
    # Fit ETS model to the filtered data
    model = ExponentialSmoothing(training_data, trend="add", seasonal="add", seasonal_periods=12)  # Example configuration, adjust as needed
    model_fit = model.fit()
    
    # Calculate forecast period based on frequency
    if frequency == 'H':
        forecast_period = (date_to_ts - date_from_ts).total_seconds() / 3600 + period -1 # Difference in hours
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(hours=forecast_period)]
    elif frequency == 'D':
        forecast_period = (date_to_ts - date_from_ts).days + period -1 # Difference in days
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(days=forecast_period)]
    elif frequency == 'W':
        forecast_period = ((date_to_ts - date_from_ts).days + 1) // 7 + period -1 # Difference in weeks
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(weeks=forecast_period)]
    elif frequency == 'M':
        forecast_period = (date_to_ts.year - date_from_ts.year) * 12 + date_to_ts.month - date_from_ts.month + period -1 # Difference in months
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(months=forecast_period)]
    elif frequency == 'Y':
        forecast_period = date_to_ts.year - date_from_ts.year + period -1 # Difference in years
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(years=forecast_period)]
    print("Forecast Period:", forecast_period)
    
    # Make predictions for the specified period
    forecast = model_fit.forecast(steps=int(forecast_period)+1)
    print("Forecast\n", forecast)
    
    # Calculate MAPE
    result_json = create_result_json(forecast, actual_values)
    mape = mean_absolute_percentage_error(actual_values, forecast)
    
    return result_json, mape

def generate_garch_forecast(data, date_from, date_to, frequency, period):
    # Filter data based on the specified date range
    date_from_ts = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
    date_to_ts = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
    date_from_ts = pd.Timestamp(date_from_ts)
    date_to_ts = pd.Timestamp(date_to_ts)
    print('Date From',date_from_ts)
    print('Date To',date_to_ts)
    print("Data",data)
    training_data = data[:date_from_ts]
    print("Frequency:", frequency)
    
    # Fit GARCH model to the filtered data
    model = arch_model(training_data, vol='Garch', p=1, q=1)  # Example configuration, adjust p and q based on your data
    model_fit = model.fit(disp='off')
    
    # Calculate forecast period based on frequency
    if frequency == 'H':
        forecast_period = (date_to_ts - date_from_ts).total_seconds() / 3600 + period -1 # Difference in hours
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(hours=forecast_period)]
    elif frequency == 'D':
        forecast_period = (date_to_ts - date_from_ts).days + period -1 # Difference in days
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(days=forecast_period)]
    elif frequency == 'W':
        forecast_period = ((date_to_ts - date_from_ts).days + 1) // 7 + period -1  # Difference in weeks
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(weeks=forecast_period)]
    elif frequency == 'M':
        forecast_period = (date_to_ts.year - date_from_ts.year) * 12 + date_to_ts.month - date_from_ts.month + period -1 # Difference in months
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(months=forecast_period)]
    elif frequency == 'Y':
        forecast_period = date_to_ts.year - date_from_ts.year + period -1  # Difference in years
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(years=forecast_period)]
    print("Forecast Period:", forecast_period)
    
    # Make predictions for the specified period
    last_volatility = model_fit.conditional_volatility.iloc[-1]
    forecast_volatility = np.sqrt(last_volatility * forecast_period)
    random_samples = np.random.normal(0, 1, int(forecast_period))  # Generate random samples
    forecast = random_samples * forecast_volatility
    
    # Create a date range for the forecast period
    forecast_index = pd.date_range(start=date_from_ts, periods=int(forecast_period), freq=frequency)
    forecast = pd.Series(data=forecast,index=forecast_index)
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(actual_values, forecast)
    result_json = create_result_json(forecast, actual_values)
    
    return result_json, mape

def generate_lstm_forecast(data, date_from, date_to, frequency, period):
    # Filter data based on the specified date range
    date_from_ts = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
    date_to_ts = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
    date_from_ts = pd.Timestamp(date_from_ts)
    date_to_ts = pd.Timestamp(date_to_ts)
    print('Date From',date_from_ts)
    print('Date To',date_to_ts)
    print("Data",data)
    training_data = data[:date_from_ts]
    print("Triaining Data",training_data)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(training_data.values.reshape(-1, 1))
    
    # Create sequences for LSTM model
    sequence_length = 10  # You can adjust the sequence length based on your data
    sequences = []
    targets = []
    for i in range(sequence_length, len(scaled_data)):
        sequences.append(scaled_data[i - sequence_length:i, 0])
        targets.append(scaled_data[i, 0])
    sequences, targets = np.array(sequences), np.array(targets)
    
    # Reshape the data for LSTM input (samples, time steps, features)
    sequences = np.reshape(sequences, (sequences.shape[0], sequences.shape[1], 1))
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequences.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Fit the model to the training data
    model.fit(sequences, targets, epochs=50, batch_size=32)
    
    # Prepare input data for forecasting
    inputs = scaled_data[len(scaled_data) - sequence_length:]
    inputs = inputs.reshape(1, -1)  # Reshape for LSTM input

    if frequency == 'H':
        forecast_period = (date_to_ts - date_from_ts).total_seconds() / 3600 + period -1 # Difference in hours
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(hours=forecast_period)]
    elif frequency == 'D':
        forecast_period = (date_to_ts - date_from_ts).days + period -1  # Difference in days
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(days=forecast_period)]
    elif frequency == 'W':
        forecast_period = ((date_to_ts - date_from_ts).days + 1) // 7 + period -1  # Difference in weeks
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(weeks=forecast_period)]
    elif frequency == 'M':
        forecast_period = (date_to_ts.year - date_from_ts.year) * 12 + date_to_ts.month - date_from_ts.month + period -1 # Difference in months
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(months=forecast_period)]
    elif frequency == 'Y':
        forecast_period = date_to_ts.year - date_from_ts.year + period -1  # Difference in years
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(years=forecast_period)]
    print("Forecast Period:", forecast_period)
    forecast_period+=1
    # Make predictions for the specified period
    forecast = []
    for _ in range(int(forecast_period)):
        predicted_value = model.predict(inputs)
        forecast.append(predicted_value[0, 0])
        inputs = np.roll(inputs, -1)  # Shift inputs
        inputs[0, -1] = predicted_value  # Add predicted value at the end
    
    # Inverse transform the forecasted data
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast_index = pd.date_range(start=date_from_ts, periods=int(forecast_period), freq=frequency)
    forecast = pd.Series(data=forecast.flatten(),index=forecast_index)
    forecast = forecast.astype('float')
    print(forecast)
    # Calculate MAPE
    print("Shape:",len(forecast),len(actual_values))
    result_json = create_result_json(forecast, actual_values)
    print("Created Json")
    mape = mean_absolute_percentage_error(actual_values, forecast)
    print()
    return result_json, mape

def generate_prophet_forecast(data, date_from, date_to, frequency, period):
    date_from_ts = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
    date_to_ts = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
    date_from_ts = pd.Timestamp(date_from_ts)
    date_to_ts = pd.Timestamp(date_to_ts)
    print('Date From',date_from_ts)
    print('Date To',date_to_ts)
    print("Data",data)
    training_data = data[:date_from_ts]
    print("Triaining Data",training_data)
    
    # Prepare the data for Prophet model
    df = pd.DataFrame({
        'ds': training_data.index,  # Timestamps
        'y': training_data.values   # Values to forecast
    })
    
    # Create and fit the Prophet model
    model = Prophet()
    model.fit(df)
    
    if frequency == 'H':
        forecast_period = (date_to_ts - date_from_ts).total_seconds() / 3600 + period + 1 # Difference in hours
        date_offset = pd.DateOffset(hours=forecast_period)
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(hours=forecast_period)]
    elif frequency == 'D':
        forecast_period = (date_to_ts - date_from_ts).days + period + 1  # Difference in days
        date_offset = pd.DateOffset(days=forecast_period)
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(days=forecast_period)]
    elif frequency == 'W':
        forecast_period = ((date_to_ts - date_from_ts).days + 1) // 7 + period +1  # Difference in weeks
        date_offset = pd.DateOffset(weeks=forecast_period)
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(weeks=forecast_period)]
    elif frequency == 'M':
        forecast_period = (date_to_ts.year - date_from_ts.year) * 12 + date_to_ts.month - date_from_ts.month + period + 1 # Difference in months
        date_offset = pd.DateOffset(months=forecast_period)
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(months=forecast_period)]
    elif frequency == 'Y':
        forecast_period = date_to_ts.year - date_from_ts.year + period + 1  # Difference in years
        date_offset = pd.DateOffset(years=forecast_period)
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(hours=forecast_period)]
    forecast_period+=1
    print("Forecast Period:", forecast_period)

    # Create a dataframe for the forecast period
    future = model.make_future_dataframe(periods=int(forecast_period), freq=frequency)  # Use 'D' for daily frequency, adjust as needed
    print(future)
    # Generate forecasts for the specified period
    forecast = model.predict(future)
    
    
    forecast_index = pd.date_range(start=date_from_ts, periods=int(forecast_period), freq=frequency)
    forecast = forecast.set_index('ds')

    forecast = forecast['yhat'][date_from_ts:date_from_ts + date_offset]
    print('Forecast',forecast)
    print('Actual',actual_values)

    # Calculate MAPE
    print("Shape:",len(forecast),len(actual_values))
    result_json = create_result_json(forecast, actual_values)
    print("Created Json")
    mape = mean_absolute_percentage_error(actual_values, forecast)
    
    return result_json, mape

def generate_sarimax_forecast(data, date_from, date_to, frequency, period):
    date_from_ts = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
    date_to_ts = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
    date_from_ts = pd.Timestamp(date_from_ts)
    date_to_ts = pd.Timestamp(date_to_ts)
    print('Date From',date_from_ts)
    print('Date To',date_to_ts)
    print("Data",data)
    training_data = data[:date_from_ts]
    print("Triaining Data",training_data)
    
    # Fit SARIMAX model to the filtered data
    order = (1, 1, 1)  # Example order, you can optimize this based on your data
    seasonal_order = (1, 1, 1, 12)  # Example seasonal order (s=12 for monthly data), adjust as needed
    model = SARIMAX(training_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # Calculate forecast period based on frequency
    if frequency == 'H':
        forecast_period = (date_to_ts - date_from_ts).total_seconds() / 3600 + period -1# Difference in hours
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(hours=forecast_period)]
    elif frequency == 'D':
        forecast_period = (date_to_ts - date_from_ts).days + period -1 # Difference in days
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(days=forecast_period)]
    elif frequency == 'W':
        forecast_period = ((date_to_ts - date_from_ts).days + 1) // 7 + period -1 # Difference in weeks
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(weeks=forecast_period)]
    elif frequency == 'M':
        forecast_period = (date_to_ts.year - date_from_ts.year) * 12 + date_to_ts.month - date_from_ts.month + period -1# Difference in months
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(months=forecast_period)]
    elif frequency == 'Y':
        forecast_period = date_to_ts.year - date_from_ts.year + period -1 # Difference in years
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(years=forecast_period)]
    print("Forecast Period:", forecast_period)
    
    # Make predictions for the specified period
    forecast = model_fit.get_forecast(steps=int(forecast_period)+1)
    print("Forecast\n", forecast.predicted_mean)
    
    # Calculate MAPE
    result_json = create_result_json(forecast.predicted_mean, actual_values)
    mape = mean_absolute_percentage_error(actual_values, forecast.predicted_mean)

    return result_json, mape



def generate_stl_forecast(data, date_from, date_to, frequency, period):
    date_from_ts = datetime.strptime(date_from, "%Y-%m-%d %H:%M:%S")
    date_to_ts = datetime.strptime(date_to, "%Y-%m-%d %H:%M:%S")
    date_from_ts = pd.Timestamp(date_from_ts)
    date_to_ts = pd.Timestamp(date_to_ts)
    print('Date From',date_from_ts)
    print('Date To',date_to_ts)
    print("Data",data)
    training_data = data[:date_from_ts]
    print("Triaining Data",training_data)
    # Perform STL decomposition on the filtered data
    decomposition_result = seasonal_decompose(training_data, model='additive', period=3)  # Example period, adjust as needed
    
    # Extract the trend, seasonal, and residual components
    trend = decomposition_result.trend.dropna()
    seasonal = decomposition_result.seasonal.dropna()
    residual = decomposition_result.resid.dropna()
    
    # Calculate forecast period based on frequency
    if frequency == 'H':
        forecast_period = (date_to_ts - date_from_ts).total_seconds() / 3600 + period -1 # Difference in hours
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(hours=forecast_period)]
    elif frequency == 'D':
        forecast_period = (date_to_ts - date_from_ts).days + period -1 # Difference in days
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(days=forecast_period)]
    elif frequency == 'W':
        forecast_period = ((date_to_ts - date_from_ts).days + 1) // 7 + period -1 # Difference in weeks
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(weeks=forecast_period)]
    elif frequency == 'M':
        forecast_period = (date_to_ts.year - date_from_ts.year) * 12 + date_to_ts.month - date_from_ts.month + period -1# Difference in months
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(months=forecast_period)]
    elif frequency == 'Y':
        forecast_period = date_to_ts.year - date_from_ts.year + period -1 # Difference in years
        actual_values = data.loc[date_from_ts:date_from_ts + pd.DateOffset(years=forecast_period)]
    print("Forecast Period:", forecast_period)
    
    # Make predictions for the specified period using the trend and seasonal components
    forecast = trend.append(seasonal).iloc[:int(forecast_period)+1]
    print("Forecast\n", forecast)
    
    # Calculate MAPE
      # Actual values for the specified date range
    result_json = create_result_json(forecast, actual_values)
    mape = mean_absolute_percentage_error(actual_values, forecast)
    
    return result_json, mape