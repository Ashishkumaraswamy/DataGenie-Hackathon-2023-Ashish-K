from pydantic import BaseModel
import joblib
from .feature_extraction_pipeline import *
from fastapi import APIRouter
import pandas as pd
from app import app
from .schema import *
from .models import *
from datetime import datetime
from dateutil import parser
import os
import lightgbm

@app.get("/")
def home():
    return "Hello World"

@app.post("/data/transform/")
def feature_extraction(data: TimeSeriesData):
    pipeline = joblib.load('feature_extractor_pipeline.pkl')
    try:
        # Convert the list of dictionaries to a DataFrame
        time_series_data = data.data
        # Extracting date and value attributes and creating a list of tuples
        data_tuples = [(entry.Date, entry.Value) for entry in time_series_data]
        df_data = pd.DataFrame(data_tuples, columns=['Date', 'Value'])
        df_data.set_index('Date',inplace=True)        
        # Perform the transformation using the pipeline
        result = pipeline.transform(df_data)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
async def process_data(data: TimeSeriesData, date_from: str,date_to: str="",period: int=0,frequency: str="",model: str=""):

    time_series_data = data.data
    if model == "":
        mapper = {0:'ARIMA',1:"ETS",2:"GARCH",3:"LSTM",4:"PROPHET",5:"SARIMAX",6:"STL"}

        loaded_model = joblib.load('lgbm_model_2.pkl')

        feature_data = feature_extraction(data)

        feature_data = np.array(feature_data)
        feature_data = feature_data.reshape(1,96)
        y_pred = loaded_model.predict(feature_data)
        y_pred_max = [max(y_pred[0])]
        # Convert predicted probabilities to class labels
        predicted_label = list(y_pred[0]).index(max(y_pred[0]))
        model = mapper[predicted_label]
    # Extracting date and value attributes and creating a list of tuples
    data_tuples = [(entry.Date, entry.Value) for entry in time_series_data]

    ts_data = pd.Series(data=[value for _, value in data_tuples], index=[parser.parse(date).strftime("%Y-%m-%dT%H:%M:%S") for date, _ in data_tuples])       
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # Call the function
    if model=="ARIMA":
        forecast_json, mape_value = generate_arima_forecast(ts_data, date_from, date_to, frequency, period)
    elif model == "ETS":
        forecast_json, mape_value = generate_ets_forecast(ts_data, date_from, date_to, frequency, period)
    elif model == "GARCH":
        forecast_json, mape_value = generate_garch_forecast(ts_data, date_from, date_to, frequency, period)
    elif model == "LSTM":
        forecast_json, mape_value = generate_lstm_forecast(ts_data, date_from, date_to, frequency, period)
    elif model == "PROPHET":
        forecast_json, mape_value = generate_prophet_forecast(ts_data, date_from, date_to, frequency, period)
    elif model == "SARIMAX":
        forecast_json, mape_value = generate_sarimax_forecast(ts_data, date_from, date_to, frequency, period)
    elif model == "STL":
        forecast_json, mape_value = generate_stl_forecast(ts_data, date_from, date_to, frequency, period)

    # Return a response
    return {"model":model,"mape": mape_value, "result": forecast_json}

