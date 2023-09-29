from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from feature_extraction_pipeline import FeatureExtractionTransformer
from typing import List
import pandas as pd

app = FastAPI()

# Load the trained pipeline
pipeline = joblib.load('feature_extractor_pipeline.pkl')

class TimeSeriesRow(BaseModel):
    Date: str
    Value: float

class TimeSeriesData(BaseModel):
    data: List[TimeSeriesRow]

@app.post("/data/transform/")
def feature_extraction(data: TimeSeriesData):
    try:
        # Convert the list of dictionaries to a DataFrame
        time_series_data = data.data
        # Extracting date and value attributes and creating a list of tuples
        data_tuples = [(entry.Date, entry.Value) for entry in time_series_data]
        df_data = pd.DataFrame(data_tuples, columns=['Date', 'Value'])
        df_data.set_index('Date',inplace=True)        
        print(df_data.head())
        # Perform the transformation using the pipeline
        result = pipeline.transform(df_data)
        
        return result
    except Exception as e:
        return {"error": str(e)}