from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from feature_extraction_pipeline import FeatureExtractionTransformer
from typing import List

app = FastAPI()

# Load the trained pipeline
pipeline = joblib.load('feature_extractor_pipeline.pkl')

class TimeSeriesRow(BaseModel):
    date: str
    value: float

class TimeSeriesData(BaseModel):
    data: List[TimeSeriesRow]

@app.post("/data/transform/")
def feature_extraction(data: TimeSeriesData):
    try:
        # Convert the list of dictionaries to a DataFrame
        df_data = pd.DataFrame(data.data)
        df_data.set_index('Date',inplace=True)        
        
        # Perform the transformation using the pipeline
        result = pipeline.transform(df_data)
        
        # Return the transformed result
        return result
    except Exception as e:
        return {"error": str(e)}