from pydantic import BaseModel
from typing import List

class TimeSeriesRow(BaseModel):
    Date: str
    Value: float

class TimeSeriesData(BaseModel):
    data: List[TimeSeriesRow]