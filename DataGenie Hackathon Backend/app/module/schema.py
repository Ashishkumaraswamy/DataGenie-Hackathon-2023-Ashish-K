from pydantic import BaseModel
from typing import List

class TimeSeriesRow(BaseModel):
    Date: str
    Value: float

class TimeSeriesData(BaseModel):
    data: List[TimeSeriesRow]

class QueryParams(BaseModel):
    date_from: str
    date_to: str
    period: int
    frequency: str
    