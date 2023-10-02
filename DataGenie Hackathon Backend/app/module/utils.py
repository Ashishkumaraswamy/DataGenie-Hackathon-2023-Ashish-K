def create_result_json(forecast,actual):
    result = []
    for i in range(len(forecast)):
        result_json = {}
        result_json['point_value'] = actual.iloc[i]
        result_json['point_timestamp'] = forecast.index[i]
        result_json['yhat'] = forecast.iloc[i]
        result.append(result_json)
    return result
