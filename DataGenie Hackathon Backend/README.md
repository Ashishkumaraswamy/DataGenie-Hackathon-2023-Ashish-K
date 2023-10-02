<div align="center">
<h1> Backend for AQI-Heatwave-Forecaster
</h1>

<p>
Basic Flask application to make model prediction of AQI and Heatwaves for the Tier-2 cites in Telangana with various endpoints as mentioned below.
</p>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![Flask](https://img.shields.io/static/v1?style=for-the-badge&message=Flask&color=000000&logo=Flask&logoColor=FFFFFF&label=)
![Amazon AWS](https://img.shields.io/static/v1?style=for-the-badge&message=Amazon+AWS&color=232F3E&logo=Amazon+AWS&logoColor=FFFFFF&label=)

<hr>
</div>


## Setting Up Application

1. Create and Activate the virtual env
```
    python3 -m venv venv
    venv\Scripts\activate.bat
```

2. Install the required python packages
```
    pip3 install -r requirements.txt
```

3. Run the Application

```
set FLASK_APP=index.py
set FLASK_ENV=development
flask run
```


## Sample request to our flask application

![Screen Recording - 3 March 2023 (1)](https://user-images.githubusercontent.com/62760269/222525992-b5b129df-dcb3-4315-b7b7-17ef88bdb451.gif)




**API-Endpoints**
----

* **URL**

 http:/ec2-65-2-179-122.ap-south-1.compute.amazonaws.com:80/data/transform
  
* **Description** 

  Runs the feature extraction pipeline for a given time series data and returns the features for the given time series dataset

* **Method:**

  `POST`  
  
*  **URL Params**
   
   None
   
* **Data Params**
   
   ```

   {
    "data": [
        {
            "Date": "2021-01-01",
            "Value": 4.370466050139905
        },
        {
            "Date": "2021-01-02",
            "Value": -5.806895275501942
        },
        {
            "Date": "2021-01-03",
            "Value": -9.984991910210416
        },
        {
            "Date": "2021-01-04",
            "Value": -0.6852342713793762
        }
    ]
}
    ```

* **Success Response:**
  
  * **Code:** 200 <br />
    **Content:** 
    ```json

    {
    "Mean": -3.5202193463724476e-17,
    "Median": -0.02275393954494072,
    "Mode": -3.153297362713809,
    "Variance": 0.9999999999999998,
    "Std Dev": 0.9999999999999999,
    "Skewness": -0.06439507844883287,
    "Kurtosis": 0.49304717187986036,
    "Linear Regression Slope": 0.014310287489994382,
    "25th Percentile": -0.6308659146302897,
    "75th Percentile": 0.6062092651014611,
    "Auto_Corr_Lag1": 0.34039891321081905,
    "Auto_Corr_Lag2": -0.05650784384377178,
    .
    .
    "poly_coefficients_1": 0.037975949394819936,
    "poly_coefficients_2": -0.000292168665491678,
    "residuals_normality_pvalue": 0.9637170781846063,
    "stationary": false,
    "significant_acf_peaks": 2,
    "significant_pacf_peaks": 3,
    "lag1_correlation": 0.34039891321081905
    }

    ```



* **URL**

 http://ec2-65-2-179-122.ap-south-1.compute.amazonaws.com:80/predict?date_from=2021-07-17%2000:00:00&date_to=2021-07-27%2000:00:00&period=1&frequency=D
  
* **Description**

  Returns the forecasted values for the given date range by fitting to the model that was predicted as the best model for this data by the classification model

* **Method:**

  `POST`  
  
*  **URL Params**

   ```
   date_from : Date From which you want to predict(Testing data start date)
   date_to : Date From which you want to predict(Testing data end date)
   period  : The time period for which you want to forecast
   frequency : The time frequency of the givem data (D: Daily, W: Weekly, H: hourly)
   model(optional) : model from which you want the prediction from(used for debugging)
   ```
   
* **Data Params**
  
  ```

    {
    "data": [
        {
            "Date": "2019-07-14",
            "Value": 6
        },
        {
            "Date": "2019-07-15",
            "Value": 7
        },
        {
            "Date": "2019-07-16",
            "Value": 6
        },
        .
        .
        .
        {
            "Date": "2021-07-26",
            "Value": 8
        },
        {
            "Date": "2021-07-27",
            "Value": 7
        }
    ]
}
    ```

* **Success Response:**
  
  * **Code:** 200 <br />
    **Content:** 
    ```json
       {
          "model":"SARIMAX"
          "mape":0.011571461543367502
          "result":[
            {
              "point_value":4355175
              "point_timestamp":"2021-09-01T00:00:00"
              "yhat":4316497.903961731
            },
            {
              "point_value":4365450
              "point_timestamp":"2021-09-02T00:00:00"
              "yhat":4306326.246557016
            },
            {
              "point_value":4369586
              "point_timestamp":"2021-09-03T00:00:00"
              "yhat":4318357.663612536
            },
            {
              "point_value":4393149
              "point_timestamp":"2021-09-04T00:00:00"
              "yhat":4319005.003038726
            },
            {
              "point_value":4367766
              "point_timestamp":"2021-09-05T00:00:00"
              "yhat":4344862.715408602
            }
            ,{
              "point_value":4345440
              "point_timestamp":"2021-09-06T00:00:00"
              "yhat":4355264.553396522
            }
            ,{
              "point_value":4386557
              "point_timestamp":"2021-09-07T00:00:00"
              "yhat":4361618.460152729
            }
            ,{
              "point_value":4395916
              "point_timestamp":"2021-09-08T00:00:00"
              "yhat":4346855.298490794
            }
            ,{
              "point_value":4397005
              "point_timestamp":"2021-09-09T00:00:00"
              "yhat":4340202.992874965
            },
            {
              "point_value":4410017
              "point_timestamp":"2021-09-10T00:00:00"
              "yhat":4342115.693806017
            },
            {
              "point_value":4434277
              "point_timestamp":"2021-09-11T00:00:00"
              "yhat":4340115.289953296
            },
            {
              "point_value":4397600
              "point_timestamp":"2021-09-12T00:00:00"
              "yhat":4357914.355241725
            },
            {
              "point_value":4378402
              "point_timestamp":"2021-09-13T00:00:00"
              "yhat":4363963.685213358
            }
        ]
    }
    ```
    
    
    
    
    
    
    

 


