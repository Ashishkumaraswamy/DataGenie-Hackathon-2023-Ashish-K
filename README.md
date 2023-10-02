


<div align="center">
<h1> DATAGENIE HACKATHON 2023
</h1>

<p>
  This hack is an innovative approach towards automating a critical part of any Machine Learning problem of model selection from the existing models for the given data. In this project, I have developed a solution to create an efficient time series model selection algorithm
</p>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![Flask](https://img.shields.io/static/v1?style=for-the-badge&message=Flask&color=000000&logo=Flask&logoColor=FFFFFF&label=)
![Amazon AWS](https://img.shields.io/static/v1?style=for-the-badge&message=Amazon+AWS&color=232F3E&logo=Amazon+AWS&logoColor=FFFFFF&label=)
![Streamlit](https://img.shields.io/static/v1?style=for-the-badge&message=Streamlit&color=FF4B4B&logo=Streamlit&logoColor=FFFFFF&label=)

<hr>
</div>

**Website Link**: [AQI Heatwave Forecaster Application](https://nasscom-capgemini-hackathon-aqi-heatwave-forecaster-home-o0tty0.streamlit.app/)

![image](https://github.com/Ashishkumaraswamy/DataGenie-Hackathon-2023-Ashish-K/assets/64360092/53dc1611-e1ac-46df-b361-7e9a2f536454)


## Overview

This project was built for the DataGenie Hackathon 2023. This project is my approach to a beautiful problem statement. When a user gives time series data of any frequency, the application must have an ML-powered pre-trained classification algorithm that predicts which Time Series Forecasting model would be the best for the given data without fitting and trying out each time series model on the dataset. Post that we built a time series model to fit the given data which was predicted by the classification model and gave the forecast for the data as well as the MAPE score.

## Architecture Diagram Workflow

1. Classification Model Construction
   
![image](https://github.com/Ashishkumaraswamy/DataGenie-Hackathon-2023-Ashish-K/assets/64360092/6d133d97-a915-4a3a-a969-36ac974a82f7)

  The most important part of any machine learning problem is data. The performance of the ML model purely depends on the power of the data available. With this in mind, I started to generate data for the classification model. So my approach towards data generation was to generate synthetic datasets for each of the time series models I considered. So I wrote code to generate 5000 time series data for each time series model in such a random way that for these time series data, the corresponding model would be the best fit. Once after generating these datasets I generated a feature extraction pipeline and converted each and every time series dataset to a feature vector of 96 features each of which corresponded to the time series characteristic of the dataset. With these feature spaces, I built an LGBM tree-based classification model to predict the time series models and pickled and stored this trained model.

2. Time Series Prediction

![image](https://github.com/Ashishkumaraswamy/DataGenie-Hackathon-2023-Ashish-K/assets/64360092/36b5efaa-8dce-4a1e-91c9-41e897499a43)

  Now when a custom time series data is given I first extract the features for that dataset and predict using a pickled pre-trained classifier and identify the best-fit model for that data. Now with that time series model, I fit the dataset to predict the required time range and evaluate the results using the MAPE value.


## Technologies Used

1. Backend: FAST API
2. Frontend: Streamlit
3. Cloud: AWS
