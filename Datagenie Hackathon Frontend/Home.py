# Libraries
import streamlit as st
import pandas as pd
import plot_utils
from datetime import datetime
from dateutil import parser
import statsmodels.api as sm
import json
import requests
import pytz



# Config
error_message=""
st.set_page_config(page_title='Auto Forecastor',
                   page_icon=':bar_chart:', layout='wide')

st.markdown("<h1 style='text-align: center; color: #014b94 ;font-size:50px'>AUTO TIME SERIES FORECASTOR</h1><hr>",
            unsafe_allow_html=True)


def button_onlick(date_from,date_to,ts_col,val_col,frequency,period,df):
    global error_message
    date_from = pd.Timestamp(date_from,tz='UTC')
    date_to = pd.Timestamp(date_to,tz='UTC')
    if frequency =='Daily':
        offset = pd.DateOffset(days=period)
    if frequency =='Weekly':
        offset = pd.DateOffset(weeks=period)
    if frequency =='Hourly':
        offset = pd.DateOffset(hours=period)
    if frequency =='Monthly':
        offset = pd.DateOffset(month=period)
    if date_from=="" or date_to=="" or ts_col=="" or val_col=="" or frequency=="":
        error_message="All the inputs are required and must be filled"
        return error_message
    if len(df)<3:
        error_message="The time series data is too small to forecast add other data"
        return error_message 
    if date_from>=date_to:
        error_message="The end date cannot be greater than start date"
        return error_message
    if date_from<pd.Timestamp(df[ts_col].iloc[0],tz='UTC'):
        error_message="The start date cannot be before the given data"
        return error_message
    if (date_to + offset)>pd.Timestamp(df[ts_col].iloc[-1],tz='UTC'):
        error_message="The end date cannot be after the given data"
        return error_message
    if ts_col==val_col:
        error_message="Both Date Column and Value column are same"
        return error_message
    return 'Y'

def make_request(df,date_from,date_to,ts_col,val_col,freq):
    median = df[val_col].median()
    df[val_col].fillna(median,inplace=True)
    date_from = date_from.replace(" ", "%20")
    date_to = date_to.replace(" ", "%20")
    df.columns = ['Date','Value']
    formatted_data = {
        "data": df.to_dict(orient="records")
    }
    mapper = {"Daily":'D',"Hourly":"H","Weekly":"W","Monthly":"M"}
    freq = mapper[freq]
    json_string = json.dumps(formatted_data)
    request_url = f"http://localhost:80/predict?date_from={date_from}&date_to={date_to}&period={period}&frequency={freq}"
    result = requests.post(request_url,data=json_string)
    result = result.json()
    return result
    

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.subheader("Get your best time series forecast results in just a button click")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,index_col=0)
    df_data = df.copy()
    date_from = ""
    date_to = ""
    ts_col = ""
    val_col = ""
    freq = ""
    period = 0
    error_message = ""
    response = ""
    col1, col2 = st.columns([2,4]) 
    with col1:
        st.table(df.iloc[:5])
        st.table(df.iloc[-5:])
        st.write("Showing first 10 rows out of {} rows".format(len(df)))
    with col2:
        if error_message:
            st.write(error_message)
        sub_col1_1, sub_col1_2 = st.columns([2,2])
        with sub_col1_1:
            ts_col = st.selectbox("Pick the Time Series column", list(df.columns), placeholder="Choose an option")
        with sub_col1_2:
            val_col = st.selectbox("Pick the Value Series column", list(df.columns)[::-1], placeholder="Choose an option")
        sub_col2_1, sub_col2_2 = st.columns([2,2])
        with sub_col2_1:
           date_from = st.date_input("Forecast start date")
           date_from_copy = date_from
           parsed_date = datetime.strptime(str(date_from), "%Y-%m-%d")
           # Convert the datetime object to the desired format "%Y-%m-%d %H:%M:%S"
           date_from = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
        with sub_col2_2:
            date_to = st.date_input("Forecast end date")
            date_to_copy = date_to
            parsed_date = datetime.strptime(str(date_to), "%Y-%m-%d")
            # Convert the datetime object to the desired format "%Y-%m-%d %H:%M:%S"
            date_to = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
        sub_col3_1, sub_col3_2, sub_col3_3 = st.columns([2,2,2])
        with sub_col3_1:
            freq = st.selectbox("Select the time frequency", ['Daily','Weekly','Monthly','Hourly'], placeholder="Choose an option")
            if freq=='Hourly':
                start_time = st.time_input("Select Start Time")
                combined_datetime = datetime.combine(date_from_copy, start_time)
                date_from = combined_datetime.strftime("%Y-%m-%d %H:%M:%S")
                end_time = st.time_input("Select End Time")
                combined_datetime = datetime.combine(date_to_copy, end_time)
                date_to = combined_datetime.strftime("%Y-%m-%d %H:%M:%S")
        with sub_col3_3:
            period = st.number_input("Enter the period to forecast", 0, 10)

        st.markdown("<hr>",unsafe_allow_html=True)
        sub_col4_1, sub_col4_2, sub_col4_3 = st.columns([5,4,2])
        with sub_col4_2:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            if st.button("Start Forecast",type="primary"):
               check = button_onlick(date_from,date_to,ts_col,val_col,freq,period,df.copy())
               if check == 'Y':
                    response  = make_request(df,date_from,date_to,ts_col,val_col,freq)
                    df_data = df_data.set_index('point_timestamp')
                    forecast = pd.DataFrame(response['result'])
                    forecast_result = forecast.copy()
                    forecast.set_index('point_timestamp',inplace=True)
                    forecast = forecast['yhat']
               else:
                    st.markdown(f"<p style='color: red'>{check}</p>",unsafe_allow_html=True)
    if response:
        st.markdown("<hr><h2 style='text-align: center;font-size:50px'>FORECAST RESULTS</h2>",
            unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")
        col1,col2=st.columns([4,6])
        with col2:
            st.table(forecast_result[['point_timestamp','point_value','yhat']])
        with col1:
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.metric("MAPE",round(response['mape'],3),0)
            with subcol2:
                st.metric("Model Used",response['model'])

        fig = plot_utils.plot_forecasted(df_data,
                                            forecast)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>",unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;font-size:50px'>GRAPHS AND CHARTS</h2>",
            unsafe_allow_html=True)
    st.subheader("Some interesting charts that you would love to see...\n\n\n")
    st.write("")
    st.write("")
    st.write("Use the slider below to select the date range you want to analyze")
    st.write("")
    df_data = df_data.reset_index()
    start_date, end_date = st.select_slider('Pick Month Range', options=(df_data[ts_col]), value=(df_data[ts_col].iloc[0], df_data[ts_col].iloc[-1]))
    st.markdown("<hr>",unsafe_allow_html=True)
    ts_filtered = df_data[(df_data[ts_col] >=
                        start_date) & (df_data[ts_col] <= end_date)]
    st.write("")
    st.subheader("Time Series Data Stats")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric("Total Number of Data Points",len(ts_filtered),0)
    with col2:
        st.metric("Mean Value",ts_filtered[val_col].mean(),0)
    with col3:
        st.metric("Frequency",freq)

    st.subheader("Time Series Plot")
    fig1 = plot_utils.linechart_with_range_slider(ts_filtered[ts_col],
                                            ts_filtered[val_col], 'Value', 'Time Series data with Rangeslider for {}'.format(val_col))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("<hr>",unsafe_allow_html=True)
    st.subheader("ACF and PACF Plots")
    col1,col2 = st.columns(2)
    with col1:
        fig2 = plot_utils.pacf_plot(ts_filtered[val_col])
        st.pyplot(fig2)
    with col2:
        fig3 = plot_utils.pacf_plot(ts_filtered[val_col])
        st.pyplot(fig3)

    ts_val = pd.to_datetime(ts_filtered[ts_col])
    ts_data = pd.DataFrame(ts_filtered[val_col]).set_index(ts_val)

    st.subheader("Trend and Seasonality Plots")
    fig4 = plot_utils.trend_seasonality_plot(ts_filtered[ts_col],
                                            ts_filtered[val_col])
    st.plotly_chart(fig4, use_container_width=True)

 
    