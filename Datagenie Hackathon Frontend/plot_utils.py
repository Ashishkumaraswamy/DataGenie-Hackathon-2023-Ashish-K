from plotly import graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import numpy as np
import pandas as pd
from pandas import json_normalize

def gauge_chart(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [None, 500]},
               'borderwidth': 2,
               'bordercolor': "gray",
               'bgcolor': "white",
               'steps': [
            {'range': [0, 50], 'color': "green"},
            {'range': [51, 100], 'color': "lightgreen"},
            {'range': [101, 200], 'color': "yellow"},
            {'range': [201, 300], 'color': "orange"},
            {'range': [301, 400], 'color': "red"},
            {'range': [401, 500], 'color': "darkred"}
        ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': value}
        },
        title={'text': "AQI levels"}))
    fig.update_layout(font={'color': "darkblue", 'family': "Arial"})

    return fig

def heat_map_chart(df, title):
    df_corr = df.corr()
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=df_corr.columns,
            y=df_corr.index,
            z=np.array(df_corr),
            text=df_corr.values,
            texttemplate='%{text:.2f}'
        )
    )
    fig.layout.update(title_text=title)
    return fig


def linechart_with_range_slider(X_axis, Y_axis, parameter_name, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_axis, y=Y_axis,
                  name=parameter_name, line_color="blue"))
    fig.layout.update(title_text=title, xaxis_rangeslider_visible=True)
    return fig


def bargraph(x_data, y_data1, y_data1_name, xaxis_title, yaxis_title, y_data2=None, y_data2_name=None):
    """
    """
    if y_data2 == None and y_data2_name == None:
        fig = go.Figure(data=[
            go.Bar(name=y_data1_name, x=x_data,
                   y=y_data1, marker_color='crimson'),
        ])
    else:
        fig = go.Figure(data=[
            go.Bar(name=y_data1_name, x=x_data,
                   y=y_data1, marker_color='crimson'),
            go.Bar(name=y_data2_name, x=x_data, y=y_data2, marker_color='navy')
        ])

    fig.update_layout(xaxis_title=xaxis_title,
                      yaxis_title=yaxis_title, barmode='group')
    return fig


def linegraph(x_data, y_data1, y_data1_name, xaxis_title, yaxis_title, y_data2=None, y_data2_name=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data1, name=y_data1_name))
    if y_data2_name != None:
        fig.add_trace(go.Scatter(x=x_data, y=y_data2,
                                 name=y_data2_name, marker_color='crimson'))
    fig.update_layout(
        xaxis_title=xaxis_title, yaxis_title=yaxis_title, font=dict(color="white"))
    return fig

def trend_seasonality_residual(ts_data,trend,seasonal,residual):
    print(trend)
    fig = go.Figure()

    # Plot Original
    fig.add_trace(go.Scatter(x=ts_data.index, y=trend, mode='lines', name='Trend'))
    fig.add_trace(go.Scatter(x=ts_data.index, y=seasonal, mode='lines', name='Seasonal'))
    fig.add_trace(go.Scatter(x=ts_data.index, y=residual, mode='lines', name='Residual'))

    # Update layout for subplot-like appearance
    fig.update_layout(
        xaxis_title="Time Stamp", yaxis_title="Value", font=dict(color="white"))

    return fig

def plot_forecasted(ts_data,forecast_values):
    ts_data_values = ts_data.values
    print(forecast_values.values)
    fig = go.Figure()

    # Add original time series data to the figure
    fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data_values.flatten(), mode='lines', name='Original', line=dict(color='blue')))

    # Add forecasted values to the figure
    fig.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values.values, mode='lines', name='Forecasted', line=dict(color='orange')))

    # Customize the layout
    fig.update_layout(title='Original vs Forecasted Time Series',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    legend=dict(x=0, y=1, traceorder='normal'))
    
    return fig
