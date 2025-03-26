# Databricks notebook source
from sktime.forecasting.naive import NaiveForecaster
import numpy as np


def naive_forecast(df1, forecast_length, forecast_period):
    if forecast_period == 'M':
        sp = 12
    elif forecast_period == 'W':
        sp = 52
    elif forecast_period == 'D':
        sp = 365
    elif forecast_period == 'Q':
        sp = 4
    fh = [*range(1, forecast_length+1, 1)]
    try:
        df_temp2 = df1.copy()
        df_temp2.reset_index(drop=True, inplace=True)
        first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
        df_temp2 = df_temp2[first_non_zero:]
        df_temp2_len = df_temp2.shape[0]
#         if forecast_period == 'M':
            
#             if df_temp2_len > 12 and df_temp2_len <= 24:
#                 sp = 6
#             elif df_temp2_len > 6 and df_temp2_len <= 12:
#                 sp = 3
#             elif df_temp2_len <= 6:
#                 sp = 1
        X = df_temp2['historical_sale']
        X = X.astype(np.float)
        forecaster = NaiveForecaster(sp=sp)
        x = forecaster.fit(X)
        y_pred = x.predict(fh)
        yhat = np.where(y_pred < 0, 0, y_pred)

    except:
        yhat = np.zeros(forecast_length)
    return yhat
