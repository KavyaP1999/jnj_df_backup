# Databricks notebook source
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
import random
import traceback

# COMMAND ----------


# ###Old code which was using

# def ets_es(df1, forecast_length, forecast_period, alpha = 0.4, beta =0, gamma =0):
#     if forecast_period == 'M':
#         sp = 12
#     elif forecast_period == 'W':
#         sp = 52
#     elif forecast_period == 'D':
#         sp = 365
#     elif forecast_period == 'Q':
#         sp = 4
#     fh = forecast_length #[*range(1, forecast_length + 1, 1)]

#     df_temp2 = df1.copy()
#     df_temp2.reset_index(drop=True, inplace=True)
#     first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
#     df_temp2 = df_temp2[first_non_zero:]
#     df_temp2_len = df_temp2.shape[0]
#     if forecast_period == 'M':
#         if df_temp2_len > 12 and df_temp2_len <= 24:
#             sp = 6
#         elif df_temp2_len > 6 and df_temp2_len <= 12:
#             sp = 3
#         elif df_temp2_len <= 6:
#             sp = 1
#     if (df_temp2['historical_sale'] == 0).any():
#         df_temp2['historical_sale'] = df_temp2['historical_sale'] + 1
#         X = df_temp2['historical_sale']
#         X = X.astype(float)
        
#         try:
#             forecaster = ExponentialSmoothing(X, trend='add', damped_trend=True, seasonal='add',
#                                           initialization_method='estimated', use_boxcox=True, seasonal_periods=sp)
#             x = forecaster.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
#             y_pred = x.forecast(fh)
#             y_pred = y_pred - 1
#         except:
#             y_pred = np.zeros(forecast_length)
#     else:
#         X = df_temp2['historical_sale']
#         X = X.astype(float)
#         try:
#             forecaster = ExponentialSmoothing(X, trend='add', damped_trend=True, seasonal='add',
#                                               initialization_method='estimated', use_boxcox=True, seasonal_periods=sp)
#             x = forecaster.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
#             y_pred = x.forecast(fh)
#         except:
#             y_pred = np.zeros(forecast_length)
#     yhat = np.where(y_pred < 0, 0, y_pred)
    
#     return yhat

# #         print(e)
# #         traceback.print_exc()

# COMMAND ----------

# changed alpha=0.5, beta=0.3, gamma=0.4 values and also initialization_method = 'estimated' to 'heuristic'

def ets_es(df1, forecast_length, forecast_period, alpha, beta, gamma):
    if forecast_period == 'M':
        sp = 12
    elif forecast_period == 'W':
        sp = 52
    elif forecast_period == 'D':
        sp = 365
    elif forecast_period == 'Q':
        sp = 4
    fh = forecast_length #[*range(1, forecast_length + 1, 1)]

    df_temp2 = df1.copy()
    df_temp2.reset_index(drop=True, inplace=True)
    first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
    df_temp2 = df_temp2[first_non_zero:]
    df_temp2_len = df_temp2.shape[0]
    if forecast_period == 'M':
        if df_temp2_len > 12 and df_temp2_len <= 24:
            sp = 6
        elif df_temp2_len > 6 and df_temp2_len <= 12:
            sp = 3
        elif df_temp2_len <= 6:
            sp = 1
    if (df_temp2['historical_sale'] == 0).any():
        df_temp2['historical_sale'] = df_temp2['historical_sale'] + 1
        X = df_temp2['historical_sale']
        X = X.astype(float)
        
        try:
            forecaster = ExponentialSmoothing(X, trend='add', damped_trend=True, seasonal='add',
                                          initialization_method='estimated', use_boxcox=True, seasonal_periods=sp)
            #x = forecaster.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
            x = forecaster.fit()
            y_pred = x.forecast(fh)
            y_pred = y_pred - 1
        except:
            y_pred = np.zeros(forecast_length)
    else:
        X = df_temp2['historical_sale']
        X = X.astype(float)
        try:
            forecaster = ExponentialSmoothing(X, trend='add', damped_trend=True, seasonal='add',
                                              initialization_method='estimated', use_boxcox=True, seasonal_periods=sp)
            #x = forecaster.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
            x = forecaster.fit()
            y_pred = x.forecast(fh)
        except:
            y_pred = np.zeros(forecast_length)
    yhat = np.where(y_pred < 0, 0, y_pred)
    
    return yhat

#         print(e)
#         traceback.print_exc()
