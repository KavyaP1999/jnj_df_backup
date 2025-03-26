# Databricks notebook source
import sktime
from sktime.forecasting.ets import AutoETS
import numpy as np
import pandas as pd
import traceback

# COMMAND ----------

# New_Code
def auto_ets(df1, forecast_length, forecast_period):
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
        if forecast_period == 'M':
            if df_temp2_len > 12 and df_temp2_len <= 24:
                sp = 6
            elif df_temp2_len > 6 and df_temp2_len <= 12:
                sp = 3
            elif df_temp2_len <= 6:
                sp = 1
        if (df_temp2['historical_sale'] == 0).any():
            df_temp2['historical_sale'] = df_temp2['historical_sale']+1
            X = df_temp2['historical_sale']
            X = X.astype(float)
            forecaster = AutoETS(auto=False, trend="add", seasonal="mul", damped_trend=True, sp=sp, initialization_method = 'heuristic',                   n_jobs=-1, maxiter=100, bounds={"smoothing_level":[0, 0.2], "smoothing_trend":[0, 0.5], "smoothing_seasonal":[0.4, 0.7]})
            x = forecaster.fit(X)
            y_pred = x.predict(fh)
            y_pred = y_pred - 1
        else:
            X = df_temp2['historical_sale']
            X = X.astype(float)
            forecaster = AutoETS(auto=False, trend="add", seasonal="mul", damped_trend=True, sp=sp, initialization_method = 'heuristic',                   n_jobs=-1, maxiter=100, bounds={"smoothing_level":[0, 0.2], "smoothing_trend":[0, 0.5], "smoothing_seasonal":[0.4, 0.7]})
            x = forecaster.fit(X)
            y_pred = x.predict(fh)
        yhat = np.where(y_pred < 0, 0, y_pred)
    except Exception as e:
        # print(e)
        # traceback.print_exc()
        yhat = np.zeros(forecast_length)
    return yhat

# COMMAND ----------

# #Older_version_2.2.10
# def auto_ets(df1, forecast_length, forecast_period):
#     if forecast_period == 'M':
#         sp = 12
#     elif forecast_period == 'W':
#         sp = 52
#     elif forecast_period == 'D':
#         sp = 365
#     elif forecast_period == 'Q':
#         sp = 4
#     fh = [*range(1, forecast_length+1, 1)]
#     try:
#         df_temp2 = df1.copy()
#         df_temp2.reset_index(drop=True, inplace=True)
#         first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
#         df_temp2 = df_temp2[first_non_zero:]
#         df_temp2_len = df_temp2.shape[0]
#         if forecast_period == 'M':
#             if df_temp2_len > 12 and df_temp2_len <= 24:
#                 sp = 6
#             elif df_temp2_len > 6 and df_temp2_len <= 12:
#                 sp = 3
#             elif df_temp2_len <= 6:
#                 sp = 1
#         if (df_temp2['historical_sale'] == 0).any():
#             df_temp2['historical_sale'] = df_temp2['historical_sale']+1
#             X = df_temp2['historical_sale']
#             X = X.astype(float)
#             forecaster = AutoETS(auto=True, sp=sp, n_jobs=-1, maxiter=100,initialization_method = 'heuristic')
#             x = forecaster.fit(X)
#             y_pred = x.predict(fh)
#             y_pred = y_pred - 1
#         else:
#             X = df_temp2['historical_sale']
#             X = X.astype(float)
#             forecaster = AutoETS(auto=True, sp=sp, n_jobs=-1, maxiter=100,initialization_method = 'heuristic')
#             x = forecaster.fit(X)
#             y_pred = x.predict(fh)
#         yhat = np.where(y_pred < 0, 0, y_pred)
#     except Exception as e:
#         # print(e)
#         # traceback.print_exc()
#         yhat = np.zeros(forecast_length)
#     return yhat

# COMMAND ----------


