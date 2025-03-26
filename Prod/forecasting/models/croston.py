# Databricks notebook source
from croston.croston import fit_croston
import numpy as np

# COMMAND ----------

def croston(df1, forecast_size):
    try:
        df_temp2 = df1.copy()
        df_temp2.reset_index(drop=True, inplace=True)
        first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
        df_temp2 = df_temp2[first_non_zero:]
        X = df_temp2['historical_sale']
        X = X.astype(np.float)
        if np.count_nonzero(X) > 0:
            fit_pred = fit_croston(X, forecast_size)
            forecast = fit_pred['croston_forecast']
            yhat = np.array(forecast)
            yhat = np.array([0 if i is None else i for i in yhat])
        else:
            forecast = np.zeros(forecast_size)
            yhat = np.array(forecast)
    except:
        yhat = np.zeros(forecast_size)
    return yhat

# COMMAND ----------


