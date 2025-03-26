# Databricks notebook source
import pmdarima as pm
import numpy as np

# COMMAND ----------

def arima(df1, forecast_length, start_p=2, d=None, start_q=2, max_p=5, max_d=2, max_q=5, seasonal=False, information_criterion='aic', maxiter=50,scoring='mse'):
    try:
        df_temp2 = df1.copy()
        df_temp2.reset_index(drop=True, inplace=True)
        first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
        df_temp2 = df_temp2[first_non_zero:]
        X = df_temp2['historical_sale']
        model = pm.auto_arima(X, start_p=start_p, d=d, start_q=start_q, max_p=max_p, max_d=max_d, max_q=max_q, seasonal=seasonal, information_criterion=information_criterion, maxiter=maxiter,scoring=scoring, n_jobs=-1)
        forecast = model.predict(forecast_length)
    except:
        forecast = np.zeros(forecast_length)
    return forecast


# COMMAND ----------


