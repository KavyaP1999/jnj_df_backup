# Databricks notebook source
import pmdarima as pm
import numpy as np

# COMMAND ----------

# New_Code
# taken d=0 and max_d =3
def arima(df1, forecast_length, start_p=2, d=0, start_q=2, max_p=12, max_d=3, m=12, max_q=12, seasonal=True, information_criterion='aic', maxiter=500, scoring='mse'):
    count =0
    try:
        df_temp2 = df1.copy()
        
        if (df_temp2['historical_sale'] == 0).any():
            df_temp2['historical_sale'] = df_temp2['historical_sale']+1
            count =1
        df_temp2.reset_index(drop=True, inplace=True)
        first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
        df_temp2 = df_temp2[first_non_zero:]
        X = df_temp2['historical_sale']
        model = pm.auto_arima(X, start_p=start_p, d=d, start_q=start_q, max_p=max_p, max_d=max_d, max_q=max_q, seasonal=seasonal, information_criterion=information_criterion, maxiter=maxiter,scoring=scoring, m=m, n_jobs=-1, max_P=12, max_D=2, max_Q=12, max_order=None)
        forecast = model.predict(forecast_length)
        if count == 0:
            forecast = forecast-1
    except:
        try:
            df_temp2 = df1.copy()
            if (df_temp2['historical_sale'] == 0).any():
                df_temp2['historical_sale'] = df_temp2['historical_sale']+1
                count =1
            df_temp2.reset_index(drop=True, inplace=True)
            first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
            df_temp2 = df_temp2[first_non_zero:]
            X = df_temp2['historical_sale']
            model = pm.auto_arima(X, start_p=start_p, d=d, start_q=start_q, max_p=max_p, max_q=max_q, seasonal=seasonal, information_criterion=information_criterion, maxiter=maxiter,scoring=scoring, m=m, n_jobs=-1, max_P=12, max_Q=12, max_order=None)
            forecast = model.predict(forecast_length)
            if count == 0:
                forecast = forecast-1
        except:
            
            forecast = np.zeros(forecast_length)
            
    return forecast


# COMMAND ----------

# # older_version_2.2.10
# def arima(df1, forecast_length, start_p=2, d=1, start_q=2, max_p=12, max_d=12, m=12, max_q=12, seasonal=True, information_criterion='aic', maxiter=500, scoring='mse'):
#     try:
#         df_temp2 = df1.copy()
#         df_temp2.reset_index(drop=True, inplace=True)
#         first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
#         df_temp2 = df_temp2[first_non_zero:]
#         X = df_temp2['historical_sale']
#         model = pm.auto_arima(X, start_p=start_p, d=d, start_q=start_q, max_p=max_p, max_d=max_d, max_q=max_q, seasonal=seasonal, information_criterion=information_criterion, maxiter=maxiter,scoring=scoring, m=m, n_jobs=-1, max_P=12, max_D=12, max_Q=12, max_order=None)
#         forecast = model.predict(forecast_length)
#     except:
#         forecast = np.zeros(forecast_length)
#     return forecast


# COMMAND ----------


