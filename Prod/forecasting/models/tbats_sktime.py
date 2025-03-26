# Databricks notebook source
import pandas as pd
import numpy as np
import time
from sktime.forecasting.tbats import TBATS

# COMMAND ----------

def tbats_sktime(df1, forecast_length,sp=[3,12],use_trend=True, use_damped_trend=True, use_arma_errors=False,
          use_box_cox=False):
    
    fh = np.arange(1,forecast_length+1)
    df_temp2 = df1.copy()
    df_temp2.reset_index(drop=True, inplace=True)
    first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
    df_temp2 = df_temp2[first_non_zero:]
    X = df_temp2['historical_sale']
    X = X.astype(np.float)
    estimator = TBATS(use_trend=use_trend,sp=sp,
                      use_damped_trend=use_damped_trend,
                      use_arma_errors=use_arma_errors, use_box_cox=use_box_cox,n_jobs=-1)
    estimator.fit(X)
    yhat = estimator.predict(fh=fh)
    yhat = np.where(yhat < 0, 0, yhat)
    print(yhat)
    #yhat = np.zeros(forecast_length)
    
    return yhat

# COMMAND ----------


