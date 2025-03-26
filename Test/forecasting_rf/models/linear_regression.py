# Databricks notebook source
from sktime.forecasting.trend import PolynomialTrendForecaster
import numpy as np


def linear_regression(df1, forecast_length):
    try:
        fh = [*range(1, forecast_length+1, 1)]
        df_temp2 = df1.copy()
        df_temp2.reset_index(drop=True, inplace=True)
        first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
        df_temp2 = df_temp2[first_non_zero:]
        X = df_temp2['historical_sale']
        X = X.astype(float)
        forecaster = PolynomialTrendForecaster()
        x = forecaster.fit(X)
        y_pred = x.predict(fh)
        y_hat = np.where(y_pred<0, 0, y_pred)
    except:
        yhat = np.zeros(forecast_length)
    return y_hat
