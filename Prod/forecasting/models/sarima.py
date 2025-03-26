# Databricks notebook source
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMA:

    def __init__(self, X: list, order = (1,0,0), seasonal_order=(0,0,0,0) ,trend= None):
        self.__model__ = SARIMAX(endog=X, order=order, seasonal_order=seasonal_order, trend=trend)

    def fit(self):
        self.__model_fit__ = self.__model__.fit()

    def evaluate(self, y_true: list, y_pred: list, func: object):
        return func(y_true, y_pred)

    def predict(self, start: int, end: int):
        return self.__model_fit__.predict(start=start, end=end)

    def forecast(self, period: int = 10):
        return self.__model_fit__.forecast(period)


# COMMAND ----------


