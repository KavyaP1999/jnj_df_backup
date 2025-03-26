# Databricks notebook source
from prophet import Prophet
from pandas import DataFrame, Timedelta


class FBProphet:

    def __init__(self):
        self.__model__ = Prophet()

    def fit(self, X: DataFrame):

        if list(X.columns) == ['ds', 'y']:
            X.ds = X.ds - Timedelta(days=1)
            self.__model__.fit(X)
        else:
            raise Exception('X doesnot contains DS and Y columns')

    def evaluate(self, y_true: list, y_pred: list, func: object):
        return func(y_true, y_pred)

    def predict(self, future: DataFrame):
        if future.columns == ['ds']:
            return self.__model__.predict(future)[['yhat']]
        else:
            raise Exception('future doesnot contains DS column')

    def forecast(self, forecast_period, periods: int = 10):
        future = self.__model__.make_future_dataframe(periods=periods, freq=forecast_period)
        return self.__model__.predict(future)

