# Databricks notebook source
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from pandas import DataFrame
import numpy as np
from datetime import datetime
from datetime import timedelta
import datetime as dt



class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


def add_month(df, forecast_length, forecast_period):
    # forecast_length=forecast_length
    # forecast_period=forecast_period
    end_point = len(df)
    df1 = pd.DataFrame(index=range(forecast_length), columns=range(2))
    df1.columns = ['historical_sale', 'date']
    df = df.append(df1)
    df = df.reset_index(drop=True)
    x = df.at[end_point - 1, 'date']
    x = pd.to_datetime(x, format='%Y-%m-%d')
    if forecast_period == 'W':
        for i in range(forecast_length):
            df.at[df.index[end_point + i], 'date'] = x + timedelta(days=7 + 7 * i)
            df.at[df.index[end_point + i], 'historical_sale'] = 0
    elif forecast_period == 'M':
        for i in range(forecast_length):
            df.at[df.index[end_point + i], 'date'] = x + timedelta(days=30 + 30 * i)
            df.at[df.index[end_point + i], 'historical_sale'] = 0

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['month'] = df['date'].dt.month
    dummy = pd.get_dummies(df['month'])
    df = pd.concat([df, dummy], axis=1)
    df = df.drop(['month', 'date'], axis=1)
    return df


def create_lag(df3, lags):
    dataframe = DataFrame()
    for i in range(lags, 0, -1):
        dataframe['t-' + str(i)] = df3.historical_sale.shift(i)
    df4 = pd.concat([df3, dataframe], axis=1)
    df4.dropna(inplace=True)
    return df4


def randomForest(df1, forecast_length, forecast_period):
    try:
        df_temp2 = df1.copy()
        df_temp2.reset_index(drop=True, inplace=True)
        first_non_zero = df_temp2['historical_sale'].ne(0).idxmax()
        # df_temp2 = df_temp2[first_non_zero:]
        # df_temp2_len = df_temp2.shape[0]
        if first_non_zero / df_temp2.shape[0] > 0.25:
            df_temp2['historical_sale'].iloc[:first_non_zero] = df_temp2['historical_sale'].iloc[
                                                                :first_non_zero].replace(0,
                                                                                         df_temp2['historical_sale'].iloc[first_non_zero:].mean())
        lags = 12

    #     df_temp2 = df_temp2.loc[first_non_zero:, :].copy()
    #     df_temp2 = df_temp2['historical_sale']
    #     df_temp2 = df_temp2.astype(float)
    #     step_in = 12
    #     step_out = 12

    #     if df_temp2.shape[0] < (step_out + step_in):
    #         zeros_added = step_out + step_in - data.shape[0]
    #         df_temp2 = pd.concat([pd.DataFrame({'historical_sale': np.zeros(zeros_added)}), df_temp2])
    #         df_temp2 = df_temp2['historical_sale']
    #         df_temp2 = df_temp2.astype(float)

        df3 = df_temp2[['historical_sale', 'date']]
        df3 = add_month(df3, forecast_length, forecast_period)
        finaldf = create_lag(df3, lags)
        finaldf = finaldf.reset_index(drop=True)
        n = forecast_length
        end_point = len(finaldf)
        x = end_point - n
        finaldf_train = finaldf.loc[:x - 1, :]
        finaldf_train_x = finaldf_train.loc[:, finaldf_train.columns != 'historical_sale']
        finaldf_train_y = finaldf_train['historical_sale']
        # print("Starting model train..")
    #     param_grid = {
    #         'max_depth': [5, 10],
    #         'max_features': ['auto', 'sqrt'],
    #         'min_samples_leaf': [1, 2, 4],
    #         'min_samples_split': [2, 4, 6],
    #         'n_estimators': [20, 40, 60, 80, 100]
    #     }
        rfe = RandomForestRegressor(random_state=1, n_jobs=-1)
        fit = rfe.fit(finaldf_train_x, finaldf_train_y)
    #     btscv = BlockingTimeSeriesSplit(n_splits=5)
    #     grid_search = GridSearchCV(estimator=rfe, param_grid=param_grid, cv=btscv, n_jobs=-1, verbose=1)
    #     fit = grid_search.fit(finaldf_train_x, finaldf_train_y)
        # print(fit.best_params_)
        # print("Model train completed..")
        # print("Creating forecasted set..")
        yhat = []
        end_point = len(finaldf)
        n = forecast_length
        df3_end = len(df3)
        for i in range(n, 0, -1):
            y = end_point - i
            inputfile = finaldf.loc[y:end_point, :]
            inputfile_x = inputfile.loc[:, inputfile.columns != 'historical_sale']
            pred_set = inputfile_x.head(1)
            pred = fit.predict(pred_set)
            df3.at[df3.index[df3_end - i], 'historical_sale'] = pred[0]
            finaldf = create_lag(df3, lags)
            finaldf = finaldf.reset_index(drop=True)
            yhat.append(pred)
        yhat = np.array(yhat)
    except:
        yhat = np.zeros(forecast_length)   
    # print("Forecast complete..")
    return yhat

