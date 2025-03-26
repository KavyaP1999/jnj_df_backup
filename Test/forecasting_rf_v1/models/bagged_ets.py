# Databricks notebook source
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import STL, DecomposeResult
from arch.bootstrap import MovingBlockBootstrap
from sktime.forecasting.ets import AutoETS
from numpy.random import RandomState
import traceback

# COMMAND ----------

#!pip freeze

# COMMAND ----------

seed_num = 123
rs = RandomState(seed_num)

# COMMAND ----------

def invboxcox(y, lmbda):
    if lmbda == 0:
        return (np.exp(y))
    else:
        return (np.exp(np.log(lmbda * y + 1) / lmbda))

# COMMAND ----------

def bootstarp_series(data1, boot):
    result, lmbda = boxcox(data1)
    result = STL(result, period=12).fit()
    #pass 6 back in period
    remainder = result.resid
    seasonal = result.seasonal
    trend = result.trend
    series = []
    sample = []
    seriesbc = []
    MBB = MovingBlockBootstrap(12, remainder, random_state=rs)
    MBB.seed(seed_num)
    for i, d in enumerate(MBB.bootstrap(boot)):
        mbb = d[0][0]
        sample.append(mbb)
        seriesbc.append(trend + seasonal + sample[i])
        series.append(invboxcox(seriesbc[i], lmbda))
    return series

# COMMAND ----------

# New code returning yhat instead of df2

def bagged_ets(df1, forecast_length, forecast_period):
    if forecast_period == 'M':
        sp = 12
    elif forecast_period == 'W':
        sp = 52
    elif forecast_period == 'D':
        sp = 365
    elif forecast_period == 'Q':
        sp = 4

    fh = [*range(1, forecast_length + 1, 1)]
    df_temp = df1.copy()
    df_temp.reset_index(drop=True, inplace=True)
    
    first_non_zero = df_temp['historical_sale'].ne(0).idxmax()
    
    df_temp = df_temp[first_non_zero:]
    df_temp_len = df_temp.shape[0]
    if forecast_period == 'M':
        if df_temp_len > 12 and df_temp_len <= 24:
            sp = 6
        elif df_temp_len > 6 and df_temp_len <= 12:
            sp = 3
        elif df_temp_len <= 6:
            sp = 1
    AIC = []
    BIC = []
    model = []
    forecast = []
    try:
        if (df_temp['historical_sale'] == 0).any():
            df_temp['historical_sale'] = df_temp['historical_sale'] + 1
            data = df_temp['historical_sale'].astype(float)
            series = bootstarp_series(data, boot=20)
            for i in range(len(series)):
                ser1 = pd.Series(series[i])
                ser1 = ser1.fillna(1)
                #forecaster = AutoETS(auto=True, sp=sp, n_jobs=-1, maxiter=100,initialization_method = 'heuristic')
                forecaster = AutoETS(auto=False, trend="add", seasonal="mul", damped_trend=True, sp=sp, initialization_method = 'heuristic' , n_jobs=-1, maxiter=100, bounds={"smoothing_level":[0, 0.2], "smoothing_trend":[0, 0.5], "smoothing_seasonal":[0.4, 0.7]})
                x = forecaster.fit(ser1)
                y_pred = x.predict(fh)
                aic = float(str(x.summary().as_text()).split('\n')[4].split(' ')[-1])
                bic = float(str(x.summary().as_text()).split('\n')[5].split(' ')[-1])

                y_pred = y_pred - 1
                result = np.where(y_pred < 0, 0, y_pred)
                
                AIC.append(aic)
                BIC.append(bic)
                model.append(x)
                forecast.append(result)
        
        else:
            data = df_temp['historical_sale'].astype(float)
            series = bootstarp_series(data, boot=20)
            for i in range(len(series)):
                ser1 = pd.Series(series[i])
                ser1 = ser1.fillna(1)
                #forecaster = AutoETS(auto=True, sp=sp, n_jobs=-1, maxiter=100,initialization_method = 'heuristic')
                forecaster = AutoETS(auto=False, trend="add", seasonal="mul", damped_trend=True, sp=sp, initialization_method = 'heuristic' , n_jobs=-1, maxiter=100, bounds={"smoothing_level":[0, 0.2], "smoothing_trend":[0, 0.5], "smoothing_seasonal":[0.4, 0.7]})
                x = forecaster.fit(ser1)
                y_pred = x.predict(fh)
                aic = float(str(x.summary().as_text()).split('\n')[4].split(' ')[-1])
                bic = float(str(x.summary().as_text()).split('\n')[5].split(' ')[-1])

                result = np.where(y_pred < 0, 0, y_pred)
                
                AIC.append(aic)
                BIC.append(bic)
                model.append(x)
                forecast.append(result)
        

        metric = {'AIC': AIC, 'BIC': BIC, 'MODEL': model, 'FORECAST': forecast}
        metric_model = pd.DataFrame(metric)
        metric_model = metric_model.sort_values(by=['AIC', 'BIC'])
        hat = metric_model['FORECAST'].iloc[0]
        

        df_agg = metric_model.copy()

        df_agg.drop(['AIC','BIC','MODEL'], axis=1,inplace = True)
        aggr = pd.DataFrame(forecast)
        yhat = aggr.mean(axis=0)
        yhat = yhat[:].values
        
    except Exception as e:
        print(e)
        traceback.print_exc()
        yhat = np.zeros(forecast_length)
        
    return yhat

# COMMAND ----------

# # New_code returning df2 instead of yhat

# #With aggregation code
# def bagged_ets(df1, forecast_length, forecast_period):
#     if forecast_period == 'M':
#         sp = 12
#     elif forecast_period == 'W':
#         sp = 52
#     elif forecast_period == 'D':
#         sp = 365
#     elif forecast_period == 'Q':
#         sp = 4

#     fh = [*range(1, forecast_length + 1, 1)]
#     df_temp = df1.copy()
#     df_temp.reset_index(drop=True, inplace=True)
    
#     first_non_zero = df_temp['historical_sale'].ne(0).idxmax()
    
#     df_temp = df_temp[first_non_zero:]
#     df_temp_len = df_temp.shape[0]
#     if forecast_period == 'M':
#         if df_temp_len > 12 and df_temp_len <= 24:
#             sp = 6
#         elif df_temp_len > 6 and df_temp_len <= 12:
#             sp = 3
#         elif df_temp_len <= 6:
#             sp = 1
#     AIC = []
#     BIC = []
#     model = []
#     forecast = []
#     try:
#         if (df_temp['historical_sale'] == 0).any():
#             df_temp['historical_sale'] = df_temp['historical_sale'] + 1
#             data = df_temp['historical_sale'].astype(float)
#             series = bootstarp_series(data, boot=20)
#             for i in range(len(series)):
#                 ser1 = pd.Series(series[i])
#                 ser1 = ser1.fillna(1)
#                 #forecaster = AutoETS(auto=True, sp=sp, n_jobs=-1, maxiter=100,initialization_method = 'heuristic')
#                 forecaster = AutoETS(auto=False, trend="add", seasonal="mul", damped_trend=True, sp=sp, initialization_method = 'heuristic' , n_jobs=-1, maxiter=100, bounds={"smoothing_level":[0, 0.2], "smoothing_trend":[0, 0.5], "smoothing_seasonal":[0.4, 0.7]})
#                 x = forecaster.fit(ser1)
#                 y_pred = x.predict(fh)
#                 aic = float(str(x.summary().as_text()).split('\n')[4].split(' ')[-1])
#                 bic = float(str(x.summary().as_text()).split('\n')[5].split(' ')[-1])

#                 y_pred = y_pred - 1
#                 result = np.where(y_pred < 0, 0, y_pred)
                
#                 AIC.append(aic)
#                 BIC.append(bic)
#                 model.append(x)
#                 forecast.append(result)
        
#         else:
#             data = df_temp['historical_sale'].astype(float)
#             series = bootstarp_series(data, boot=20)
#             for i in range(len(series)):
#                 ser1 = pd.Series(series[i])
#                 ser1 = ser1.fillna(1)
#                 #forecaster = AutoETS(auto=True, sp=sp, n_jobs=-1, maxiter=100,initialization_method = 'heuristic')
#                 forecaster = AutoETS(auto=False, trend="add", seasonal="mul", damped_trend=True, sp=sp, initialization_method = 'heuristic' , n_jobs=-1, maxiter=100, bounds={"smoothing_level":[0, 0.2], "smoothing_trend":[0, 0.5], "smoothing_seasonal":[0.4, 0.7]})
#                 x = forecaster.fit(ser1)
#                 y_pred = x.predict(fh)
#                 aic = float(str(x.summary().as_text()).split('\n')[4].split(' ')[-1])
#                 bic = float(str(x.summary().as_text()).split('\n')[5].split(' ')[-1])

#                 result = np.where(y_pred < 0, 0, y_pred)
                
#                 AIC.append(aic)
#                 BIC.append(bic)
#                 model.append(x)
#                 forecast.append(result)
        

#         metric = {'AIC': AIC, 'BIC': BIC, 'MODEL': model, 'FORECAST': forecast}
#         metric_model = pd.DataFrame(metric)
#         metric_model = metric_model.sort_values(by=['AIC', 'BIC'])
#         yhat = metric_model['FORECAST'].iloc[0]
#         df_agg = metric_model.copy()

#         df_agg.drop(['AIC','BIC','MODEL'], axis=1,inplace = True)
#         aggr = pd.DataFrame(forecast, columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'])
#         df2 = aggr.mean(axis=0)
#         df2 = df2[:].values
        
#     except Exception as e:
#         print(e)
#         traceback.print_exc()
#         yhat = np.zeros(forecast_length)
#     return df2

# COMMAND ----------

# # Older code
# def bagged_ets(df1, forecast_length, forecast_period):
#     if forecast_period == 'M':
#         sp = 12
#     elif forecast_period == 'W':
#         sp = 52
#     elif forecast_period == 'D':
#         sp = 365
#     elif forecast_period == 'Q':
#         sp = 4

#     fh = [*range(1, forecast_length + 1, 1)]
#     df_temp = df1.copy()
#     df_temp.reset_index(drop=True, inplace=True)
#     first_non_zero = df_temp['historical_sale'].ne(0).idxmax()
#     df_temp = df_temp[first_non_zero:]
#     df_temp_len = df_temp.shape[0]
#     if forecast_period == 'M':
#         if df_temp_len > 12 and df_temp_len <= 24:
#             sp = 6
#         elif df_temp_len > 6 and df_temp_len <= 12:
#             sp = 3
#         elif df_temp_len <= 6:
#             sp = 1
#     AIC = []
#     BIC = []
#     model = []
#     forecast = []
#     try:
#         if (df_temp['historical_sale'] == 0).any():
#             df_temp['historical_sale'] = df_temp['historical_sale'] + 1
#             data = df_temp['historical_sale'].astype(float)
#             series = bootstarp_series(data, boot=20)
#             for i in range(len(series)):
#                 ser1 = pd.Series(series[i])
#                 ser1 = ser1.fillna(1)
#                 forecaster = AutoETS(auto=True, sp=sp, n_jobs=-1, maxiter=100)
#                 x = forecaster.fit(ser1)
#                 y_pred = x.predict(fh)
#                 y_pred = y_pred - 1
#         else:
#             data = df_temp['historical_sale'].astype(float)
#             series = bootstarp_series(data, boot=20)
#             for i in range(len(series)):
#                 ser1 = pd.Series(series[i])
#                 ser1 = ser1.fillna(1)
#                 forecaster = AutoETS(auto=True, sp=sp, n_jobs=-1, maxiter=100)
#                 x = forecaster.fit(ser1)
#                 y_pred = x.predict(fh)
#         aic = float(str(x.summary().as_text()).split('\n')[4].split(' ')[-1])
#         bic = float(str(x.summary().as_text()).split('\n')[5].split(' ')[-1])

#         result = np.where(y_pred < 0, 0, y_pred)

#         AIC.append(aic)
#         BIC.append(bic)
#         model.append(x)
#         forecast.append(result)

#         metric = {'AIC': AIC, 'BIC': BIC, 'MODEL': model, 'FORECAST': forecast}
#         metric_model = pd.DataFrame(metric)
#         metric_model = metric_model.sort_values(by=['AIC', 'BIC'])
#         yhat = metric_model['FORECAST'].iloc[0]

#     except Exception as e:
#         print(e)
#         traceback.print_exc()
#         yhat = np.zeros(forecast_length)
#     return yhat

# COMMAND ----------


# def bagged_ets(df1, forecast_length, forecast_period):
#     if forecast_period == 'M':
#         sp = 12
#     elif forecast_period == 'W':
#         sp = 52
#     elif forecast_period == 'D':
#         sp = 365
#     elif forecast_period == 'Q':
#         sp = 4

#     fh = [*range(1, forecast_length + 1, 1)]
#     df_temp = df1.copy()
#     df_temp.reset_index(drop=True, inplace=True)
#     first_non_zero = df_temp['historical_sale'].ne(0).idxmax()
#     df_temp = df_temp[first_non_zero:]
#     df_temp_len = df_temp.shape[0]
#     if forecast_period == 'M':
#         if df_temp_len > 12 and df_temp_len <= 24:
#             sp = 6
#         elif df_temp_len > 6 and df_temp_len <= 12:
#             sp = 3
#         elif df_temp_len <= 6:
#             sp = 1
#     AIC = []
#     BIC = []
#     model = []
#     forecast = []
#     try:
#         if (df_temp['historical_sale'] == 0).any():
#             df_temp['historical_sale'] = df_temp['historical_sale'] + 1
#             data = df_temp['historical_sale'].astype(float)
#             series = bootstarp_series(data, boot=20)
#             for i in range(len(series)):
#                 ser1 = pd.Series(series[i])
#                 ser1 = ser1.fillna(1)
#                 forecaster = AutoETS(auto=True, sp=sp, n_jobs=-1, maxiter=100,initialization_method = 'heuristic')
#                 x = forecaster.fit(ser1)
#                 y_pred = x.predict(fh)
#                 y_pred = y_pred - 1
#         else:
#             data = df_temp['historical_sale'].astype(float)
#             series = bootstarp_series(data, boot=20)
#             for i in range(len(series)):
#                 ser1 = pd.Series(series[i])
#                 ser1 = ser1.fillna(1)
#                 forecaster = AutoETS(auto=True, sp=sp, n_jobs=-1, maxiter=100,initialization_method = 'heuristic')
#                 x = forecaster.fit(ser1)
#                 y_pred = x.predict(fh)
#         aic = float(str(x.summary().as_text()).split('\n')[4].split(' ')[-1])
#         bic = float(str(x.summary().as_text()).split('\n')[5].split(' ')[-1])

#         result = np.where(y_pred < 0, 0, y_pred)

#         AIC.append(aic)
#         BIC.append(bic)
#         model.append(x)
#         forecast.append(result)

#         metric = {'AIC': AIC, 'BIC': BIC, 'MODEL': model, 'FORECAST': forecast}
#         metric_model = pd.DataFrame(metric)
#         metric_model = metric_model.sort_values(by=['AIC', 'BIC'])
#         yhat = metric_model['FORECAST'].iloc[0]

#     except Exception as e:
#         print(e)
#         traceback.print_exc()
#         yhat = np.zeros(forecast_length)
#     return yhat
