# Databricks notebook source
import numpy as np
import pandas as pd
import os, sys
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import *
import time
from functools import reduce
from pyspark.sql import SparkSession
import warnings
import ast
import re
import pyspark.sql.functions as F

# COMMAND ----------

pip list

# COMMAND ----------

ingestion = 1

# COMMAND ----------

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

# def db_ingestion(df, table_name, mode):
#     counts = df.count()
#     paritions = counts // 1000000 + 1
#     #default partition = 24
#     df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)
    
#     return

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/arima

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/lstm

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/auto_ets

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/croston

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/trend

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/linear_regression

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/naive_forecast

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/prophet_univariate

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/randomForest_univariate

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/sarima

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/tbats_sktime

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/xgBoost_simple

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/static_ets

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/MLP

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/models/bagged_ets

# COMMAND ----------

def run_parameter_data(run_id,url,properties):
    
    parameter = spark.read.jdbc(url=url,table=f"(select parameter_id,run_value from run_parameter where run_id = '{run_id}') as run_value",properties=properties)
    parameter = parameter.toPandas()
   
    parameter = dict(zip(parameter['parameter_id'],parameter['run_value']))
    
    return parameter

# COMMAND ----------

dbutils.widgets.text('run_id', '', 'run_id')
dbutils.widgets.text('domain_id', '', 'domain_id')
run_id = dbutils.widgets.get("run_id")
run_id = str(run_id)
domain_id = dbutils.widgets.get("domain_id")

# COMMAND ----------

parameters = run_parameter_data(run_id,url,properties)
model_select = parameters[22]
run_mode = parameters[25]
rule_based_model_selection = parameters[96]
forecast_length = int(parameters[24])
test_size = int(parameters[160])
folds = ','.join(map(str, range(int(parameters[162]))))
model_id = str(parameters[23])
error_weight = float(parameters[163])
me_weight = float(parameters[165])
phase_weight = float(parameters[164])
best_model_selection_metric = parameters[76]
rule_based_model_selection = ast.literal_eval(rule_based_model_selection)
post_forecast_adj = parameters[158]
post_forecast_adj = ast.literal_eval(post_forecast_adj)
fcast_norm_models = list(map(str,post_forecast_adj['fcst_norm']))
s445_models = list(map(str,post_forecast_adj['system_445']))
smoothing_rules = ast.literal_eval(parameters[100])
forecast_normalization = parameters[34]
blended_smoothing = parameters[155]
window_length_smoothing = int(parameters[92])
# window_length_static = 3
system445 = parameters[156]
# alpha = 0
# beta = 0
# gamma = 0.3

# COMMAND ----------

# run_mode = 'forecast'

# COMMAND ----------

# dp3_data = spark.read.jdbc(url=url, table= f"(SELECT org_unit_id, channel_id, product_id, period, historical_sale, forecast,segmentation FROM run_data WHERE run_id = '{run_id}'and  run_state = 'DP3') as dp3_data", properties = properties)

working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP3_run_data.parquet"
dp3_data = spark.read.format('parquet').load(working_path,inferSchema=True,header = True)
dp3_data = dp3_data.select('org_unit_id', 'channel_id', 'product_id', 'period', 'historical_sale', 'forecast','segmentation')


df_norm = spark.read.jdbc(url=url, table=f"(select * from normalization where run_id = '{run_id}') as normalization", properties = properties)

empty_rdd = spark.sparkContext.emptyRDD()
columns = StructType([StructField('org_unit_id',
                              StringType(), True),
                StructField('channel_id',
                            StringType(), True),
                StructField('product_id',
                            StringType(), True),
                StructField('period',
                        StringType(), True),
                StructField('historical_sale',
                        FloatType(), True),
                StructField('forecast',
                        FloatType(), True),      
                StructField('segmentation',
                            StringType(), True),
                StructField('model',
                            StringType(), True)])
df_temp = spark.createDataFrame(data = empty_rdd,schema=columns)

for k,v in rule_based_model_selection.items():

    df_filter_segment = dp3_data.filter(dp3_data.segmentation==k)
    model_id = ",".join(list(map(str,v['seasonal'])))
    df_filter_segment = df_filter_segment.withColumn('model', lit(model_id)).withColumn('model', split(col('model'), ',')).withColumn('model', explode('model'))
    df_temp = df_temp.union(df_filter_segment)

dp3_data = df_temp
dp3_data = dp3_data.withColumn('fold', lit(folds)).withColumn('fold', split(col('fold'), ',')).withColumn('fold', explode('fold'))
dp3_data = dp3_data.select(concat(dp3_data.org_unit_id, lit('@'), dp3_data.channel_id, lit('@'), dp3_data.product_id, lit('@'), dp3_data.model, lit('@'), dp3_data.fold).alias('key'), 'period', 'historical_sale', 'forecast', 'segmentation')
dp3_data = dp3_data.withColumn('historical_sale', dp3_data['historical_sale'].cast(IntegerType()))
dp3_data = dp3_data.withColumn('forecast', dp3_data['forecast'].cast(IntegerType()))
dp3_data = dp3_data.drop('org_unit_id', 'channel_id', 'product_id')
# dp3_data = dp3_data.repartition('key')

# COMMAND ----------

# dp3_data = dp3_data.filter(dp3_data.key == '104@Revenue@TR0113002203@2@2')

# COMMAND ----------

def get_dates(df):
    # year = int(re.split('W|D|M|Q', input_period)[0])
    # time = int(re.split('W|D|M|Q', input_period)[1])
    if df['period'][0][4] == 'W':
        df['temp_period'] = [string[: 4] + string[6:] for string in df['period']]
        df['date'] = pd.to_datetime(df.temp_period.astype(str).add('-1'), format='%G%V-%u')
    elif df['period'][0][4] == 'M':
        df['temp_period'] = [string[: 4] + string[6:] for string in df['period']]
        df['date'] = pd.to_datetime(df.temp_period.astype(str), format='%Y%m')
    elif df['period'][0][4] == 'D':
        df['year'] = [string[: 4] for string in df['period']]
        df['doy'] = [string[5:] for string in df['period']]
        df['year'] = df['year'].astype(int)
        df['doy'] = df['doy'].astype(int)
        df['date'] = compose_date(df['year'], days=df['doy'])
    elif df['period'][0][4] == 'Q':
        pass
        # print("Dates can't be converted")
    if df['period'][0][4] == 'D':
        df.drop(['year', 'doy'], axis=1, inplace=True)
    else:
        df.drop(['temp_period'], axis=1, inplace=True)
    # # print(df)
    return df

# COMMAND ----------

@pandas_udf(dp3_data.schema, PandasUDFType.GROUPED_MAP)
def model_training(data):
    data.sort_values('period', inplace=True)
    data.reset_index(drop=True, inplace=True)
#     global smoothing_rules, window_length
    global window_length_smoothing
    global alpha
    global beta
    global gamma
    global blended_smoothing
    global forecast_length
    global window_length_static
    global smoothing_rules
    global run_mode
    global test_size
    global fold
    
    if(run_mode=='poc'):
        data = data[:-forecast_length]
    model = data['key'].map(lambda x:str(x).split('@')[-2]).drop_duplicates().values[0]
    fold = int(data['key'].map(lambda x:str(x).split('@')[-1]).drop_duplicates().values[0])
    fold_data = pd.DataFrame()
    if fold != 0:
        fold_data = data.iloc[-fold:].copy()
        data = data.iloc[:-fold].copy()
    train_data = data.iloc[:-test_size].copy()
    
    def wheighted_moving_average_smoothing(df: DataFrame):
        target_var = 'historical_sale'
        df['moving_average'] = 0.0
        df.reset_index(drop=True,inplace=True)
        if df['historical_sale'].sum()==0:

            df_result = df

        else:

            df_x = df
            first_non_zero = df_x[target_var].ne(0).idxmax()
            df_e = df_x[first_non_zero:]
            df_e.reset_index(drop=True, inplace=True)
            key = df_e['key'].values

            df_z = df_x[:first_non_zero]

            Sales = df_e[target_var].map(float)
            if len(Sales) < 9:
                df_result = df_x
            else:
                data = df_e.copy()
                data.historical_sale = data.historical_sale.map(float)

                wts_data = [{'first':0,'second':0,'third':0,'fourth':0,'regular':0.024691358,'last':0.020576132,'secondlast':0.020576132,'thirdlast':0.020576132,'fourthlast':0.020576132},
          {'first':0,'second':0,'third':0,'fourth':0.086419753,'regular':0.061728395,'last':0.061728395,'secondlast':0.061728395,'thirdlast':0.061728395,'fourthlast':0.061728395},{'first':0,'second':0,'third':0.209876543,'fourth':0.12345679,'regular':0.12345679,'last':0.12345679,'secondlast':0.12345679,'thirdlast':0.12345679,'fourthlast':0.12345679},
    {'first':0,'second':0.395061728,'third':0.185185185,'fourth':0.185185185,'regular':0.185185185,'last':0.185185185,'secondlast':0.185185185,'thirdlast':0.185185185,'fourthlast':0.185185185},{'first':0.604938272,'second':0.209876543,'third':0.209876543,'fourth':0.209876543,'regular':0.209876543,'last':0.604938272,'secondlast':0.209876543,'thirdlast':0.209876543
    ,'fourthlast':0.209876543},{'first':0.19,'second':0.185185185,'third':0.185185185,'fourth':0.185185185,'regular':0.185185185,'last':0,'secondlast':0.395061728,'thirdlast':0.185185185,'fourthlast':0.185185185},{'first':0.12,'second':0.12345679,'third':0.12345679,'fourth':0.12345679,'regular':0.12345679,'last':0,'secondlast':0,'thirdlast':0.209876543,'fourthlast':0.12345679},{'first':0.06,'second':0.061728395,'third':0.061728395,'fourth':0.061728395,'regular':0.061728395,'last':0,'secondlast':0,'thirdlast':0,'fourthlast':0.086419753
    },{'first':0.02,'second':0.020576132,'third':0.020576132,'fourth':0.020576132,'regular':0.020576132,'last':0,'secondlast':0,'thirdlast':0,'fourthlast':0}]
                
                wheights_data = pd.DataFrame(wts_data)
                historical_sale_index = data.columns.get_loc('historical_sale')
                moving_average_index = data.columns.get_loc('moving_average')
                for i in range(data.shape[0]):

                    if i == 0:
                        wheights = np.array(wheights_data['first'])
                        moving_average = (data.iloc[0, historical_sale_index] * wheights[4]) + (
                            data.iloc[1, historical_sale_index] * wheights[5]) + (
                                         data.iloc[2, historical_sale_index] * wheights[6]) + (
                                         data.iloc[3, historical_sale_index] * wheights[7]) + (
                                         data.iloc[4, historical_sale_index] * wheights[8])
                        data.iloc[0, moving_average_index] = moving_average
                    elif i == 1:
                        wheights = np.array(wheights_data['second'])
                        moving_average = (data.iloc[1, historical_sale_index] * wheights[4]) + (
                            data.iloc[2, historical_sale_index] * wheights[5]) + (
                                         data.iloc[3, historical_sale_index] * wheights[6]) + (
                                         data.iloc[4, historical_sale_index] * wheights[7]) + (
                                         data.iloc[5, historical_sale_index] * wheights[8]) + (
                                         data.iloc[0, historical_sale_index] * wheights[3])
                        data.iloc[1, moving_average_index] = moving_average
                    elif i == 2:
                        wheights = np.array(wheights_data['third'])
                        moving_average = (data.iloc[2, historical_sale_index] * wheights[4]) + (
                            data.iloc[3, historical_sale_index] * wheights[5]) + (
                                         data.iloc[4, historical_sale_index] * wheights[6]) + (
                                         data.iloc[5, historical_sale_index] * wheights[7]) + (
                                         data.iloc[6, historical_sale_index] * wheights[8]) + (
                                         data.iloc[0, historical_sale_index] * wheights[2]) + (
                                         data.iloc[1, historical_sale_index] * wheights[3])
                        data.iloc[2, moving_average_index] = moving_average
                    elif i == 3:

                        wheights = np.array(wheights_data['fourth'])
                        moving_average = (data.iloc[3, historical_sale_index] * wheights[4]) + (
                            data.iloc[4, historical_sale_index] * wheights[5]) + (
                                         data.iloc[5, historical_sale_index] * wheights[6]) + (
                                         data.iloc[6, historical_sale_index] * wheights[7]) + (
                                         data.iloc[7, historical_sale_index] * wheights[8]) + (
                                         data.iloc[0, historical_sale_index] * wheights[1]) + (
                                         data.iloc[1, historical_sale_index] * wheights[2]) + (
                                         data.iloc[2, historical_sale_index] * wheights[3])
                        data.iloc[3, moving_average_index] = moving_average

                    elif i == (data.shape[0] - 1):
                        wheights = np.array(wheights_data['last'])
                        moving_average = (data.iloc[(data.shape[0] - 1), historical_sale_index] * wheights[4]) + (
                            data.iloc[(data.shape[0] - 2), historical_sale_index] * wheights[3]) + (
                                         data.iloc[(data.shape[0] - 3), historical_sale_index] * wheights[2]) + (
                                         data.iloc[(data.shape[0] - 4), historical_sale_index] * wheights[1]) + (
                                         data.iloc[(data.shape[0] - 5), historical_sale_index] * wheights[0])
                        data.iloc[(data.shape[0] - 1), moving_average_index] = moving_average

                    elif i == (data.shape[0] - 2):

                        wheights = np.array(wheights_data['secondlast'])
                        moving_average = (data.iloc[(data.shape[0] - 1), historical_sale_index] * wheights[5]) + (
                        data.iloc[(data.shape[0] - 2), historical_sale_index] * wheights[4]) + (
                                         data.iloc[(data.shape[0] - 3), historical_sale_index] * wheights[3]) + (
                                         data.iloc[(data.shape[0] - 4), historical_sale_index] * wheights[2]) + (
                                         data.iloc[(data.shape[0] - 5), historical_sale_index] * wheights[1]) + (
                                         data.iloc[(data.shape[0] - 6), historical_sale_index] * wheights[0])

                        data.iloc[(data.shape[0] - 2), moving_average_index] = moving_average

                    elif i == (data.shape[0] - 3):
                        wheights = np.array(wheights_data['thirdlast'])

                        moving_average = (data.iloc[(data.shape[0] - 7), historical_sale_index] * wheights[0]) + (
                        data.iloc[(data.shape[0] - 6), historical_sale_index] * wheights[1]) + (
                                         data.iloc[(data.shape[0] - 5), historical_sale_index] * wheights[2]) + (
                                         data.iloc[(data.shape[0] - 4), historical_sale_index] * wheights[3]) + (
                                         data.iloc[(data.shape[0] - 3), historical_sale_index] * wheights[4]) + (
                                         data.iloc[(data.shape[0] - 2), historical_sale_index] * wheights[5]) + (
                                         data.iloc[(data.shape[0] - 1), historical_sale_index] * wheights[6])
                        data.iloc[(data.shape[0] - 3), moving_average_index] = moving_average

                    elif i == (data.shape[0] - 4):


                        wheights = np.array(wheights_data['fourthlast'])
                        moving_average = (data.iloc[(data.shape[0] - 8), historical_sale_index] * wheights[0]) + (
                        data.iloc[(data.shape[0] - 7), historical_sale_index] * wheights[1]) + (
                                         data.iloc[(data.shape[0] - 6), historical_sale_index] * wheights[2]) + (
                                         data.iloc[(data.shape[0] - 5), historical_sale_index] * wheights[3]) + (
                                         data.iloc[(data.shape[0] - 4), historical_sale_index] * wheights[4]) + (
                                         data.iloc[(data.shape[0] - 3), historical_sale_index] * wheights[5]) + (
                                         data.iloc[(data.shape[0] - 2), historical_sale_index] * wheights[6]) + (
                                         data.iloc[(data.shape[0] - 1), historical_sale_index] * wheights[7])


                        data.iloc[(data.shape[0] - 4), moving_average_index] = moving_average
                    else:
                        wheights = np.array(wheights_data['regular'])
                        moving_average = (data.iloc[i, historical_sale_index] * wheights[4]) + (
                            data.iloc[i - 1, historical_sale_index] * wheights[3]) + (
                                         data.iloc[i - 2, historical_sale_index] * wheights[2]) + (
                                         data.iloc[i - 3, historical_sale_index] * wheights[1]) + (
                                         data.iloc[i - 4, historical_sale_index] * wheights[0]) + (
                                         data.iloc[i + 1, historical_sale_index] * wheights[5]) + (
                                         data.iloc[i + 2, historical_sale_index] * wheights[6]) + (
                                         data.iloc[i + 3, historical_sale_index] * wheights[7]) + (
                                         data.iloc[i + 4, historical_sale_index] * wheights[8])
                        data.iloc[i, moving_average_index] = moving_average


                data = data.reset_index(drop=True)


                data = data['moving_average']

                cumsales = np.cumsum(Sales)
                flag = np.where(cumsales == 0, 0, 1)
                start = np.cumsum(flag)
                max1 = np.amax(start)
                PercSUP = 0.95  # percentile SUP predefined
                PercINF = 0.05  # percentile INF predefined

                Differnce = Sales - data

                SUP = (PercSUP * (max1 - 1) + 1)

                INF = (PercINF * (max1 - 1) + 1)

                Differnce = Differnce.sort_values()

                Rank = np.cumsum(flag)

                PSUP = np.array((Differnce[Rank == m.floor(SUP)]) + (SUP % 1) * (
                        np.array(Differnce[Rank == m.ceil(SUP)]) - np.array(Differnce[Rank == m.floor(SUP)])))
                PINF = np.array((Differnce[Rank == m.floor(INF)]) + (INF % 1) * (
                        np.array(Differnce[Rank == m.ceil(INF)]) - np.array(Differnce[Rank == m.floor(INF)])))


                UCL = (data + 1.2 * PSUP)
                LCL = (data + 1.2 * PINF)
                new_sales = Sales.copy()
                for i in range(len(new_sales)):
                    if new_sales[i] >= LCL[i] and new_sales[i] <= UCL[i]:
                        new_sales[i] = new_sales[i]
                    elif new_sales[i] >= UCL[i]:
                        new_sales[i] = UCL[i]
                    else:
                        new_sales[i] = LCL[i]
                df_e.drop(target_var, axis=1, inplace=True)
                df_e[target_var] = new_sales
                df_result = pd.DataFrame()
                df_result = pd.concat((df_z, df_e), axis=0, ignore_index=True)
                df_result.drop('moving_average',axis = 1, inplace = True)

        df_result['historical_sale'] = df_result['historical_sale'].round()
        df_result['historical_sale'] = df_result['historical_sale'].map(int)
        return df_result
    
    def blended_history(data1):
        data1.reset_index(drop=True, inplace=True)
        first_non_zero = data1['historical_sale'].ne(0).idxmax()
        df_e = data1[first_non_zero:]
        df_e.reset_index(drop=True, inplace=True)
        df_z = data1[:first_non_zero]
        data = df_e
        if len(data['historical_sale']) < 3:
            df_result = data1
        else:
            data.historical_sale = data.historical_sale.map(float)
            weights_data = [
            {'first':0,'regular':0.1,'last':0.1},
            {'first':0.9,'regular':0.8,'last':0.9},
            {'first':0.1,'regular':0.1,'last':0.1}]
            weights_data = pd.DataFrame(weights_data)
            historical_sale_index = data.columns.get_loc('historical_sale')  # 6
            for i in range(data.shape[0]):
                if i == 0:
                    weights = np.array(weights_data['first'])
                    moving_average = (data.iloc[0, historical_sale_index] * weights[1]) + (
                            data.iloc[1, historical_sale_index] * weights[2])
                    data.iloc[0, historical_sale_index] = moving_average

                elif i == (data.shape[0] - 1):
                    weights = np.array(weights_data['last'])
                    moving_average = (data.iloc[(data.shape[0] - 1), historical_sale_index] * weights[1]) + (
                            data.iloc[(data.shape[0] - 2), historical_sale_index] * weights[0])
                    data.iloc[(data.shape[0] - 1), historical_sale_index] = moving_average

                else:
                    weights = np.array(weights_data['regular'])
                    moving_average = (data.iloc[i, historical_sale_index] * weights[1]) + (
                            data.iloc[i - 1, historical_sale_index] * weights[0]) + (
                                             data.iloc[i + 1, historical_sale_index] * weights[2])
                    data.iloc[i, historical_sale_index] = moving_average

            df_result = pd.DataFrame()
            df_result = pd.concat((df_z, data), axis=0, ignore_index=True)
        df_result['historical_sale'] = df_result['historical_sale'].map(float)
        df_result['historical_sale'] = df_result['historical_sale'].map(int)

        return df_result
    
    if not blended_smoothing:

        try:
            smoothing_type = [i for i in smoothing_rules.keys() if int(model) in smoothing_rules[i]][0]
        except:
            smoothing_type = 'Moving Average'
        if window_length < 2:
            smoothing_type = 'No Smoothing'
    else:
        smoothing_type = 'Blended History'

    if smoothing_type == 'Moving Average':
        #train_data = moving_average_smoothing(train_data.copy(), window_length)
        train_data['historical_sale'] = train_data.historical_sale.rolling(window=window_length, min_periods=1).mean().map(int)
        train_data['historical_sale'] = train_data['historical_sale'].round()
        train_data['historical_sale'] = train_data['historical_sale'].map(int)
        
    elif smoothing_type == '9-point Weighted Average':
        train_data = wheighted_moving_average_smoothing(train_data.copy())
        
    elif smoothing_type == 'Blended History':
        train_data = blended_history(train_data.copy())

    test_data = data.iloc[-test_size:].copy()
    test_data.reset_index(drop=True, inplace=True)
    if str(model) == '1':
        yhat = lstm(train_data, test_size)
        yhat = np.array(yhat).flatten()
        yhat = yhat.astype(int)
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:, 'forecast'] = yhat
        data = pd.concat([train_data, test_data,fold_data], ignore_index=True)
    elif str(model) =='2':
        yhat = croston(train_data, test_size)
        yhat = np.round(yhat, 0)
        yhat = yhat.astype(int)
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:, 'forecast'] = yhat
#         test_data['forecast'] = yhat
        data = pd.concat([train_data, test_data, fold_data])
    elif str(model) == '3':
        yhat = arima(train_data, test_size)
        yhat = np.round(yhat, 0)
        yhat = yhat.astype(int)
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:, 'forecast'] = yhat
#         test_data['forecast'] = yhat
        data = pd.concat([train_data, test_data, fold_data])
    elif str(model) == '4':
        temp = train_data.copy()
        temp.reset_index(drop=True, inplace=True)
        first_non_zero = temp['historical_sale'].ne(0).idxmax()
        temp = temp[first_non_zero:]
        try:
            
            model = SARIMA(temp['historical_sale'].astype(int))
            model.fit()
            yhat = model.forecast(test_size)
            yhat = np.array(yhat)
            yhat = np.round(yhat, 0)
            yhat = yhat.astype(int)
        except:
            yhat = np.zeros(test_size)
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:, 'forecast'] = yhat
#         test_data['forecast'] = yhat
        data = pd.concat([train_data, test_data, fold_data])
    elif str(model) == '5':
        yhat = mlp(train_data, test_size)
        yhat = yhat.astype(int)
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:,'forecast'] = yhat
        data = pd.concat([train_data, test_data,fold_data], ignore_index=True)
    elif str(model) == '6':
        model_train_data = train_data[['period', 'historical_sale']].copy()
        model_train_data['period'] = pd.to_datetime(model_train_data['period'].map(lambda x:str(x).replace('M0', '-') + '-01'))
        model_train_data.rename(columns={'period':'ds', 'historical_sale':'y'}, inplace=True)
        model_test_data = test_data[['period', 'historical_sale']].copy()
        model_test_data['period'] = pd.to_datetime(model_test_data['period'].map(lambda x:str(x).replace('M0', '-') + '-01'))
        model_test_data.rename(columns={'period':'ds', 'historical_sale':'y'}, inplace=True)
        model = Prophet()
        model.fit(model_train_data)
#         future = model.make_future_dataframe(periods=test_size, freq='M')
        preds = model.predict(model_test_data[['ds']])
        preds = preds[['ds', 'yhat']].copy()
        preds['yhat'] = preds['yhat'].map(int)
        preds['yhat'] = preds['yhat'].map(lambda x:0 if x < 0 else x)
        preds['ds'] = preds['ds'].map(lambda x:str(x.year) + 'M0' + ['0' + str(x.month) if x.month < 10 else str(x.month)][0])
        test_data['forecast'] = test_data['period'].map(lambda x:preds[preds['ds'] == x]['yhat'].values[0])
        data = pd.concat([train_data, test_data, fold_data])
    elif str(model) == '10':
        yhat = tbats_sktime(train_data, test_size)
        yhat = np.round(yhat, 0)
        yhat = yhat.astype(int)
        yhat = np.array(yhat)
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:, 'forecast'] = yhat
#         test_data['forecast'] = yhat
        data = pd.concat([train_data, test_data, fold_data])
    elif str(model) == '11':
        model_data = get_dates(train_data.copy())
        yhat = randomForest(model_data, test_size, 'M')
        yhat = np.round(yhat, 0)
        yhat = yhat.astype(int)
        yhat = np.array(yhat).flatten()
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:, 'forecast'] = yhat
#         test_data['forecast'] = yhat
        data = pd.concat([train_data, test_data, fold_data])
        data.sort_values('period', inplace=True)
    elif str(model) == '12':
        model_data = get_dates(train_data.copy())
        yhat = xgBoost_simple(model_data, test_size, 'M')
        yhat = np.round(yhat, 0)
        yhat = yhat.astype(int)
        yhat = np.array(yhat).flatten()
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:, 'forecast'] = yhat
#         test_data['forecast'] = yhat
        data = pd.concat([train_data, test_data, fold_data])
        data.sort_values('period', inplace=True)
    elif str(model) == '15':
        window_length_ma_6 = 6
        avg_values = []
        last_history_values = list(train_data.iloc[-window_length_ma_6:]['historical_sale'])
        for i in range(test_size):
            if i!=0:
                sum_of_fcst = reduce(lambda a,b : a + b, last_history_values[i:])
            else:
                sum_of_fcst = reduce(lambda a,b : a + b, last_history_values)
            last_history_values.append(sum_of_fcst // window_length_ma_6)
            avg_values.append(sum_of_fcst // window_length_ma_6)
#             test_data.loc[i, 'forecast'] = sum_of_fcst // window_length_ma_6
        avg_values = np.array(avg_values)
        avg_values = np.where(avg_values < 0, 0, avg_values)
        test_data['forecast'] = avg_values
        data = pd.concat([train_data, test_data,fold_data], ignore_index=True)
    elif str(model) == '16':
        yhat = auto_ets(train_data,test_size,'M')
        yhat = yhat.astype(int)
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:, 'forecast'] = yhat
#         test_data['forecast'] = yhat
        data = pd.concat([train_data, test_data, fold_data])
    elif str(model) == '17':
        yhat = naive_forecast(train_data, test_size, 'M')
        yhat = yhat.astype(int)
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:, 'forecast'] = yhat
#         test_data['forecast'] = yhat
        data = pd.concat([train_data, test_data, fold_data])
    elif str(model) == '18':
        yhat = linear_regression(train_data, test_size)
        yhat = np.round(yhat, 0)
        yhat = yhat.astype(int)
        yhat = np.array(yhat).flatten()
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:, 'forecast'] = yhat
#         test_data['forecast'] = yhat
        data = pd.concat([train_data, test_data, fold_data])
    elif str(model) == '19':
        yhat = bagged_ets(train_data,test_size,'M')
        yhat = yhat.astype(int)
        yhat = np.array(yhat).flatten()
        yhat = np.where(yhat < 0, 0, yhat)
        test_data.loc[:, 'forecast'] = yhat
#         test_data['forecast'] = yhat
        data = pd.concat([train_data, test_data,fold_data], ignore_index=True)
    elif str(model) == '20':
        window_length_ma_12 = 12
        avg_values = []
        last_history_values = list(train_data.iloc[-window_length_ma_12:]['historical_sale'])
        for i in range(test_size):
            if i!=0:
                sum_of_fcst = reduce(lambda a,b : a + b, last_history_values[i:])
            else:
                sum_of_fcst = reduce(lambda a,b : a + b, last_history_values)
            last_history_values.append(sum_of_fcst // window_length_ma_12)
            avg_values.append(sum_of_fcst // window_length_ma_12)
#             test_data.loc[i, 'forecast'] = sum_of_fcst // window_length_ma_12
        avg_values = np.array(avg_values)
        avg_values = np.where(avg_values < 0, 0, avg_values)
        test_data['forecast'] = avg_values
        data = pd.concat([train_data, test_data,fold_data], ignore_index=True)
#     elif str(model) == '22':
#         model_id = data.model_id.iloc[0]
#         model_key_split = model_id.split('_')
#         alpha = float(model_key_split[1])
#         beta = float(model_key_split[2])
#         gamma = float(model_key_split[3])
#         yhat = ets_es(train_data,forecast_length, 'M', alpha, beta, gamma)
#         yhat = np.array(yhat).flatten()
#         yhat = yhat.astype(int)
#         yhat = np.where(yhat < 0, 0, yhat)
#         test_data.loc[:, 'forecast'] = yhat
#         data = pd.concat([train_data, test_data,fold_data], ignore_index=True)
#         data['model_id'] = '22'
#     elif str(model) == '23': 
#         model_id = data.model_id.iloc[0]
#         model_key_split = model_id.split('_')
#         window_length_static = int(model_key_split[1])
#         test_data_temp = train_data[-window_length_static:]
#         x = test_data_temp['historical_sale'].mean()
#         x = np.where(x < 0, 0, x)
#         test_data['forecast'] = x
#         data = pd.concat([train_data, test_data,fold_data], ignore_index=True)
#         if len(data[data['model_id'].str.contains('_s')]) >0:
#             data['model_id'] = '23_s'
#         else:
#             data['model_id'] = '23'
    return data

# COMMAND ----------

@pandas_udf(dp3_data.schema, PandasUDFType.GROUPED_MAP)
def system_445(data):
    data.sort_values('period', inplace=True)
    data.reset_index(drop=True, inplace=True)
    model = data['key'].map(lambda x:str(x).split('@')[-2]).drop_duplicates().values[0]
    fold = int(data['key'].map(lambda x:str(x).split('@')[-1]).drop_duplicates().values[0])
    fold_data = pd.DataFrame()
    if fold != 0:
        fold_data = data.iloc[-fold:].copy()
        data = data.iloc[:-fold].copy()
    train_data = data.iloc[:-test_size].copy()

    test_data = data.iloc[-test_size:].copy()
    test_data.reset_index(drop=True, inplace=True)

    test_data['Month'] = pd.to_numeric(test_data['period'].map(lambda x:str(x)[-2:]))
    
    test_data['forecast'] = test_data.apply(lambda x: x['forecast'] * (15 / 13) if x['Month'] % 3 == 0 else x['forecast'] * (12 / 13) , axis=1)

    mask = ~test_data['forecast'].isnull()
    test_data.loc[mask, 'forecast'] = test_data.loc[mask, 'forecast'].map(lambda x:np.round(x, 0))
    test_data.loc[mask, 'forecast'] = test_data.loc[mask, 'forecast'].map(int)

    train_data = train_data[data.columns]
    test_data = test_data[data.columns]
    data = pd.concat([train_data, test_data, fold_data])
    data.sort_values('period', inplace=True)
    return data

# COMMAND ----------

outSchema = StructType([StructField('key',StringType(),True),
                        StructField('period',StringType(),True),
                        StructField('historical_sale',FloatType(),True),
                        StructField('forecast',FloatType(),True),
#                         StructField('model',StringType(),True),
                       ])

# COMMAND ----------

@pandas_udf(dp3_data.schema, PandasUDFType.GROUPED_MAP)
def create_forecast_allocation_data_frame(df_in):
    global forecast_length
    df_in.sort_values(['period'],inplace = True)
    dfx = df_in.iloc[:-forecast_length]
    df_temp = df_in.iloc[df_in.shape[0] - forecast_length:]
    yhat = trend_regression(dfx, forecast_length)
    historical_sum = df_temp['allocations'].sum()
    forecast_sum = df_temp['forecast'].sum()
    df_temp['allocations'] = df_temp['allocations'].div(historical_sum)
    df_temp['forecast'] = df_temp['allocations'].mul(forecast_sum)
    yhat_mean = np.average(yhat)
    df_temp['trend'] = yhat
    df_temp['trend'] = (df_temp['trend'] - yhat_mean)/df_temp['trend']
    df_temp['forecast'] = (1 + df_temp['trend']) * df_temp['forecast']
    df_temp['forecast'] = df_temp['forecast'].round()
    df_temp[df_temp['forecast'] < 0]['forecast'] = 0
    dfx = pd.concat(([dfx, df_temp]), axis=0, ignore_index=True)
    dfx = dfx.drop(['allocations','org_unit_id','product_id','channel_id','allocated_period','trend','fold'], axis=1)
    return dfx

# COMMAND ----------

# @pandas_udf(dp3_data.schema, PandasUDFType.GROUPED_MAP)
# def create_forecast_allocation_data_frame(df_in):
#     global forecast_length
#     df_in.sort_values(['period'],inplace = True)
#     dfx = df_in.iloc[:-forecast_length]
#     df_temp = df_in.iloc[df_in.shape[0] - forecast_length:]
#     yhat = trend_regression(dfx, forecast_length)
#     historical_sum = df_temp['allocations'].sum()
#     forecast_sum = df_temp['forecast'].sum()
#     df_temp['allocations'] = df_temp['allocations'].div(historical_sum)
#     df_temp['forecast'] = df_temp['allocations'].mul(forecast_sum)
#     yhat_mean = yhat[0]
#     df_temp['trend'] = yhat
#     df_temp['trend'] = (df_temp['trend'] - yhat_mean)/df_temp['trend']
#     df_temp['forecast'] = (1 + df_temp['trend']) * df_temp['forecast']
#     df_temp['forecast'] = df_temp['forecast'].round()
#     df_temp[df_temp['forecast'] < 0]['forecast'] = 0
#     dfx = pd.concat(([dfx, df_temp]), axis=0, ignore_index=True)
#     dfx = dfx.drop(['allocations','org_unit_id','product_id','channel_id','allocated_period','trend'], axis=1)
#     return dfx

# COMMAND ----------

# @pandas_udf(dp3_data.schema, PandasUDFType.GROUPED_MAP)
# def create_forecast_allocation_data_frame(df_in):
#     global forecast_length
#     df_in.sort_values(['period'],inplace = True)
#     dfx = df_in.iloc[:-forecast_length]
#     df_temp = df_in.iloc[df_in.shape[0] - forecast_length:]
#     historical_sum = df_temp['allocations'].sum()
#     forecast_sum = df_temp['forecast'].sum()
#     df_temp['allocations'] = df_temp['allocations'].div(historical_sum)
#     df_temp['forecast'] = df_temp['allocations'].mul(forecast_sum)
#     df_temp['forecast'] = df_temp['forecast'].round()
#     dfx = pd.concat(([dfx, df_temp]), axis=0, ignore_index=True)
#     dfx = dfx.drop(['allocations','org_unit_id','product_id','channel_id','allocated_period'], axis=1)
#     return dfx

# COMMAND ----------

def get_allocation(input_period):
    allocated_period = re.split('W|D|M|Q', input_period)[1]
    return allocated_period
allocation_udf = udf(get_allocation, StringType())

# COMMAND ----------

def forecast_normalization(df,df_norm):
    df_norm = df_norm.withColumn('feature_key_ref', regexp_replace('feature_key_ref', '~', '@'))
    df_norm = df_norm.withColumnRenamed('feature_key_ref','key').withColumnRenamed('feature_value','allocations').withColumnRenamed('period','allocated_period')
   
    df = df.withColumn('allocated_period',allocation_udf(df.period)).dropna(subset = ['period'])
    df = df.withColumn("org_unit_id", split(col("key"), "@").getItem(0)).withColumn("channel_id", split(col("key"), "@").getItem(1)).withColumn("product_id", split(col("key"), "@").getItem(2)).withColumn("model", split(col("key"), "@").getItem(3)).withColumn("fold", split(col("key"), "@").getItem(4))
    df = df.withColumn('key', concat_ws("@",df.org_unit_id,df.channel_id,df.product_id))#.select(col('key'),col('period'),col('allocated_period'),col('historical_sale'),col('forecast'),col('model'),col('fold'))
    df_in = df.join(df_norm, (df.key == df_norm.key) & (df.allocated_period == df_norm.allocated_period)).drop(df_norm.key).drop(df_norm.allocated_period).drop(df_norm.domain_id).drop(df_norm.run_id)
    df_in = df_in.withColumn('key', concat_ws("@",df_in.key,df_in.model,df_in.fold))
    df_in = df_in.withColumn('forecast',F.col('forecast').cast(FloatType())).withColumn('historical_sale',F.col('historical_sale').cast(FloatType())).withColumn('allocations',F.col('allocations').cast(FloatType())).distinct()
    df_in_m = df_in.filter(col('model').isin(fcast_norm_models)).drop(df_in.model)
    df_out = df_in_m.groupby('key').apply(create_forecast_allocation_data_frame)
#     df_out = df_out.withColumn('segmenatation',F.lit(''))
    return df_out

# COMMAND ----------

dp3_data_res = dp3_data.repartition('key').groupBy('key').apply(model_training)
schema_dp3_res = dp3_data_res.schema
dp3_data_res = dp3_data_res.toPandas()
dp3_data_res = spark.createDataFrame(data=dp3_data_res, schema=schema_dp3_res)

if model_select == 'dynamic':
    
    dp3_data_res_m = dp3_data_res.withColumn("model", split(col("key"), "@").getItem(3))
    
    dp3_data_res_m = dp3_data_res_m.filter(col('model').isin(s445_models)).drop(dp3_data_res_m.model)
    
    dp3_data_res_445 = dp3_data_res_m.repartition('key').groupBy('key').apply(system_445)
    dp3_data_res_445 = dp3_data_res_445.withColumn('key', concat(col('key'), lit('@s')))
    dp3_data_final = dp3_data_res.union(dp3_data_res_445)
    
    dp3_data_res_fn = forecast_normalization(dp3_data_res,df_norm)
    dp3_data_res_fn = dp3_data_res_fn.withColumn('key', concat(col('key'), lit('@f')))
    dp3_data_final = dp3_data_final.union(dp3_data_res_fn)

elif model_select == 'static':
    if system_445:
        dp3_data_res_445 = dp3_data_res.repartition('key').groupBy('key').apply(system_445)
        dp3_data_res_445 = dp3_data_res_445.withColumn('key', concat(col('key'), lit('@s')))
        dp3_data_final = dp3_data_res.union(dp3_data_res_445)
    elif fcast_norm:
        dp3_data_res_fn = forecast_normalization(dp3_data_res,df_norm)
        dp3_data_res_fn = dp3_data_res_fn.withColumn('key', concat(col('key'), lit('@f')))
        dp3_data_final = dp3_data_res.union(dp3_data_res_fn)
    else:
        dp3_data_final = dp3_data_res.withColumn('key', concat(col('key'), lit('@n')))

dp3_data_final = dp3_data_final.withColumn('org_unit_id', split(col('key'), '@')[0])
dp3_data_final = dp3_data_final.withColumn('channel_id', split(col('key'), '@')[1])
dp3_data_final = dp3_data_final.withColumn('product_id', split(col('key'), '@')[2])
dp3_data_final = dp3_data_final.withColumn('model', split(col('key'), '@')[3])
dp3_data_final = dp3_data_final.withColumn('fold', split(col('key'), '@')[4])
dp3_data_final = dp3_data_final.withColumn('post_forecast_adjustment', split(col('key'), '@')[5])
dp3_data_final = dp3_data_final.withColumn('mae', lit(0.0))
dp3_data_final = dp3_data_final.withColumn('me', lit(0.0))
dp3_data_final = dp3_data_final.withColumn('phase', lit(0.0))
dp3_data_final = dp3_data_final.withColumn('score', lit(0.0))
dp3_data_final = dp3_data_final.withColumn('rank_mae', lit(0.0))
dp3_data_final = dp3_data_final.withColumn('rank_me', lit(0.0))
dp3_data_final = dp3_data_final.withColumn('rank_phase', lit(0.0))
dp3_data_final = dp3_data_final.withColumn('score_mae', lit(0.0))
dp3_data_final = dp3_data_final.withColumn('score_me', lit(0.0))
dp3_data_final = dp3_data_final.withColumn('score_phase', lit(0.0))
dp3_data_final = dp3_data_final.select(concat(dp3_data_final.org_unit_id, lit('@'), dp3_data_final.channel_id, lit('@'), dp3_data_final.product_id).alias('key'), 'period', 'historical_sale', 'forecast', 'model', 'fold', 'post_forecast_adjustment', 'mae', 'me', 'phase', 'rank_mae', 'rank_me', 'rank_phase', 'score_mae', 'score_me', 'score_phase', 'score')

# COMMAND ----------

def phase_check(y_true, y_pred):
    df = pd.DataFrame(data={'true': y_true, 'pred': y_pred})
    df['l_true'] = df['true'].shift(1)
    df['l_pred'] = df['pred'].shift(1)
    df.dropna(inplace=True)
    df['true_pct'] = (df['l_true'] - df['true']) / df['true']
    df['pred_pct'] = (df['l_pred'] - df['pred']) / df['pred']
    df['abs_pct'] = np.abs((df['true_pct'] - df['pred_pct']) / (df['true_pct'] + df['pred_pct']))
    phase_value = np.average(df['abs_pct'])
    if phase_value < 0.6:
        return 1
    else:
        return 0

# COMMAND ----------

@pandas_udf(dp3_data_final.schema, PandasUDFType.GROUPED_MAP)
def best_model_selection(data):
    data = data[~data['forecast'].isna()]
    data['post_forecast_adjustment'] = data['post_forecast_adjustment'].fillna('n')
    data = data.groupby(by=['key', 'model', 'fold', 'post_forecast_adjustment']).agg({'historical_sale': list, 'forecast':list}).reset_index()
    if best_model_selection_metric=='mae':
        data['mae'] = data.apply(lambda x:np.average(np.abs(np.array(x['historical_sale']) - np.array(x['forecast']))), axis=1)
    else:
        data['mae'] = data.apply(lambda x: np.average(np.abs((np.array(x['historical_sale']) - np.array(x['forecast']))/np.array(x['historical_sale']))) * 100)
    data['me'] = data.apply(lambda x:np.average(np.array(x['historical_sale']) - np.array(x['forecast'])), axis=1)
    data['phase'] = data.apply(lambda x: phase_check(x['historical_sale'], x['forecast']), axis=1)
    data = data.groupby(by=['key', 'model', 'post_forecast_adjustment']).agg({'mae': np.average, 'me': np.average, 'phase': np.sum}).reset_index()
    
    data['rank_mae'] = data['mae'].rank()
    data['rank_me'] = data['me'].map(lambda x:np.abs(x)).rank()
    data['rank_phase'] = data['phase'].max() - data['phase'] + 1
    
    data['score_mae'] = data['rank_mae'] / (np.sum(data['rank_mae']) + 1)
    data['score_me'] = data['rank_me'] / (np.sum(data['rank_me']) + 1)
    data['score_phase'] = data['rank_phase'] / (np.sum(data['rank_phase']) + 1)
    
    data['score'] = data['score_mae'] * error_weight + data['score_me'] * me_weight + data['score_phase'] * phase_weight
    
    data.sort_values('score', inplace=True)
#     data = data.head(1)

    data['period'] = None
    data['historical_sale'] = None
    data['forecast'] = None
    data['fold'] = None
    return data

# COMMAND ----------

dp3_data_final = dp3_data_final.repartition('key').groupBy('key').apply(best_model_selection)

# COMMAND ----------

dp3_data_final = dp3_data_final.withColumnRenamed('mae','metric_value').withColumnRenamed('me','mean_error').withColumnRenamed('phase','phasing').withColumnRenamed('rank_mae','metric_value_rank').withColumnRenamed('rank_me','mean_error_rank').withColumnRenamed('rank_phase','phasing_rank').withColumnRenamed('score_mae','metric_value_score').withColumnRenamed('score_me','mean_error_score').withColumnRenamed('score_phase','phasing_score')

dp3_data_final = dp3_data_final.withColumn('run_id',lit(run_id))
dp3_data_final = dp3_data_final.withColumn('domain_id',lit(domain_id))
dp3_data_final = dp3_data_final.withColumn('model_id',concat_ws("_",dp3_data_final.model,dp3_data_final.post_forecast_adjustment)).drop(dp3_data_final.period).drop(dp3_data_final.historical_sale).drop(dp3_data_final.forecast).drop(dp3_data_final.model).drop(dp3_data_final.fold).drop(dp3_data_final.post_forecast_adjustment)

# COMMAND ----------

# dp3_data_final.display()

# COMMAND ----------

# DBTITLE 1,writing as a parquet
# filename = f'model_selection_scores_{run_id}.parquet'
# folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_model_selection_scores.parquet"
if ingestion == 1:
#     dp3_data_final.repartition(128).write.mode('overwrite').parquet(folderPath)
    dp3_data_final.repartition(128).write.mode('overwrite').parquet(working_path)
else:
    print("Ingestion not done")
    dp3_data_final_pd = dp3_data_final.toPandas()

# COMMAND ----------

# dp3_data_final_pd.display()
