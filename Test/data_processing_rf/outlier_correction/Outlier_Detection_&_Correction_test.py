# Databricks notebook source
import sys
import os
import pandas as pd
import numpy as np
import random
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import logging
import pandas as pd
import pandas as pd
from pandas import DataFrame
from numpy.random import RandomState
from sklearn.base import BaseEstimator, TransformerMixin
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType, DoubleType
from pyspark.sql.functions import when, col, size, split
from pyspark.sql.functions import pandas_udf, PandasUDFType 
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import first
from pyspark.sql.functions import col, pandas_udf
import math as m
import traceback
import warnings
warnings.filterwarnings('ignore')
from pyspark.sql.types import LongType
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

ingestion_flag=1

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

dbutils.widgets.text('run_id', '', 'run_id')
dbutils.widgets.text('domain_id', '', 'domain_id')
dbutils.widgets.text('outlier_detection_choice', '', 'outlier_detection_choice')
dbutils.widgets.text('forecast_length', '', 'forecast_length')
dbutils.widgets.text('run_mode', '', 'run_mode')
run_id=str(dbutils.widgets.get("run_id"))
domain_id= dbutils.widgets.get("domain_id")
outlier_detection_choice = dbutils.widgets.get("outlier_detection_choice")
forecast_length= dbutils.widgets.get("forecast_length")
forecast_length = int(forecast_length)
run_mode= dbutils.widgets.get("run_mode")

# COMMAND ----------

working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_I5_run_data.parquet"
run_data_I5_stage = spark.read.format('parquet').load(working_path,inferSchema=True,header=True)
run_data_I5_stage = run_data_I5_stage.withColumn('key',concat_ws('-',run_data_I5_stage.org_unit_id,run_data_I5_stage.channel_id,run_data_I5_stage.product_id))
run_data_I5_stage = run_data_I5_stage.withColumn('forecast',F.col('forecast').cast(IntegerType()))  #Type casting from string to integer as per schema definition

# COMMAND ----------

def run_parameter_data(run_id,url,properties):
    
    parameter = spark.read.jdbc(url=url,table=f"(select parameter_id,run_value from run_parameter where run_id = '{run_id}') as run_value",properties=properties)
    parameter = parameter.toPandas()
   
    parameter = dict(zip(parameter['parameter_id'],parameter['run_value']))
    
    return parameter

# COMMAND ----------

outSchema = StructType([StructField('domain_id',StringType(),True),
                        StructField('run_id',StringType(),True),
                        StructField('run_state',StringType(),True),
                        StructField('org_unit_id',StringType(),True),
                        StructField('channel_id',StringType(),True),
                        StructField('product_id',StringType(),True),
                        StructField('period',StringType(),True),
                        StructField('historical_sale',IntegerType(),True),
                        StructField('forecast',IntegerType(),True),
                        StructField('segmentation',StringType(),True),
                        StructField('model_id',StringType(),True),
                        StructField('promotion',StringType(),True),
                        StructField('key',StringType(),True),
                       ])

# COMMAND ----------

# MAGIC %md
# MAGIC Missing value Object Definition

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValues:

    def __init__(self):
        return

    def check_null_values(self, df, column):
     
        df = df[column]
        null_values_count = df.isnull().sum()
        return null_values_count

    def fill_null_values(self, df, column, method="zero"):
#         df = df.toPandas()
        df[column] = df[column].apply(pd.to_numeric, errors='coerce')
        if method == 'mean':
            df[column] = df[column].fillna(df[column].mean())
        elif method == 'median':
            df[column] = df[column].fillna(df[column].median())
        elif method == 'mode':
            df[column] = df[column].fillna(df[column].mode())
        elif method == 'zero':
            df[column] = df[column].fillna(0)
        else:
            pass
            # print("invalid imputation method selected")
        
        return df

# COMMAND ----------

@F.pandas_udf(outSchema, F.PandasUDFType.GROUPED_MAP)
def missing_values_treatment(data: DataFrame):
    column = 'historical_sale'
    #sort with period
    data.sort_values(['period'], inplace=True)
    
    if run_mode == 'poc':
        data, data_test = data[0:-forecast_length], data[-forecast_length:data.shape[0]]
    
    miss_value = MissingValues()
    if miss_value.check_null_values(data, column) > 0:
        data = miss_value.fill_null_values(data, column)
    if run_mode == 'poc':
        data = pd.concat([data, data_test], axis=0, ignore_index=True)
    
    return data

# COMMAND ----------

# MAGIC %md
# MAGIC ZSCORE function definition

# COMMAND ----------

@F.pandas_udf(outSchema, F.PandasUDFType.GROUPED_MAP)
def zscore_outlier_correction(data: pd.DataFrame):
    #sorting on period
    data.sort_values(['period'],inplace=True)
    if run_mode == 'poc':
        data, data_test = data[0:-forecast_length], data[-forecast_length:data.shape[0]]
    
    df_x = data
    target_var = 'historical_sale'
    ZOUTLIER = 4
    first_non_zero = df_x['historical_sale'].ne(0).idxmax()
    df_e = df_x[first_non_zero:]
    df_e.reset_index(drop=True, inplace=True)
    df_z = df_x[:first_non_zero]
    df_e[target_var] = df_e[target_var].astype(float)
    stdev = df_e[target_var].std(ddof=0)
    mn = df_e[target_var].mean()
    threshold = ZOUTLIER
    df_e['zscore'] = (df_e[target_var] - mn) / stdev
    df_e['target_var1'] = df_e[target_var]
    for i in range(len(df_e)):
        if (df_e.at[i,'zscore'] > threshold):
            df_e.at[i, 'target_var1'] = mn + (threshold * stdev)
        if (df_e.at[i,'zscore'] < (threshold * -1)):
            df_e.at[i, 'target_var1'] = mn - (threshold * stdev)
    df_e['historical_sale'] = df_e['target_var1']
    df_e.drop(['zscore','target_var1'], axis=1, inplace=True)
    df_result = pd.DataFrame()
    df_result = pd.concat((df_z, df_e), axis=0, ignore_index=True)
    df_result.historical_sale = df_result.historical_sale.round()
    df_result['historical_sale'] = df_result['historical_sale'].map(int)
    
    if run_mode == 'poc':
        df_result = pd.concat([df_result, data_test], axis=0, ignore_index=True)
    
    return df_result

# COMMAND ----------

# MAGIC %md
# MAGIC Moving Average Outlier

# COMMAND ----------

@F.pandas_udf(outSchema, F.PandasUDFType.GROUPED_MAP)
def moving_average_outliers(df: DataFrame):
    target_var = 'historical_sale'
    df['moving_average'] = 0.0
    df = df.sort_values(['org_unit_id', 'channel_id', 'product_id', 'period'])
    df.reset_index(drop=True,inplace=True)
    
    if run_mode == 'poc':
        df, df_test = df[0:-forecast_length], df[-forecast_length:df.shape[0]]
    
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
        

        #print(mov_avg)
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
        #print(wheights_data)
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
            #df_e['lcl'] = LCL
            #df_e['ucl'] = UCL
            #df_e['moving_average'] = data
            df_result = pd.DataFrame()
            df_result = pd.concat((df_z, df_e), axis=0, ignore_index=True)
    
    
    if run_mode == 'poc':
        df_result = pd.concat([df_result, df_test], axis=0, ignore_index=True)
        
    df_result.drop('moving_average',axis = 1, inplace = True)
    df_result['historical_sale'] = df_result['historical_sale'].round()
    df_result['historical_sale'] = df_result['historical_sale'].map(int)
    return df_result

# COMMAND ----------

try:
    

    run_data_I5_stage = run_data_I5_stage.sort('org_unit_id','channel_id','product_id','period')

    df_missing = run_data_I5_stage.groupby("key").apply(missing_values_treatment)

    if outlier_detection_choice == 'ZScore': 
    
        df_zscore_data = df_missing.groupby("key").apply(zscore_outlier_correction)
        df_temp = df_zscore_data.drop('key')
        df_final = df_temp.replace(to_replace='I5',value='DP1',subset=['run_state'])
        print("Data-Processing zscore successful")
    
    elif outlier_detection_choice == 'MovingAverage':
    
        df_moving_average = df_missing.groupby("key").apply(moving_average_outliers)
        df_temp = df_moving_average.drop('key')
        df_final = df_temp.replace(to_replace='I5',value='DP1',subset=['run_state'])
        print("Data-Processing Moving Average successful")
        
    elif outlier_detection_choice == 'No':
        
        df_temp = df_missing.drop('key')
        df_final = df_temp.replace(to_replace='I5',value='DP1',subset=['run_state'])
        print("Data-Processing No outlier successful")

        #
        
    df_final = df_final.select('domain_id', 'run_id','run_state','org_unit_id','channel_id','product_id','period','historical_sale','forecast','segmentation','model_id','promotion')
    
    if ingestion_flag==1:
        
        working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP1_run_data.parquet"
        df_final.repartition(1).write.mode('overwrite').parquet(working_path)        
    else:
        
        df_final = df_final.toPandas()
        #if outlier detection moving average or not
        print("demo for moving average")
        df_final.display()
except:
    traceback.print_exc()
    10/0    
    


# COMMAND ----------


