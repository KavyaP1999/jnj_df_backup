# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, lower, col, split
from pyspark.sql import functions as f
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
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
import warnings
import traceback
warnings.filterwarnings('ignore')
from pyspark.sql.types import LongType
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

dbutils.widgets.text("run_id", "","")
run_id = dbutils.widgets.get("run_id")
print ("run_id:",run_id)

dbutils.widgets.text("domain_id", "","")
domain_id = dbutils.widgets.get("domain_id")
print ("domain_id:",domain_id)

# COMMAND ----------

# dbutils.widgets.text("forecast_length", "","")
# forecast_length = dbutils.widgets.get("forecast_length")
# forecast_length = int(forecast_length)

# dbutils.widgets.text("run_mode", "","")
# run_mode = dbutils.widgets.get("run_mode")
# run_mode = int(run_mode)

# COMMAND ----------

dbutils.widgets.text("org_unit_input_level", "","")
org_unit_input_level = dbutils.widgets.get("org_unit_input_level")
org_unit_input_level = int(org_unit_input_level)

dbutils.widgets.text("channel_input_level", "","")
channel_input_level = dbutils.widgets.get("channel_input_level")
channel_input_level = int(channel_input_level)

dbutils.widgets.text("product_input_level", "","")
product_input_level = dbutils.widgets.get("product_input_level")
product_input_level = int(product_input_level)

print("org_unit_input_level:",org_unit_input_level)
print("channel_input_level:",channel_input_level)
print("product_input_level:",product_input_level)

# COMMAND ----------

dbutils.widgets.text("org_unit_forecast_level", "","")
org_unit_forecast_level = dbutils.widgets.get("org_unit_forecast_level")
org_unit_forecast_level = int(org_unit_forecast_level)

dbutils.widgets.text("channel_forecast_level", "","")
channel_forecast_level = dbutils.widgets.get("channel_forecast_level")
channel_forecast_level = int(channel_forecast_level)

dbutils.widgets.text("product_forecast_level", "","")
product_forecast_level = dbutils.widgets.get("product_forecast_level")
product_forecast_level = int(product_forecast_level)

print("org_unit_forecast_level:",org_unit_forecast_level)
print("channel_forecast_level:",channel_forecast_level)
print("product_forecast_level:",product_forecast_level)

# COMMAND ----------

dbutils.widgets.text("org_unit_output_level", "","")
org_unit_output_level = dbutils.widgets.get("org_unit_output_level")
org_unit_output_level = int(org_unit_output_level)

dbutils.widgets.text("channel_output_level", "","")
channel_output_level = dbutils.widgets.get("channel_output_level")
channel_output_level = int(channel_output_level)

dbutils.widgets.text("product_output_level", "","")
product_output_level = dbutils.widgets.get("product_output_level")
product_output_level = int(product_output_level)

print("org_unit_forecast_level:",org_unit_output_level)
print("channel_forecast_level:",channel_output_level)
print("product_forecast_level:",product_output_level)

# COMMAND ----------

dbutils.widgets.text("period_input_level", "","")
period_input_level = dbutils.widgets.get("period_input_level")


dbutils.widgets.text("period_forecast_level", "","")
period_forecast_level = dbutils.widgets.get("period_forecast_level")


dbutils.widgets.text("period_output_level", "","")
period_output_level = dbutils.widgets.get("period_output_level")

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

#%run /Shared/Test/Databasewrapper_py_test

# COMMAND ----------

rounded_schema = StructType([StructField('domain_id',StringType(),True),
                        StructField('org_unit_id',StringType(),True),
                        StructField('channel_id',StringType(),True),
                        StructField('product_id',StringType(),True),
                        StructField('period',StringType(),True),
                        StructField('historical_sale',IntegerType(),True),
                        StructField('forecast',IntegerType(),True),
                        StructField('model_id',StringType(),True),
                        StructField('promotion',StringType(),True),
                        StructField('segmentation',StringType(),True)
                       ])

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

#proportion = spark.read.jdbc(url=url, table="proportion", properties=properties)
# run_data = spark.read.jdbc(url=url, table="run_data", properties=properties)
future_discontinue_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM future_discontinue_master WHERE domain_id = '{domain_id}') AS rd", properties = properties)
# future_discontinue_master = future_discontinue_master.toPandas()

# COMMAND ----------

df_hierarchy = spark.read.jdbc(url=url, table= f"(SELECT * FROM hierarchy WHERE domain_id = '{domain_id}') AS hm", properties = properties)
df_hierarchy_new=df_hierarchy.toPandas()

# COMMAND ----------

@F.pandas_udf(rounded_schema, F.PandasUDFType.GROUPED_MAP)
def rounding_logic(df1):
    #df1 = df.copy()
    df_final = pd.DataFrame()

    hist_df1 = df1[df1['forecast'].isna()].copy()
    df1 = df1[df1['forecast'] >= 0].copy()

#     hist_df1 = df1[-df1['historical_sale'].isna()].copy()
#     df1 = df1[df1['historical_sale'].isna()].copy()
    
    df1.reset_index(drop=True, inplace=True)
#     df1["forecast_new"] = np.nan
#     df1['cum_values'] = lit(np.nan)
    df1["cum_values"] = df1["forecast"].cumsum()
    rounded = []
    
    for i in range(df1.index.start, df1.index.stop):
        vl = float(df1["cum_values"][i]) - np.sum(rounded)
        rn = np.round(vl)
        rounded.append(rn)

    df1.loc[:, "forecast"] = rounded

    df_final = pd.concat([df_final, hist_df1, df1])
        
    df_final.drop(columns=["psuedo_key", "cum_values"], inplace=True)
#     df_final.rename(columns={"forecast_new": "forecast"}, inplace=True)
    df_final = df_final[['domain_id','org_unit_id','channel_id','product_id','period','historical_sale',
                         'forecast','model_id','promotion','segmentation']]
    df_final.drop_duplicates(inplace=True)

    return df_final

# COMMAND ----------

def create_keys(df):
    #df = df1.copy()
    df["keys"] = df['org_unit_id'].map(str) + '~' + df['channel_id'].map(str) + '~' + df['product_id'].map(str)
    df.sort_values(by='keys',inplace =True)
    keys = df['keys'].unique()
    df.set_index('keys', inplace=True, drop=True)
    return df

# COMMAND ----------

def generate_hierarchy(data, suffix='product'):
    data = data[data['hierarchy_type'] == suffix]
    data = data[['hierarchy_value', 'parent_value', 'level_id']].copy()
    split_df = {}
    for i in np.unique(data.level_id):
        split_df[i] = data[data['level_id'] == i].copy()
    df = pd.DataFrame()
    df = split_df[list(split_df.keys())[-1]][['hierarchy_value']].copy()
    df.rename(columns={'hierarchy_value': f'{suffix}_{list(split_df.keys())[-1]}'}, inplace=True)
    for i in list(split_df.keys())[-2::-1]:
        data = split_df[i][['hierarchy_value', 'parent_value']].copy()
        data.rename(columns={'hierarchy_value': f'{suffix}_{i}'}, inplace=True)
        df = pd.merge(df, data, left_on=f'{suffix}_{int(i)+1}', right_on='parent_value', how='inner')
        df.drop(columns={'parent_value'}, inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# COMMAND ----------

def product_aggregator(df, Heirarchy, product_input_level, product_output_level):
    #df = df1.copy()
    product_heirarchy = np.unique(Heirarchy[Heirarchy['hierarchy_type'] == 'product']['level_id'])
    prod_heirarchy = Heirarchy[Heirarchy['hierarchy_type'] == 'product']
    if product_output_level > np.amax(product_heirarchy):
        product_output_level = np.amax(product_heirarchy)
    product_input_index = int(np.where(product_heirarchy == product_input_level)[0])
    product_output_index = int(np.where(product_heirarchy == product_output_level)[0])
    aggregate_levels = [product_input_index,product_output_index]

    suffix = 'product'
    
    hm = generate_hierarchy(prod_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{product_input_level}',f'{suffix}_{product_output_level}']]
    df = pd.merge(df, hm, left_on=['product_id'], right_on=f'{suffix}_{int(product_input_index)}', how='left')
    df.drop(columns={'product_id',f'{suffix}_{int(product_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(product_output_index)}': 'product_id'}, inplace=True)
    df['historical_sale'] = df['historical_sale'].astype(int)
    df['forecast'] = df['forecast'].astype(float)
    df = df.groupby(['domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period'])['historical_sale', 'forecast'].sum().reset_index()
    return(df)

# COMMAND ----------

def channel_aggregator(df, Heirarchy, channel_input_level, channel_output_level):
    #df = df1.copy()
    channel_heirarchy = np.unique(Heirarchy[Heirarchy['hierarchy_type'] == 'channel']['level_id'])
    chann_heirarchy = Heirarchy[Heirarchy['hierarchy_type'] == 'channel']
    if channel_output_level > np.amax(channel_heirarchy):
        channel_output_level = np.amax(channel_heirarchy)
    channel_input_index = int(np.where(channel_heirarchy == channel_input_level)[0])
    channel_output_index = int(np.where(channel_heirarchy == channel_output_level)[0])
    aggregate_levels = [channel_input_index, channel_output_index]
    suffix = 'channel'
    
    hm = generate_hierarchy(chann_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{channel_input_level}',f'{suffix}_{channel_output_level}']]
    df = pd.merge(df, hm, left_on=['channel_id'], right_on=f'{suffix}_{int(channel_input_index)}', how='left')
    df.drop(columns={'channel_id',f'{suffix}_{int(channel_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(channel_output_index)}': 'channel_id'}, inplace=True)
    df['historical_sale'] = df['historical_sale'].astype(int)
    df['forecast'] = df['forecast'].astype(float)
    df = df.groupby(['domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period'])['historical_sale', 'forecast'].sum().reset_index()
    return(df)

# COMMAND ----------

def org_unit_aggregator(df, Heirarchy, org_unit_input_level, org_unit_output_level):
    #df = df1.copy()
    org_unit_heirarchy = np.unique(Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit']['level_id'])
    org_heirarchy = Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit']
    if org_unit_output_level > np.amax(org_unit_heirarchy):
        org_unit_output_level = np.amax(org_unit_heirarchy)
    org_unit_input_index = int(np.where(org_unit_heirarchy == org_unit_input_level)[0])
    org_unit_output_index = int(np.where(org_unit_heirarchy == org_unit_output_level)[0])
    aggregate_levels = [org_unit_input_index,org_unit_output_index]
    suffix = 'org_unit'
    
    hm = generate_hierarchy(org_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{org_unit_input_level}',f'{suffix}_{org_unit_output_level}']]
    df = pd.merge(df, hm, left_on=['org_unit_id'], right_on=f'{suffix}_{int(org_unit_input_index)}', how='left')
    df.drop(columns={'org_unit_id',f'{suffix}_{int(org_unit_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(org_unit_output_index)}': 'org_unit_id'}, inplace=True)
    df['historical_sale'] = df['historical_sale'].astype(int)
    df['forecast'] = df['forecast'].astype(float)
    df = df.groupby(['domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period'])['historical_sale', 'forecast'].sum().reset_index()
    return(df)

# COMMAND ----------

def create_dataframe(df2, ratio_table):
    #df2 = df3.reset_index(drop=True)
    df2.reset_index(drop=True,inplace=True)
    ratio_table['org_unit_id'] = pd.Series(ratio_table['org_unit_id'], dtype="string")
    ratio_table['channel_id'] = pd.Series(ratio_table['channel_id'], dtype="string")
    ratio_table['product_id'] = pd.Series(ratio_table['product_id'], dtype="string")
    ratio_table['ratio'] = ratio_table['ratio'].astype('float')

    # df2 = df2.drop(['keys'], axis=1)
    #df2 = create_keys(df2)
    #df2 = df2.reset_index(drop=False)
    # dfx = pd.DataFrame()
    # for i in np.unique(df2['keys']):
    #     df_temp = df2[(df2['keys'] == i)]
    #     df_temp['historical_sale'] = df_temp['historical_sale'].astype('float')
    #     df_temp['forecast'] = df_temp['forecast'].astype('float')
    #     org_unit = str(np.unique(df_temp['org_unit_id'])[0])
    #     channel = str(np.unique(df_temp['channel_id'])[0])
    #     product = str(np.unique(df_temp['product_id'])[0])
    #     ratio = float(ratio_table[(ratio_table['org_unit_id'] == org_unit) & (ratio_table['channel_id'] == channel) & (
    #             ratio_table['product_id'] == product)]['ratio'])
    #     df_temp['forecast'] = df_temp['forecast'].multiply(ratio)
    #     #df_temp['historical_sale'] = df_temp['historical_sale'].multiply(ratio)
    #     dfx = pd.concat(([dfx, df_temp]), axis=0, ignore_index=True)
    dfx = pd.merge(df2,ratio_table[['org_unit_id','channel_id','product_id','ratio']], on =['org_unit_id','channel_id','product_id'],how ="left")
    dfx['forecast_new'] = dfx['forecast'].map(lambda x: np.nan if x == None else float(x))
    dfx['forecast_new'] = dfx['forecast_new'].map(float) * dfx["ratio"].map(float)
    dfx.drop(columns=["ratio",'forecast'],inplace=True)
    dfx.rename(columns={"forecast_new":"forecast"},inplace=True)
    return dfx

# COMMAND ----------

def product_dis_aggregator(df, Heirarchy, product_input_level, product_output_level):
    
    product_heirarchy = np.unique(Heirarchy[Heirarchy['hierarchy_type'] == 'product']['level_id'])
#     print(Heirarchy[Heirarchy['hierarchy_type'] == 'product']['level_id'])

#     print(product_heirarchy)
    prod_heirarchy = Heirarchy[Heirarchy['hierarchy_type'] == 'product']
#     print(product_input_level)
#     print(product_output_level)
#     print(np.amax(product_heirarchy))
    if product_input_level > np.amax(product_heirarchy):
        product_input_level = np.amax(product_heirarchy)
#     print(product_input_level)
#     print(product_output_level)
    product_input_index = int(np.where(product_heirarchy == product_input_level)[0])
    product_output_index = int(np.where(product_heirarchy == product_output_level)[0])
#     print('index value')
#     print(product_input_index)
#     print(product_output_index)
    aggregate_levels = product_heirarchy[product_output_index:product_input_index]
    aggregate_levels[::-1].sort()
#     for i in aggregate_levels:
#         prod_heirarchy_level = prod_heirarchy[prod_heirarchy['level_id'] == i]
#         prod_heirarchy_level = prod_heirarchy_level[['parent_value', 'hierarchy_value']]
#         d = prod_heirarchy_level.groupby(['parent_value'])['hierarchy_value'].apply(list).reset_index()
#         dictionary = dict(zip(d.parent_value, d.hierarchy_value))
#         # # print(dictionary)
#         df['product_id'] = df['product_id'].map(dictionary).to_list()
#         df = df.explode('product_id').reset_index(drop=True)
#         df['product_id'] = df['product_id'].astype(str)
#         # # print(np.unique(df['product_id']))

    suffix = 'product'
#     print(product_input_level)
#     print(product_output_level)
    hm = generate_hierarchy(prod_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{product_input_level}',f'{suffix}_{product_output_level}']]
    df = pd.merge(df, hm, left_on=['product_id'], right_on=f'{suffix}_{int(product_input_index)}', how='left')
    df.drop(columns={'product_id',f'{suffix}_{int(product_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(product_output_index)}': 'product_id'}, inplace=True)
    df['historical_sale'] = df['historical_sale'].astype(int)
    df['forecast'] = df['forecast'].astype(float)
    #df = df.groupby(['domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period'])['historical_sale', 'forecast'].sum().reset_index()
    return df

# COMMAND ----------

def channel_dis_aggregator(df, Heirarchy, channel_input_level, channel_output_level):
#     print(channel_input_level)
#     print(channel_output_level)
    channel_heirarchy = np.unique(Heirarchy[Heirarchy['hierarchy_type'] == 'channel']['level_id'])
    chann_heirarchy = Heirarchy[Heirarchy['hierarchy_type'] == 'channel']
    if channel_input_level > np.amax(channel_heirarchy):
        channel_input_level = np.amax(channel_heirarchy)
    channel_input_index = int(np.where(channel_heirarchy == channel_input_level)[0])
    channel_output_index = int(np.where(channel_heirarchy == channel_output_level)[0])
    aggregate_levels = channel_heirarchy[channel_output_index:channel_input_index]
    aggregate_levels[::-1].sort()
    suffix = 'channel'
#     print(channel_input_level)
#     print(channel_output_level)
    hm = generate_hierarchy(chann_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{channel_input_level}',f'{suffix}_{channel_output_level}']]
    print(hm)
    print(df)
    df = pd.merge(df, hm, left_on=['channel_id'], right_on=f'{suffix}_{int(channel_input_index)}', how='left')
    print('channel')
    print(df)
    df.drop(columns={'channel_id',f'{suffix}_{int(channel_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(channel_output_index)}': 'channel_id'}, inplace=True)
    df['historical_sale'] = df['historical_sale'].astype(int)
    df['forecast'] = df['forecast'].astype(float)

    return df

# COMMAND ----------

def org_unit_dis_aggregator(df, Heirarchy, org_unit_input_level, org_unit_output_level):
    Heirarchy['hierarchy_type'] = Heirarchy['hierarchy_type'].map(str)
    org_unit_heirarchy = np.unique(Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit']['level_id'])
    org_heirarchy = Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit']
    if org_unit_input_level > np.amax(org_unit_heirarchy):
        org_unit_input_level = np.amax(org_unit_heirarchy)
    org_unit_input_index = int(np.where(org_unit_heirarchy == org_unit_input_level)[0])
    org_unit_output_index = int(np.where(org_unit_heirarchy == org_unit_output_level)[0])
    print("org_unit_input",org_unit_input_index)
    print("org_unit_output",org_unit_output_index)
    print("org_unit_heirarchy:",org_unit_heirarchy)
    # aggregate_levels = org_unit_heirarchy[org_unit_input_index:org_unit_output_index]
    aggregate_levels = np.arange(int(org_unit_output_level),int(org_unit_input_level),1)
    print("aggregate_levels:", aggregate_levels)
    aggregate_levels[::-1].sort()
    print("aggregate_levels:", aggregate_levels)
    suffix = 'org_unit'
    
    hm = generate_hierarchy(org_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{org_unit_input_level}',f'{suffix}_{org_unit_output_level}']]
    df = pd.merge(df, hm, left_on=['org_unit_id'], right_on=f'{suffix}_{int(org_unit_input_index)}', how='left')
    df.drop(columns={'org_unit_id',f'{suffix}_{int(org_unit_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(org_unit_output_index)}': 'org_unit_id'}, inplace=True)
    df['historical_sale'] = df['historical_sale'].astype(int)
    df['forecast'] = df['forecast'].astype(float)
    
    return df

# COMMAND ----------

def agg_dis_agg_function(df, product_output_level, channel_output_level, org_unit_output_level, product_input_level,
                         channel_input_level, org_unit_input_level):
    df.reset_index(drop=True,inplace=True)

#     df = df1.copy()
#     print(product_output_level)
#     print(product_input_level)
    df['product_id'] = df['product_id'].astype(str)
    domain_id = np.unique(np.array(df.domain_id))[0]
    #db_obj = DatabaseWrapper()
    #conn = db_obj.connect()
    # 1. NewProduct filtering in data_aggregator,
    # ratio_aggregator, forecast_allocation
    # filter seasonal and new
    Heirarchy = df_hierarchy_new
#     print(Heirarchy[Heirarchy['hierarchy_type']=='product'])
    hm = pd.DataFrame()
    if product_input_level != product_output_level:
        level_difference = product_output_level - product_input_level
        suffix = 'product'

        if level_difference > 0:
            df = product_aggregator(df, Heirarchy, product_input_level, product_output_level)
            hmt = generate_hierarchy(Heirarchy,suffix = suffix)
            hmt = hmt[[f'{suffix}_{product_input_level}',f'{suffix}_{product_output_level}']]
            hmt.rename(columns={f'{suffix}_{int(product_input_level)}': 'heirarchy_value',f'{suffix}_{int(product_output_level)}': 'parent_value'}, inplace=True)

        elif level_difference < 0:
            df = product_dis_aggregator(df, Heirarchy, product_input_level, product_output_level)
            hmt = generate_hierarchy(Heirarchy,suffix = suffix)
            hmt = hmt[[f'{suffix}_{product_input_level}',f'{suffix}_{product_output_level}']]
            hmt.rename(columns={f'{suffix}_{int(product_input_level)}': 'heirarchy_value',f'{suffix}_{int(product_output_level)}': 'parent_value'}, inplace=True)
        hm = hm.append(hmt)
    
    if channel_input_level != channel_output_level:
        level_difference = channel_output_level - channel_input_level
        suffix = 'channel'
        
        if level_difference > 0:
            
            df = channel_aggregator(df, Heirarchy, channel_input_level, channel_output_level)
            hmt = generate_hierarchy(Heirarchy,suffix = suffix)
            hmt = hmt[[f'{suffix}_{channel_input_level}',f'{suffix}_{channel_output_level}']]
            hmt.rename(columns={f'{suffix}_{int(channel_input_level)}': 'heirarchy_value',f'{suffix}_{int(channel_output_level)}': 'parent_value'}, inplace=True)
            
        elif level_difference < 0:
            df = channel_dis_aggregator(df, Heirarchy, channel_input_level, channel_output_level)
            #df = df.reset_index(drop=True)
            hmt = generate_hierarchy(Heirarchy,suffix = suffix)
            hmt = hmt[[f'{suffix}_{product_input_level}',f'{suffix}_{product_output_level}']]
            hmt.rename(columns={f'{suffix}_{int(product_input_level)}': 'heirarchy_value',f'{suffix}_{int(product_output_level)}': 'parent_value'}, inplace=True)
            
        hm = hm.append(hmt)

    if org_unit_input_level != org_unit_output_level:
        level_difference = org_unit_output_level - org_unit_input_level
        suffix = 'org_unit'
        if level_difference > 0:
            df = org_unit_aggregator(df, Heirarchy,org_unit_input_level, org_unit_output_level)
            hmt = generate_hierarchy(Heirarchy,suffix = suffix)
            hmt = hmt[[f'{suffix}_{org_unit_input_level}',f'{suffix}_{org_unit_output_level}']]
            hmt.rename(columns={f'{suffix}_{int(org_unit_input_level)}': 'heirarchy_value',f'{suffix}_{int(org_unit_output_level)}': 'parent_value'}, inplace=True)
        elif level_difference < 0:
            #print("org_disagg")
            df = org_unit_dis_aggregator(df, Heirarchy, org_unit_input_level, org_unit_output_level)
            #df.reset_index(drop=True)
            hmt = generate_hierarchy(Heirarchy,suffix = suffix)
            hmt = hmt[[f'{suffix}_{product_input_level}',f'{suffix}_{product_output_level}']]
            hmt.rename(columns={f'{suffix}_{int(product_input_level)}': 'heirarchy_value',f'{suffix}_{int(product_output_level)}': 'parent_value'}, inplace=True)
        
        hm = hm.append(hmt)
        

    
    df.drop_duplicates(keep='first', inplace=True)
    df = df.reset_index(drop=True)
    df['historical_sale'] = np.round(df['historical_sale'], 0)
    
    hm.drop_duplicates(keep = 'first',inplace = True)
    
    return (df,hm)


# COMMAND ----------

def final_agg(df_final):
    #df_final = df3.copy()


    if (product_input_level == product_output_level) and (channel_input_level == channel_output_level) and (
            org_unit_input_level == org_unit_output_level):
        pass
   
  
    elif (product_input_level > product_output_level) or (channel_input_level > channel_output_level) or (
            org_unit_input_level > org_unit_output_level):
        pass
        # print("Input level cannot be greater than output level")
    elif (product_input_level < product_output_level) or (channel_input_level < channel_output_level) or (
            org_unit_input_level < org_unit_output_level):
        # df_final.to_csv('final_df_before.csv')
        df_final = agg_dis_agg_function(df_final, product_output_level, channel_output_level, org_unit_output_level,
                                        product_input_level, channel_input_level, org_unit_input_level)
        # df_final.to_csv('final_df_after.csv')
        df_final['segmentation'] = 'None'
        df_final['promotion'] = 'No'
        df_final['model_id'] = 0


    df_final['forecast'] = df_final['forecast'].astype(float)
   
    df_final.sort_values(by=['org_unit_id', 'channel_id', 'product_id', 'period'], inplace=True)

    return df_final

# COMMAND ----------

def fetch_run_data(run_id):
    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F3_run_data.parquet"
    run_data = spark.read.format('parquet').load(working_path,inferSchema=True)
    run_data_temp = spark.read.format('parquet').load(working_path,inferSchema=True)

#     run_data = spark.read.jdbc(url=url, table= f"(SELECT * FROM run_data WHERE run_id = '{run_id}' and run_state = 'F3') AS rd", properties = properties)
#     run_data_temp = spark.read.jdbc(url=url, table= f"(SELECT * FROM run_data WHERE run_id = '{run_id}' and run_state = 'F3') AS rdt", properties = properties)
#     run_data = spark.read.jdbc(url=url, table="run_data", properties=properties)
#     run_data1 = run_data.filter((run_data.run_state=='F3') & (run_data.run_id==str(run_id)))
#     run_data_temp = run_data.filter((run_data.run_state=='F3') & (run_data.run_id==run_id))
    run_data1 = run_data.toPandas()
    run_data_temp = run_data_temp.toPandas()
    run_data_temp['forecast'] = ''
    return run_data1, run_data_temp

# COMMAND ----------

def fetch_original_data(fetch_status):
    
#     run_data = spark.read.jdbc(url=url, table="run_data", properties=properties)
#     run_data1 = run_data.filter((run_data.run_state==fetch_status) & (run_data.run_id==str(run_id)))
#     run_data1 = spark.read.jdbc(url=url, table= f"(SELECT * FROM run_data WHERE run_id = '{run_id}' and run_state = '{fetch_status}') AS od", properties = properties)
    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_{fetch_status}_run_data.parquet"
    run_data1 = spark.read.format('parquet').load(working_path,inferSchema=True)
    run_data1 = run_data1.toPandas()
    
    return run_data1

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)
    
    return

# COMMAND ----------

# def update_flag(db_obj: object, db_conn: object, run_id: str, flag: str):
    
#     db_obj.execute_query(db_conn, "update run set run_state = '" + flag + "' where run_id = " + run_id, 'update')
    
#     return

# COMMAND ----------

# db_obj = DatabaseWrapper()
# db_conn = db_obj.connect()

try:
    pipeline_flag = 0
#     status = 'F6'
#     update_flag(db_obj, db_conn, run_id, status)

    df3, df3_temp = fetch_run_data(str(run_id))

    #df2 = df3.copy() #used in poc mode

    if (product_forecast_level == product_input_level) and (channel_forecast_level == channel_input_level) and (
            org_unit_forecast_level == org_unit_input_level):
        print('Disaggregation not required')
        fetch_status = 'F3'
        df2 = fetch_original_data(fetch_status)
        if 'keys' in df2.columns:
            print('original_sales_data Already has keys')
        else:
            df2 = create_keys(df2)
            df2 = df2.reset_index(drop=False)
        df2['segmentation'] = ''
        df2['model_id'] = 0
        df2['promotion'] = ''

    else:
        print('Disaggregation started')

        ratio_table = spark.read.jdbc(url=url, table= f"(SELECT * FROM proportion WHERE run_id = '{run_id}' and domain_id = '{domain_id}') as ratio_table", properties = properties)


        if ratio_table.toPandas().shape[0]>0:

            ratio_table_columns = ['run_id', 'domain_id', 'org_unit_id', 'channel_id', 'product_id', 'ratio']
            ratio_table = ratio_table.select(*ratio_table_columns)
            ratio_table = ratio_table.toPandas()                                                                   


            df,hm = agg_dis_agg_function(df3, product_input_level, channel_input_level, org_unit_input_level,
                                                   product_forecast_level, channel_forecast_level, org_unit_forecast_level)
            print('agg_dis_agg_function call completed')
            df.dropna(subset=['forecast'],inplace=True)
            print('Ratio_table dataframe disagg creation started')
            df = create_dataframe(df, ratio_table)
            df["keys"] = df['org_unit_id'].map(str) + '~' + df['channel_id'].map(str) + '~' + df['product_id'].map(str)
            print('Ratio_table dataframe disagg creation completed')
            df.dropna(subset=['forecast'],inplace=True)
            df.sort_values(by=['keys','period'],inplace =True)
            df2 = df.reset_index(drop=True)
            df2['promotion'] = ''

    df_final = df2.dropna(subset=['product_id', 'channel_id', 'org_unit_id', 'domain_id'])
    df_final['forecast'] = df_final['forecast'].astype(float)
    df_final.sort_values(by=['org_unit_id', 'channel_id', 'product_id', 'period'], inplace=True)
    df_final = final_agg(df_final)
    df_final.drop('keys',axis=1,inplace=True)


    df_final = spark.createDataFrame(df_final)
#     # Ingesting in raw disagg data db
#     db_ingestion(df_final, 'raw_disag_data', 'append')
    print("Disaggregation completed, rounding logic started")
    df_final = df_final.withColumn("psuedo_key",concat_ws("~","org_unit_id","channel_id","product_id"))
    df_final = df_final.groupby("psuedo_key").apply(rounding_logic)
    print('rounding logic completed, future discontinue master started')
    if future_discontinue_master.count() == 0:
        print("future discontinue master is empty.\n future discontinue master is not applied")

    if future_discontinue_master.count() > 0:
        df_final = df_final.toPandas()
        df_final['key'] = df_final['org_unit_id'] + '@' + df_final['channel_id']+ '@' + df_final['product_id'].astype(str)
        fd_data = future_discontinue_master.toPandas()

        dffd = pd.merge(df_final,fd_data[['key', 'period']], on=['key'], how='left')
        f1 = dffd[dffd.period_x>dffd.period_y]

        dffd['forecast'].loc[(dffd.period_x > dffd.period_y)] = 0

        df_final  = dffd[['domain_id', 'org_unit_id', 'channel_id','product_id', 'period_x', 'historical_sale', 'forecast', 'segmentation','model_id','promotion']]
        df_final.rename(columns = {'period_x':'period'},inplace = True)

        df_final = spark.createDataFrame(df_final)
        
    print('future discontinue master completed')
    status = 'F5'
    df_final = df_final.withColumn('run_id',lit(run_id).cast(IntegerType()))
    df_final = df_final.withColumn('run_state',lit(status))
    df_final = df_final.select('domain_id', 'run_id', 'run_state', 'org_unit_id', 'channel_id','product_id', 'period', 'historical_sale', 'forecast',         'segmentation','model_id','promotion')
    
    print('Ingestion started')
    
#     filename = f'f5_run_data_{run_id}.csv'
#     folderPath=f"/dbfs/mnt/test-rundata/" + filename
    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F5_run_data.parquet"
    df_final.repartition(1).write.mode('overwrite').parquet(working_path)
#     df_final.toPandas().to_csv(folderPath, header=True, index=False)

#     db_ingestion(df_final, 'run_data', 'append')
#     update_flag(db_obj, db_conn, run_id, status)
#     db_obj.close(db_conn)
    print('disaggregation sucessful')
    
except:
    traceback.print_exc()
    10/0

#     status = 'F7'
#     print('Disaggregation unsuccessful')
#     #insert_data(db_obj, run_id, df_final, status)
#     update_flag(db_obj, db_conn, run_id, status)
#     db_obj.close(db_conn)
#     pipeline_flag = 1


# COMMAND ----------

# dbutils.fs.mount(
# source = "wasbs://test-rundata@jnjmddevstgacct.blob.core.windows.net",
# mount_point = "/mnt/test-rundata",
# extra_configs = {"fs.azure.account.key.jnjmddevstgacct.blob.core.windows.net":"YeytTvTZCBfz7BcZjT4e4MrtsmFI+1g7wEBYkyrZ9W0beN0SaYRMHXNWUiSmX9CcWhjDu/0nhf2R6/PbSQHmWw=="})

# COMMAND ----------

# if pipeline_flag == 0:
#     print("Success")
#     #update run table with success state
# else:
#     print("failure")
#     #update run table with failure state
#     10/0

# COMMAND ----------

df_final.display()
