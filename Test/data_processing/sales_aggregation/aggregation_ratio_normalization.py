# Databricks notebook source
# dbutils.widgets.removeAll()
# dbutils.widgets.remove('forecast_length')

# COMMAND ----------

# DBTITLE 1,Libraries
import numpy as np
import pandas as pd
import os,sys,re
import pyspark
from scipy import optimize
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
import datetime
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType, DoubleType,DecimalType
import traceback


# COMMAND ----------

# DBTITLE 1,Running the configuration file
# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

#%run /Shared/Test/Databasewrapper_py_test

# COMMAND ----------

# DBTITLE 1,JDBC connection
properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

# def update_flag(db_obj:object,db_conn:object,run_id,flag):
    
#     db_obj.execute_query(db_conn, "update run set run_state = '" + flag + "' where run_id = " + str(run_id), 'update')
    
#     return

# COMMAND ----------

dbutils.widgets.text('run_id', '', 'run_id')
run_id= dbutils.widgets.get("run_id")
run_id = str(run_id)

dbutils.widgets.text('domain_id', '', 'domain_id')
domain_id= dbutils.widgets.get("domain_id")

dbutils.widgets.text('forecast_period', '', 'forecast_period')
input_period= dbutils.widgets.get("forecast_period")
# dbutils.widgets.text('forecast_length', '', 'forecast_length')
# forecast_length= dbutils.widgets.get("forecast_length")
dbutils.widgets.text('run_mode', '', 'run_mode')
run_mode= dbutils.widgets.get("run_mode")

dbutils.widgets.text('proportion_length', '', 'proportion_length')
proportion_length= dbutils.widgets.get("proportion_length")
ratio_weeks = int(proportion_length)

# COMMAND ----------

dbutils.widgets.text('input_period', '', 'input_period')
input_period= dbutils.widgets.get("input_period")

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

dbutils.widgets.text("ref_master_flag", "","")
ref_master_flag = dbutils.widgets.get("ref_master_flag")
ref_master_flag = int(ref_master_flag)
print ("ref_master_flag:",ref_master_flag)

# COMMAND ----------

org_unit_normalization_level = org_unit_forecast_level
channel_normalization_level = channel_forecast_level
product_normalization_level = product_forecast_level

# COMMAND ----------

def run_parameter_data(run_id,url,properties):
    
    parameter = spark.read.jdbc(url=url,table=f"(select parameter_id,run_value from run_parameter where run_id = '{run_id}') as run_value",properties=properties)
    parameter = parameter.toPandas()
   
    parameter = dict(zip(parameter['parameter_id'],parameter['run_value']))
    
    return parameter

# COMMAND ----------

# DBTITLE 1,Run table update
# parameters = run_parameter_data(run_id,url,properties)
# status = parameters[6]
# update_flag(db_obj,db_conn,run_id,status)
# db_obj.close(db_conn)

# COMMAND ----------

df_hierarchy = spark.read.jdbc(url=url, table= f"(SELECT * FROM hierarchy WHERE domain_id = '{domain_id}') AS hm", properties = properties)
df_hierarchy_new=df_hierarchy.toPandas()

# COMMAND ----------

# df_rundata = spark.read.jdbc(url=url, table= f"(SELECT * FROM run_data WHERE domain_id = '{domain_id}' and run_id = '{run_id}' and run_state = 'DP1') AS rd", properties = properties)
working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP1_run_data.parquet"
df_rundata = spark.read.format('parquet').load(working_path,inferSchema=True)
df_rundata=df_rundata.toPandas()

# COMMAND ----------

# df_rundata_tmp = df_rundata.copy()
# df_rundata_tmp['key'] = df_rundata_tmp['org_unit_id'] + df_rundata_tmp['channel_id'] + df_rundata_tmp['product_id']
# df_rundata_tmp['key'].nunique()

# COMMAND ----------

# DBTITLE 1,Generate_hierarchy (creates the hierarchy data in wide format instade of long)
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

# DBTITLE 1,Product Aggregator function (aggregates the product from the product input level to the product output level)
def product_aggregator(df1, Heirarchy, product_input_level, product_output_level):
    df = df1.copy()
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

# DBTITLE 0,Product Dis_aggregator function (disaggregates the data from product output level to product input level)
#product_disaggregator
def product_dis_aggregator(df, Heirarchy, product_input_level, product_output_level):
#     df = df1.copy()
    product_heirarchy = np.unique(Heirarchy[Heirarchy['hierarchy_type'] == 'product']['level_id'])
    prod_heirarchy = Heirarchy[Heirarchy['hierarchy_type'] == 'product']
    if product_input_level > np.amax(product_heirarchy):
        product_input_level = np.amax(product_heirarchy)
    product_input_index = int(np.where(product_heirarchy == product_input_level)[0])
    product_output_index = int(np.where(product_heirarchy == product_output_level)[0])
    aggregate_levels = product_heirarchy[product_output_index:product_input_index]
    aggregate_levels[::-1].sort()
    for i in aggregate_levels:
        input_levels = prod_heirarchy[prod_heirarchy['level_id'] == i]['parent_value']
        input_levels = input_levels.astype(str)
        target_levels = []
        dictionary = {}
        for j in np.unique(input_levels):
            target_levels = np.array(prod_heirarchy[prod_heirarchy['parent_value'] == j]['hierarchy_value'])
            target_levels = target_levels.tolist()
            dictionary[j] = target_levels
        print(dictionary)
        df['product_id'] = df['product_id'].map(dictionary).to_list()
        df = df.explode('product_id').reset_index(drop=True)
        df['product_id'] = df['product_id'].astype(str)
        # # print(np.unique(df['product_id']))
    return df

# COMMAND ----------

# DBTITLE 1,channel Aggregator function (aggregates the channel from the channel input level to the channel output level)
def channel_aggregator(df1, Heirarchy, channel_input_level, channel_output_level):
    df = df1.copy()
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

# DBTITLE 0,channel Dis_aggregator function (disaggregates the data from channel output level to channel input level)
def channel_dis_aggregator(df, Heirarchy, channel_input_level, channel_output_level):
#     df =df1.copy()
    channel_heirarchy = np.unique(Heirarchy[Heirarchy['hierarchy_type'] == 'channel']['level_id'])
    chann_heirarchy = Heirarchy[Heirarchy['hierarchy_type'] == 'channel']
    if channel_input_level > np.amax(channel_heirarchy):
        channel_input_level = np.amax(channel_heirarchy)
    channel_input_index = int(np.where(channel_heirarchy == channel_input_level)[0])
    channel_output_index = int(np.where(channel_heirarchy == channel_output_level)[0])
    aggregate_levels = channel_heirarchy[channel_output_index:channel_input_index]
    aggregate_levels[::-1].sort()
    for i in aggregate_levels:
        input_levels = chann_heirarchy[chann_heirarchy['level_id'] == i]['parent_value']
        input_levels = input_levels.astype(str)
        target_levels = []
        dictionary = {}
        for j in np.unique(input_levels):
            target_levels = np.array(chann_heirarchy[chann_heirarchy['parent_value'] == j]['hierarchy_value'])
            target_levels = target_levels.tolist()
            dictionary[j] = target_levels
        # df['product_id'] = df['product_id'].astype(object)
        df['channel_id'] = df['channel_id'].map(dictionary).to_list()
        df = df.explode('channel_id').reset_index(drop=True)
        df['channel_id'] = df['channel_id'].astype(str)

    return df


# COMMAND ----------

# DBTITLE 1,org_unit Aggregator function (aggregates the org_unit from the org_unit input level to the org_unit output level)
def org_unit_aggregator(df1, Heirarchy, org_unit_input_level, org_unit_output_level):
    df = df1.copy()
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

# DBTITLE 0,org_unit Dis_aggregator function (disaggregates the data from org_unit output level to org_unit input level)
def org_unit_dis_aggregator(df, Heirarchy, org_unit_input_level, org_unit_output_level):
#     df = df1.copy()
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
    for i in aggregate_levels:
        input_levels = org_heirarchy[org_heirarchy['level_id'] == i]['parent_value']
        input_levels = input_levels.astype(str)
        target_levels = []
        dictionary = {}
        for j in np.unique(input_levels):
            target_levels = np.array(org_heirarchy[org_heirarchy['parent_value'] == j]['hierarchy_value'])
            target_levels = target_levels.tolist()
            dictionary[j] = target_levels
        # df['product_id'] = df['product_id'].astype(object)
        print("dictionary:",dictionary)
        # df.to_csv("df_beforemap.csv")
        df['org_unit_id'] = df['org_unit_id'].map(dictionary).to_list()
        # df.to_csv("df_aftermap.csv")
        df = df.explode('org_unit_id').reset_index(drop=True)
        # df.to_csv("df_afterexplode.csv")
        df['org_unit_id'] = df['org_unit_id'].astype(str)
    return df

# COMMAND ----------

# DBTITLE 1,The main aggregation and disaggregation function which calls the individual functions and is being called in the dataprocessing
def agg_dis_agg_function(df1, product_output_level, channel_output_level, org_unit_output_level, product_input_level=0,
                         channel_input_level=0, org_unit_input_level=0):
    df = df1.reset_index(drop=True)
#     df = df1.copy()
    df['product_id'] = df['product_id'].astype(str)
    #db_obj = DatabaseWrapper()
    #conn = db_obj.connect()
    # 1. NewProduct filtering in data_aggregator,
    # ratio_aggregator, forecast_allocation
    # filter seasonal and new
    Heirarchy = df_hierarchy_new
    hm = pd.DataFrame()
    df['historical_sale'] = df['historical_sale'].astype(int)
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
            df = df.reset_index(drop=True)
    
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
            df.reset_index(drop=True)
        hm = hm.append(hmt)
    df.drop_duplicates(keep='first', inplace=True)
    df = df.reset_index(drop=True)
    print("shape of df",df.shape)
    print("display of df",df)
    print("datatypes",df.dtypes)
    df['historical_sale'] = np.round(df['historical_sale'], 0)
    
    hm.drop_duplicates(keep = 'first',inplace = True)

    return (df,hm)

# COMMAND ----------

# DBTITLE 1,forecast_allocation_ratios
def forecast_allocation_ratios(df1, domain_id, run_mode):
    df = df1.copy()
    df['keys'] = df['org_unit_id'].map(str) + '~' + df['channel_id'].map(str) + '~' + df['product_id'].map(str)
    
#     df, keys = create_keys(df)
    df['historical_sale'] = df['historical_sale'].astype(float)
    df_allocations = pd.DataFrame()
    for i in np.unique(df['keys']):
        df_temp = df[df['keys'] == i]
        df_temp = df_temp.sort_values(by=['period'], ascending=True).reset_index(drop=True)
#         if run_mode == 'poc':
#             df_temp = df_temp[0:-forecast_length]
        df_temp['period'] = df_temp['period'].astype(str)
        df_temp['allocated_period'] = df_temp['period'].apply(get_allocation)
        df_temp = df_temp[['keys', 'allocated_period', 'historical_sale']]
        df_temp['historical_sale'] = df_temp['historical_sale'].astype(float)
        df_temp = df_temp.groupby(['keys', 'allocated_period'])['historical_sale'].mean().reset_index()
        df_allocations = pd.concat(([df_allocations, df_temp]), axis=0, ignore_index=True)
    df_allocations.columns = ['keys', 'period', 'fn_ratio']
    df_allocations.insert(loc=3, column='domain_id', value=domain_id)
    df_allocations['org_unit_id'] = df_allocations['keys'].apply(get_org)
    df_allocations['channel_id'] = df_allocations['keys'].apply(get_channel)
    df_allocations['product_id'] = df_allocations['keys'].apply(get_product)
    # print("Flag -1:", df_allocations.columns)
    # print(df_allocations)
    return df_allocations

# COMMAND ----------

# DBTITLE 1,Calculate_ratios() gives the ratios which will be used for disaggregation
def calculate_ratios(df1, df_original1,hierarchy_map, domain_id, run_mode, run_id, ratio_weeks=52):
    
    lst = ['product','channel','org_unit']
    for i in lst:
        df = pd.merge(df_original1,hierarchy_map,left_on = f'{i}_id',right_on = 'heirarchy_value',how = 'left')
        df.drop(columns={'heirarchy_value'}, inplace=True)
        df.rename(columns={'parent_value': f'parent_{i}_id'}, inplace=True)
        df_original1 = df.copy()
    
    df['parent_product_id'] = df['parent_product_id'].combine_first(df['product_id'])
    df['parent_channel_id'] = df['parent_channel_id'].combine_first(df['channel_id'])
    df['parent_org_unit_id'] = df['parent_org_unit_id'].combine_first(df['org_unit_id'])
    
    df.rename(columns={'historical_sale': 'historical_sale_original'}, inplace=True)
    
    df3 = df1.copy()
    df3.rename(columns = {'org_unit_id':'agg_org_unit_id','product_id':'agg_product_id'},inplace = True)
    
    df = pd.merge(df,df3,left_on=['domain_id','parent_org_unit_id','channel_id','parent_product_id','period'],
                  right_on= ['domain_id','agg_org_unit_id','channel_id','agg_product_id','period'], how = 'left')
    
    periods = list(set(df['period']))
    periods.sort(reverse = False)
    last_period_ratio = periods[-ratio_weeks:len(periods)]
#     print("last_period_ratio",last_period_ratio)
    df_temp = df[df['period'].isin(last_period_ratio)].copy()
#     print(df_temp.head(2))
    df_temp = df_temp.groupby(by=['org_unit_id', 'channel_id', 'product_id']).agg({'historical_sale': 'sum', 'historical_sale_original':'sum'}).reset_index()
    df_temp['historical_sale'] = df_temp['historical_sale'].map(lambda x: int(x) + 1 if int(x) == 0 else int(x))
    df_temp['ratio'] = df_temp['historical_sale_original'] / df_temp['historical_sale']
    df_temp['domain_id'] = domain_id
    ratio_table = df_temp.copy()

    ratio_table['run_id'] = run_id
    ratio_table = ratio_table[['run_id', 'domain_id', 'org_unit_id', 'channel_id', 'product_id', 'ratio']]
    ratio_table = ratio_table.astype(
        {'run_id': 'int', 'domain_id': 'str', 'org_unit_id': 'str', 'channel_id': 'str', 'product_id': 'str',
         'ratio': 'float64'})
    return ratio_table

# COMMAND ----------

# DBTITLE 1,Calling the agg_dis_agg_function() for validation
# agg_dis_agg,hm=agg_dis_agg_function(df_rundata,product_forecast_level,channel_forecast_level,org_unit_forecast_level,
#                                   product_input_level,channel_input_level, org_unit_input_level)

# COMMAND ----------

# DBTITLE 1,Displaying the output
# agg_dis_agg.display()

# COMMAND ----------

def get_allocation(input_period):
    allocated_period = re.split('W|D|M|Q', input_period)[1]
    return allocated_period

def get_org(input_data):
    org = re.split('~', input_data)[0]
    return org


def get_channel(input_data):
    channel = re.split('~', input_data)[1]
    return channel


def get_product(input_data):
    product = re.split('~', input_data)[2]
    return product


# COMMAND ----------

# DBTITLE 1,Forecast normalization
# historical_sales_data4 = df_rundata.copy()
# historical_sales_data4,hm1 = agg_dis_agg_function(historical_sales_data4, product_normalization_level,
#                                                           channel_normalization_level, org_unit_normalization_level,
#                                                           product_input_level,
#                                                           channel_input_level, org_unit_input_level)
# # print("agg dis",historical_sales_data4.head(2))

# print('Forecast allocation started')
# historical_sales_data4.dtypes
# forecast_normalization_ratio_table = forecast_allocation_ratios(historical_sales_data4,domain_id, run_mode)
# print('Forecast allocation completed')
# forecast_normalization_ratio_table['product_id'] = forecast_normalization_ratio_table['product_id'].astype(str)
# forecast_normalization_ratio_table1 = forecast_normalization_ratio_table.reset_index(drop=True).copy()

# forecast_normalization_ratio_table2 = forecast_normalization_ratio_table1.copy()
# forecast_normalization_ratio_table2['keys'] = forecast_normalization_ratio_table2['org_unit_id'].map(str) + '~' + forecast_normalization_ratio_table2['channel_id'].map(str) + '~' + forecast_normalization_ratio_table2['product_id'].map(str)
# forecast_normalization_ratio_table2.drop_duplicates(inplace=True)
# forecast_normalization_ratio_table2.drop(columns = {"domain_id","org_unit_id","channel_id","product_id"},inplace = True)
# forecast_normalization_ratio_table2.rename(columns={'keys':'feature_key_ref','fn_ratio':'feature_value'}, inplace=True)
# forecast_normalization_ratio_table2.insert(loc=1, column='domain_id', value=domain_id)
# forecast_normalization_ratio_table2.insert(loc=2, column='run_id', value=run_id)
# forecast_normalization_ratio_table2 = forecast_normalization_ratio_table2[['run_id', 'feature_key_ref', 'feature_value', 'period', 'domain_id']]

# COMMAND ----------

# forecast_normalization_ratio_table2.display()

# COMMAND ----------

# DBTITLE 1,Calling the create_ratios()
# ratio_table = calculate_ratios(agg_dis_agg, df_rundata,hm, domain_id, run_mode, run_id, ratio_weeks)

# COMMAND ----------

# DBTITLE 1,Post processing on Ratio table
# ratio_table1 = spark.createDataFrame(ratio_table.astype(str))
# ratio_table1 = ratio_table1.withColumn("ratio", ratio_table1["ratio"].cast('decimal(15,5)'))

# COMMAND ----------

# ratio_table.display()

# COMMAND ----------

forecast_normalization_ratio_table2_Schema = StructType([StructField('run_id',StringType(),True),
                        StructField('feature_key_ref',StringType(),True),
                        StructField('feature_value',FloatType(),True),
                        StructField('period',StringType(),True),
                        StructField('domain_id',StringType(),True)
                       ])


# COMMAND ----------

# DBTITLE 1,Post processing on forecast_normalization_ratio_table2
# forecast_normalization_ratio_table2 = sqlContext.createDataFrame(forecast_normalization_ratio_table2, forecast_normalization_ratio_table2_Schema)

# COMMAND ----------

# DBTITLE 1,Writing to the blob storage which is required for segmentation and forecastability cutoff
# filename = f'DP3_run_data_{run_id}.csv'
# folderPath='/dbfs/mnt/azurestorage/SS4/' + filename
# agg_dis_agg.to_csv(folderPath, header=True, index=False)

# COMMAND ----------

# DBTITLE 1,Ratio Table into DB as Proportion Table
# db_ingestion(ratio_table1, 'proportion', 'append')

# COMMAND ----------

# DBTITLE 1,forecast_normalization_ratio_table2 into DB as normalization table
# db_ingestion(forecast_normalization_ratio_table2, 'normalization', 'append')

# COMMAND ----------

# db_obj = DatabaseWrapper()
# db_conn = db_obj.connect()

try:
    
    pipeline_flag = 0
#     parameters = run_parameter_data(run_id,url,properties)
#     status = parameters[6]
#     update_flag(db_obj,db_conn,run_id,status)

    if (product_forecast_level == product_input_level) and (channel_forecast_level == channel_input_level) and (org_unit_forecast_level == org_unit_input_level):
        if ref_master_flag == 1 :
            reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{domain_id}') as   reference_master",properties = properties)
            reference_master = reference_master.toPandas()
            df_rundata['key']  = df_rundata['org_unit_id'] + '@' + df_rundata['channel_id'] + '@' + df_rundata['product_id']
            reference_master['key']  = reference_master['org_unit_id'] + '@' + reference_master['channel_id'] + '@' + reference_master['product_id']
            df_rundata = df_rundata[df_rundata['key'].isin(reference_master['key'])]
            df_rundata.drop(columns={'key'}, inplace=True)
                        
       
    ## Ingesting in the blob storage
#         filename = f'DP3_run_data_{run_id}.csv'
#         folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
        df_rundata1 = df_rundata.drop(columns = ['run_id','run_state','segmentation','promotion','model_id'])
#         df_rundata1.to_csv(folderPath, header=True, index=False)

        df_rundata1 = spark.createDataFrame(df_rundata1)
        working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP3_Int_data.parquet"
        df_rundata1.repartition(1).write.mode('overwrite').parquet(working_path)
        df_rundata1 = df_rundata1.toPandas()
    
    else:
        agg_dis_agg,hm=agg_dis_agg_function(df_rundata,product_forecast_level,channel_forecast_level,org_unit_forecast_level,
                                     product_input_level,channel_input_level, org_unit_input_level)

        if ref_master_flag == 1 :
            reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{domain_id}') as   reference_master",properties = properties)
            reference_master = reference_master.toPandas()
            agg_dis_agg['key']  = agg_dis_agg['org_unit_id'] + '@' + agg_dis_agg['channel_id'] + '@' + agg_dis_agg['product_id']
            reference_master['key']  = reference_master['org_unit_id'] + '@' + reference_master['channel_id'] + '@' + reference_master['product_id']

            agg_dis_agg = agg_dis_agg[agg_dis_agg['key'].isin(reference_master['key'])]

            agg_dis_agg.drop(columns={'key'}, inplace=True)

        agg_dis_agg = spark.createDataFrame(agg_dis_agg)


        #Ingesting in the blob storage
#         filename = f'DP3_run_data_{run_id}.csv'
#         folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
        working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP3_Int_data.parquet"
#         agg_dis_agg.to_csv(folderPath, header=True, index=False)
        agg_dis_agg.repartition(1).write.mode('overwrite').parquet(working_path)
    
        agg_dis_agg = agg_dis_agg.toPandas()
        
                # Ratio creation
        ratio_table = calculate_ratios(agg_dis_agg, df_rundata,hm, domain_id, run_mode, run_id, ratio_weeks)
    #     filename = f'ratio_table1_{run_id}.csv'
    #     folderPath=f"/dbfs/mnt/disk/staging_data/test/{domain_id}/" + filename
    #     ratio_table.to_csv(folderPath, header=True, index=False)
        ratio_table1 = spark.createDataFrame(ratio_table.astype(str))
        ratio_table1 = ratio_table1.withColumn("ratio", ratio_table1["ratio"].cast('decimal(15,5)'))

        #  DB ingestion Proportion table
        db_ingestion(ratio_table1, 'proportion', 'append')

        print('aggreagtion and ratio creation and normalization table creation successful')

    historical_sales_data4 = df_rundata.copy()
    historical_sales_data4,hm1 = agg_dis_agg_function(historical_sales_data4, product_normalization_level,
                                                              channel_normalization_level, org_unit_normalization_level,
                                                              product_input_level,
                                                              channel_input_level, org_unit_input_level)

    print('Forecast allocation started')
    historical_sales_data4.dtypes
    forecast_normalization_ratio_table = forecast_allocation_ratios(historical_sales_data4,domain_id, run_mode)
    print('Forecast allocation completed')
    forecast_normalization_ratio_table['product_id'] = forecast_normalization_ratio_table['product_id'].astype(str)
    forecast_normalization_ratio_table1 = forecast_normalization_ratio_table.reset_index(drop=True).copy()

    forecast_normalization_ratio_table2 = forecast_normalization_ratio_table1.copy()
    forecast_normalization_ratio_table2['keys'] = forecast_normalization_ratio_table2['org_unit_id'].map(str) + '~' + forecast_normalization_ratio_table2['channel_id'].map(str) + '~' + forecast_normalization_ratio_table2['product_id'].map(str)
    forecast_normalization_ratio_table2.drop_duplicates(inplace=True)
    forecast_normalization_ratio_table2.drop(columns = {"domain_id","org_unit_id","channel_id","product_id"},inplace = True)
    forecast_normalization_ratio_table2.rename(columns={'keys':'feature_key_ref','fn_ratio':'feature_value'}, inplace=True)
    forecast_normalization_ratio_table2.insert(loc=1, column='domain_id', value=domain_id)
    forecast_normalization_ratio_table2.insert(loc=2, column='run_id', value=run_id)
    forecast_normalization_ratio_table2 = forecast_normalization_ratio_table2[['run_id', 'feature_key_ref', 'feature_value', 'period', 'domain_id']]
#     filename = f'forecast_normalization_ratio_{run_id}.csv'
#     folderPath=f"/dbfs/mnt/disk/staging_data/test/{domain_id}/" + filename
#     forecast_normalization_ratio_table2.to_csv(folderPath, header=True, index=False)
    print("The count of forecast_normalization_ratio_table2",forecast_normalization_ratio_table2.shape)    
    forecast_normalization_ratio_table2 = spark.createDataFrame(data = forecast_normalization_ratio_table2, schema = forecast_normalization_ratio_table2_Schema)

    #     DB ingestion normalization table
    db_ingestion(forecast_normalization_ratio_table2, 'normalization', 'append')
    
    print("ingestion completed for the normalization table")

#     db_obj.close(db_conn)
except:
    traceback.print_exc()
    10/0
#     parameters = run_parameter_data(run_id,url,properties)
#     status = parameters[8]
#     update_flag(db_obj,db_conn,run_id,status)
#     traceback.print_exc()
#     db_obj.close(db_conn)
#     pipeline_flag = 1
#     print('aggreagtion or ratio creation or normalization table creation failed')

# COMMAND ----------

# if pipeline_flag == 0:
#     print("Success")
# else:
#     print("failure")
#     10/0
