# Databricks notebook source
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import concat_ws, lower, col,countDistinct
import pandas as pd
import numpy as np
import pyspark.sql.functions as f
from pyspark.sql.functions import split
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType, DoubleType,DecimalType
import warnings
import traceback

# COMMAND ----------

# MAGIC %md
# MAGIC ####To get run_id

# COMMAND ----------

dbutils.widgets.text("run_id", "","")
run_id = dbutils.widgets.get("run_id")
print ("run_id:",run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Getting the domain_id

# COMMAND ----------

dbutils.widgets.text("domain_id", "","")
Domain = dbutils.widgets.get("domain_id")
print ("domain_id:",Domain)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Database connection

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

#%run /Shared/Test/Databasewrapper_py_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

# def update_flag(db_obj:object,db_conn:object,run_id,flag):
    
#     db_obj.execute_query(db_conn, "update run set run_state = '" + flag + "' where run_id = " + run_id, 'update')
    
#     return

# COMMAND ----------

# MAGIC %md
# MAGIC ####Ingestion Function

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Get Aggregation Parameters

# COMMAND ----------

# MAGIC %md
# MAGIC #####input level parameters

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

# MAGIC %md
# MAGIC #####forecast level parameters

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

# MAGIC %md
# MAGIC ###Reverse hierarchy logic

# COMMAND ----------

def generate_hierarchy(data, suffix='product'):
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

# raw_history_master = spark.read.jdbc(url=url, table= "raw_history_master", properties = properties)
# raw_history_master.createOrReplaceTempView('raw_history_master')
raw_history_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM raw_history_master WHERE run_state = 'I5' and domain_id = '{Domain}') as raw_history_master", properties = properties)
raw_history_master = raw_history_master.withColumn('forecast',f.col('forecast').cast(FloatType()))

# COMMAND ----------

# raw_run_data = spark.sql("SELECT * from raw_history_master where run_state = 'I5' AND domain_id = '{}'".format(Domain))
raw_run_data = raw_history_master.toPandas()
raw_run_data.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ###check if reference_master is uploaded or not and filter the raw history master

# COMMAND ----------

dbutils.widgets.text("ref_master_flag", "","")
ref_master_flag = dbutils.widgets.get("ref_master_flag")
ref_master_flag = int(ref_master_flag)
print ("ref_master_flag:",ref_master_flag)

# COMMAND ----------

 #Read the heirarchy master for reverse hierarchy
d_file = spark.read.jdbc(url=url, table= "hierarchy", properties = properties)
d_file.createOrReplaceTempView('d_file')
d_file = spark.sql("SELECT * from d_file where domain_id = '{}'".format(Domain))
d_file = d_file.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ###NPF Filtering

# COMMAND ----------

#npf_master = spark.read.jdbc(url=url, table= "npf_master", properties = properties)

#new code npf
npf_master = spark.read.csv(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_npf_master_{domain_id}.csv",inferSchema=True,header=True)
npf_master = npf_master.filter(npf_master.domain_id == Domain)

npf_master = npf_master.toPandas()
#npf_master.display()

# COMMAND ----------

def filter_active_products(domain_id, dataframe, npf_master_data):
    npf_master_columns = ['org_unit_id', 'channel_id', 'product_id', 'status']
    npf_master_data = npf_master_data[npf_master_columns]
    npf_master_data.columns = npf_master_columns
    npf_master_data['keys'] = (npf_master_data['org_unit_id']).astype(str) + '~' + (
        npf_master_data['channel_id']).astype(str) + '~' + (npf_master_data['product_id']).astype(str)

    dataframe['keys'] = (dataframe['org_unit_id']).astype(str) + '~' + (dataframe['channel_id']).astype(
        str) + '~' + (dataframe['product_id']).astype(str)

    final_data = dataframe.merge(npf_master_data, on='keys', how='left', suffixes=('', '_drop'))
    to_drop = [x for x in final_data if x.endswith('_drop')]
    final_data.drop(to_drop, axis=1, inplace=True)
    final_data = final_data[(final_data['status'] == 'Active')]
    final_data.drop(columns=['keys', 'status'], inplace=True)
    return final_data

# COMMAND ----------

outSchema = StructType([StructField('domain_id',StringType(),True),
                        StructField('run_state',StringType(),True),
                        StructField('org_unit_id',StringType(),True),
                        StructField('channel_id',StringType(),True),
                        StructField('product_id',StringType(),True),
                        StructField('period',StringType(),True),                      
                        StructField('forecast',FloatType(),True),
                        StructField('segmentation',StringType(),True),
                        StructField('model_id',StringType(),True),
                        StructField('promotion',StringType(),True),
                        StructField('historical_sale',IntegerType(),True),
                        StructField('run_id',StringType(),True),
                       ])

# COMMAND ----------

# MAGIC %md
# MAGIC #####Ingesting into run_data

# COMMAND ----------

# def db_ingestion(df, table_name, mode):
#     counts = df.count()
#     paritions = counts // 1000000 + 1
#     #default partition = 24                                                                         #This function is already implemeneted in previous cell in this notebook
#     df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

if ref_master_flag == 1 :
    
    reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{Domain}') as reference_master",properties = properties)

# COMMAND ----------

def save_output1(df,excel_path,status):
    file_name = excel_path
    lim = 1000000
    rows = df.count()
    
    def get_col_name(col): # col is 1 based
        excelCol = str()
        div = col
        while div:
            (div, mod) = divmod(div-1, 26) # will return (x, 0 .. 25)
            excelCol = chr(mod + 65) + excelCol

        return excelCol
    
    last_cell = get_col_name(len(df.columns)) + str(rows + 1)


    if rows > lim:
        files = [x for x in range(0,rows,lim)]
        files.append(rows)
        for i in range(len(files)):
            if i < len(files)-1:
                df.withColumn('index',monotonically_increasing_id()).where(col("index").between(files[i],files[i+1]-1)).drop('index')\
                .write.format("com.crealytics.spark.excel")\
                .option("dataAddress",f"{status}_split_{str(i+1)}!A1:{last_cell}")\
                .option("header", "true")\
                .mode("append")\
                .save(file_name)
    else:
        df.write.format("com.crealytics.spark.excel")\
        .option("dataAddress",f"{status}!A1:{last_cell}")\
        .option("header", "true")\
        .mode("overwrite")\
        .save(file_name)

    return file_name

# COMMAND ----------

try:
    pipeline_flag = 0
    if ref_master_flag == 1 :
        print("Inside IF condition")
        #Reading reference master 
         #reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{Domain}') as reference_master",properties = properties)
#         reference_master.createOrReplaceTempView('reference_master')
#         reference_master = spark.sql("SELECT * from reference_master where run_id = '{0}' AND domain_id = '{1}'".format(run_id,Domain))
        print("The count of reference_master",reference_master.count())
        #splitting the key in reference master
        print('splitting keys')
#         try: 
#             print("The columns are:",reference_master.columns)
#             reference_master_split = reference_master.select(f.split(reference_master.key,"@")).rdd.flatMap(
#                   lambda x: x).toDF(schema=["org_unit_id","channel_id","product_id"])            
#         except Exception as e:
            #print("The error is",e)
#         split_col = split(reference_master['key'], '@')
#         reference_master_split = reference_master.withColumn('org_unit_id', split_col.getItem(0)) \
#             .withColumn('channel_id', split_col.getItem(1)) \
#             .withColumn('product_id', split_col.getItem(2))
#         print('splitting keys to ended')
        #reference_master_split = reference_master.toPandas()
        reference_master_split = reference_master.toPandas()
        
        print('copying dataframe by filtering')
        p_file = d_file[d_file['hierarchy_type'] == 'product'].copy()
        c_file = d_file[d_file['hierarchy_type'] == 'channel'].copy()
        o_file = d_file[d_file['hierarchy_type'] == 'org_unit'].copy()
        print('copying dataframe ended, generate hierarchy started')
        p_file = generate_hierarchy(p_file, 'product')
        c_file = generate_hierarchy(c_file, 'channel')
        o_file = generate_hierarchy(o_file, 'org_unit')
        print('generate hierarchy ended, filtering started')
        
        
        p_file = p_file[[f'product_{product_input_level}', f'product_{product_forecast_level}']].copy()
        p_file.columns = ['input_level', 'forecast_level']
        
        c_file = c_file[[f'channel_{channel_input_level}', f'channel_{channel_forecast_level}']].copy()
        c_file.columns = ['input_level', 'forecast_level']
        
        o_file = o_file[[f'org_unit_{org_unit_input_level}', f'org_unit_{org_unit_forecast_level}']].copy()
        o_file.columns = ['input_level', 'forecast_level']
        
        pseudo_table = p_file.copy()
        pseudo_table = pd.concat([pseudo_table, c_file.copy()])
        pseudo_table = pd.concat([pseudo_table, o_file.copy()])
        pseudo_table.reset_index(inplace=True, drop=True)
        
        raw_run_data = pd.merge(raw_run_data, pseudo_table, how='inner', left_on='product_id', right_on='input_level')
        raw_run_data.drop(columns=['input_level'], inplace=True)
        raw_run_data.rename(columns={'forecast_level': 'product_forecast_level'}, inplace=True)
        
        raw_run_data = pd.merge(raw_run_data, pseudo_table, how='inner', left_on='channel_id', right_on='input_level')
        raw_run_data.drop(columns=['input_level'], inplace=True)
        raw_run_data.rename(columns={'forecast_level': 'channel_forecast_level'}, inplace=True)
        
        raw_run_data = pd.merge(raw_run_data, pseudo_table, how='inner', left_on='org_unit_id', right_on='input_level')
        raw_run_data.drop(columns=['input_level'], inplace=True)
        raw_run_data.rename(columns={'forecast_level': 'org_unit_forecast_level'}, inplace=True)
        raw_run_data['key'] = raw_run_data['org_unit_forecast_level'] + '@' + raw_run_data['channel_forecast_level'] + '@' + raw_run_data['product_forecast_level']
        reference_master_split['key'] = reference_master_split['org_unit_id'] + '@' + reference_master_split['channel_id'] + '@' + reference_master_split['product_id']
        raw_run_data = raw_run_data[raw_run_data['key'].isin(reference_master_split['key'])]
        raw_run_data.drop(columns=['org_unit_forecast_level', 'channel_forecast_level', 'product_forecast_level', 'key'], inplace=True)

        
#         #Reference master filtering
#         hierarchy_master = p_file[p_file[f'product_{product_forecast_level}'].isin(reference_master_split['product_id'])]
#         raw_run_data = raw_run_data[raw_run_data['product_id'].isin(hierarchy_master[f'product_{product_input_level}'])]
#         hierarchy_master = c_file[c_file[f'channel_{channel_forecast_level}'].isin(reference_master_split['channel_id'])]
#         raw_run_data = raw_run_data[raw_run_data['channel_id'].isin(hierarchy_master[f'channel_{channel_input_level}'])]
#         hierarchy_master = o_file[o_file[f'org_unit_{org_unit_forecast_level}'].isin(reference_master_split['org_unit_id'])]
#         raw_run_data = raw_run_data[raw_run_data['org_unit_id'].isin(hierarchy_master[f'org_unit_{org_unit_input_level}'])]
        print('filtering ended,npf filter started')
    
    
    npf_filter = filter_active_products(Domain,raw_run_data,npf_master)
    raw_history_master = npf_filter.copy() 
    npf_filter['run_id'] = run_id
    npf_filter.rename(columns = {'expost_sales' : 'historical_sale'}, inplace = True)
    npf_filter.drop(columns = {'cdh'}, inplace = True)
    npf_filter = spark.createDataFrame(npf_filter, schema=outSchema)
    print("npf filter successful")
    
    npf_filter  = npf_filter.select('domain_id', 'run_id','run_state','org_unit_id','channel_id','product_id','period','historical_sale','forecast','segmentation','model_id','promotion')

    
    # Ingesting to the DB
#     db_ingestion(npf_filter, 'run_data', 'append')
    working_path=f"/mnt/disk/staging_data/{env}/{Domain}/Run_id_{run_id}/{run_id}_I5_run_data.parquet"
    npf_filter.repartition(1).write.mode('overwrite').parquet(working_path)
#     excel_path = f'/mnt/jnj_output_file/{env}/{Domain}/Run_ID_{run_id}/I5_{run_id}.xlsx'
#     save_output1(npf_filter,excel_path,'I5')
    print("ingestion successful")
    
except:
    traceback.print_exc()
    10/0
#     pipeline_flag = 1
#     try:
#         db_obj = DatabaseWrapper()
#         db_conn = db_obj.connect()
#         run_status_flag = 0
#         status = "I4"
#         update_flag(db_obj, db_conn, run_id, status)
#         db_obj.close(db_conn)
#     except:
#         run_status_flag = 1

# COMMAND ----------

# if pipeline_flag == 0:
#     print("Success")
# elif (pipeline_flag ==1 and run_status_flag == 0):
#     print("suceeded in updating flag in exception")
#     10/0
# elif (pipeline_flag ==1 and run_status_flag == 1):
#     print("failed in updating flag in exception")
#     10/0

# COMMAND ----------

# MAGIC %md
# MAGIC ###Aggregation of history master

# COMMAND ----------

# MAGIC %run /Shared/Test/data_processing/masters_aggregation/history_master_aggregation

# COMMAND ----------

outSchema = StructType([StructField('domain_id',StringType(),True),                                            
                        StructField('period',StringType(),True),                      
                        StructField('cdh',IntegerType(),True),
                        StructField('expost_sales',IntegerType(),True),
                        StructField('run_id',StringType(),True),
                        StructField('key',StringType(),True),
                       ])

# COMMAND ----------

try:
    pipeline_flag = 0
    raw_history_master.drop(columns = {'forecast','segmentation','model_id','promotion'}, inplace = True)
    agg_history_master = history_master_agg_function(raw_history_master, d_file, product_forecast_level,channel_forecast_level,org_unit_forecast_level,
                                                 product_input_level,channel_input_level,org_unit_input_level)
    agg_history_master['run_id'] = run_id
    agg_history_master['key'] = agg_history_master.org_unit_id + "@" + agg_history_master.channel_id + "@" + agg_history_master.product_id
    agg_history_master.drop(columns = ['org_unit_id','channel_id','product_id'], inplace = True)
    agg_history_master = spark.createDataFrame(agg_history_master, schema=outSchema)
    db_ingestion(agg_history_master, 'agg_history_master', 'append')
except:
    traceback.print_exc()
    10/0
#     pipeline_flag = 1
#     try:
#         db_obj = DatabaseWrapper()
#         db_conn = db_obj.connect()
#         run_status_flag = 0
#         status = "I4"
#         update_flag(db_obj, db_conn, run_id, status)
#         db_obj.close(db_conn)
#     except:
#         run_status_flag = 1    

# COMMAND ----------

# if pipeline_flag == 0:
#     print("Success")
# elif (pipeline_flag ==1 and run_status_flag == 0):
#     print("suceeded in updating flag in exception")
#     10/0
# elif (pipeline_flag ==1 and run_status_flag == 1):
#     print("failed in updating flag in exception")
#     10/0
