# Databricks notebook source
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import concat_ws, lower, col,countDistinct,monotonically_increasing_id
import pandas as pd
import numpy as np
import pyspark.sql.functions as f
from pyspark.sql.functions import split
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType, DoubleType,DecimalType
import traceback

# COMMAND ----------

ingestion_flag = 1

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
domain_id = dbutils.widgets.get("domain_id")
print ("domain_id:",domain_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Database connection

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

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
raw_run_data = spark.read.jdbc(url=url, table= f"(SELECT * FROM raw_history_master WHERE run_state = 'I5' and domain_id = '{domain_id}') as raw_history_master", properties = properties)
# raw_run_data = raw_run_data.withColumn('forecast',f.col('forecast').cast(FloatType()))

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
d_file = spark.read.jdbc(url=url, table= f"(SELECT * FROM hierarchy WHERE domain_id = '{domain_id}') AS hm", properties = properties)
d_file = d_file.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ###NPF Filtering

# COMMAND ----------

# old code npf line----
#npf_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM npf_master WHERE domain_id = '{domain_id}') as npf_master", properties = properties)
# New code NPF-----
npf_master = spark.read.csv(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_npf_master_{domain_id}.csv",inferSchema=True,header=True)


# COMMAND ----------

def filter_active_products(raw_run_data, npf_master):
    
    raw_run_data = raw_run_data.withColumn('key',concat_ws('@',col('org_unit_id'),col('channel_id'),col('product_id')))
    npf_master = npf_master.withColumn('key',concat_ws('@',col('org_unit_id'),col('channel_id'),col('product_id')))
    npf_master = npf_master.withColumnRenamed('org_unit_id','org_unit_id_npf').withColumnRenamed('channel_id','channel_id_npf').withColumnRenamed('product_id','product_id_npf').withColumnRenamed('domain_id','domain_id_npf')
    npf_master = npf_master.filter(npf_master.status == 'Active')
    npf_master.createOrReplaceTempView(f'npf_master_{run_id}')
    raw_run_data.createOrReplaceTempView(f'raw_run_data_{run_id}')
    filter_active_products = spark.sql(f'select * from raw_run_data_{run_id} rd inner join npf_master_{run_id} npf on rd.key=npf.key')
    filter_active_products = filter_active_products.drop('key','org_unit_id_npf','channel_id_npf','product_id_npf','status','domain_id_npf')
    
    return filter_active_products

# COMMAND ----------

# MAGIC %md
# MAGIC #####Ingesting into run_data

# COMMAND ----------

if ref_master_flag == 1 :
    
    reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{domain_id}') as reference_master",properties = properties)

# COMMAND ----------

try:
    pipeline_flag = 0
    if ref_master_flag == 1 :
        print("Inside IF condition")       
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
        pseudo_table = spark.createDataFrame(pseudo_table)
        pseudo_table.createOrReplaceTempView(f'pseudo_table_{run_id}')
        raw_run_data.createOrReplaceTempView(f'raw_run_data_{run_id}')
        raw_run_data = spark.sql(f'select * from raw_run_data_{run_id} r inner join pseudo_table_{run_id} p on r.product_id = p.input_level ')
        raw_run_data = raw_run_data.drop('input_level')
        raw_run_data = raw_run_data.withColumnRenamed('forecast_level','product_forecast_level')
        raw_run_data.createOrReplaceTempView(f'raw_run_data_{run_id}')
        raw_run_data = spark.sql(f'select * from raw_run_data_{run_id} r inner join pseudo_table_{run_id} p on r.channel_id = p.input_level ')
        raw_run_data = raw_run_data.drop('input_level')
        raw_run_data = raw_run_data.withColumnRenamed('forecast_level','channel_forecast_level')
        raw_run_data.createOrReplaceTempView(f'raw_run_data_{run_id}')
        raw_run_data = spark.sql(f'select * from raw_run_data_{run_id} r inner join pseudo_table_{run_id} p on r.org_unit_id = p.input_level ')
        raw_run_data = raw_run_data.drop('input_level')
        raw_run_data = raw_run_data.withColumnRenamed('forecast_level','org_unit_forecast_level')
        raw_run_data = raw_run_data.withColumn('key',concat_ws('@',col('org_unit_forecast_level'),col('channel_forecast_level'),col('product_forecast_level')))
        reference_master = reference_master.withColumn('key',concat_ws('@',col('org_unit_id'),col('channel_id'),col('product_id')))
        raw_run_data = raw_run_data.filter(raw_run_data.key.isin(reference_master.select('key').distinct().rdd.map(lambda r:r[0]).collect()))
        raw_run_data = raw_run_data.drop('org_unit_forecast_level','channel_forecast_level','product_forecast_level')


    #print(raw_run_data.count())
    npf_filter = filter_active_products(raw_run_data,npf_master)
    raw_history_master = npf_filter.select('*')
    npf_filter = npf_filter.withColumn('run_id',f.lit(run_id)).withColumnRenamed('expost_sales','historical_sale').drop('cdh')
    
    npf_filter  = npf_filter.select('domain_id', 'run_id','run_state','org_unit_id','channel_id','product_id','period','historical_sale','forecast','segmentation','model_id','promotion')
#     print(npf_filter.count())
    if ingestion_flag == 1:
        working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_I5_run_data.parquet"
        npf_filter.repartition(1).write.mode('overwrite').parquet(working_path)
except:
    traceback.print_exc()
    10/0


# COMMAND ----------

# MAGIC %md
# MAGIC ###Aggregation of history master

# COMMAND ----------

# MAGIC %run /Shared/Test/data_processing_rf/masters_aggregation/history_master_aggregation_test

# COMMAND ----------

try:
    pipeline_flag = 0
    raw_history_master = raw_history_master.drop('forecast','segmentation','model_id','promotion')
    agg_history_master = history_master_agg_function(raw_history_master, d_file, product_forecast_level,channel_forecast_level,org_unit_forecast_level,
                                                 product_input_level,channel_input_level,org_unit_input_level,run_id)
    agg_history_master = agg_history_master.withColumn('run_id',lit(run_id)).withColumn('key',concat_ws('@',col('org_unit_id'),col('channel_id'),col('product_id')))
    agg_history_master = agg_history_master.drop('org_unit_id','channel_id','product_id')
#     agg_history_master.repartition(1).write.mode('overwrite').parquet(f"dbfs:/FileStore/reference_data_test/{env}/agg_history_master_{domain_id}_{run_id}.parquet")
    
    if ingestion_flag == 1:
        db_ingestion(agg_history_master, 'agg_history_master', 'append')
    else:
        print("ingestion not done")
    
except:
    traceback.print_exc()
    10/0   
