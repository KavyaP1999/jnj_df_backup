# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, lower, col, split
from pyspark.sql import functions as f
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import traceback

# COMMAND ----------

# MAGIC %md
# MAGIC ####Get run_id and domain_id 

# COMMAND ----------

dbutils.widgets.text("run_id", "","")
run_id = dbutils.widgets.get("run_id")
print ("run_id:",run_id)

# COMMAND ----------

dbutils.widgets.text("domain_id", "","")
domain = dbutils.widgets.get("domain_id")
print ("domain_id:",domain)

# COMMAND ----------

spark = SparkSession.builder.appName("demo").getOrCreate()

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

#%run /Shared/Test/Databasewrapper_py_test

# COMMAND ----------

# def update_flag(db_obj:object,db_conn:object,run_id,flag):
    
#     db_obj.execute_query(db_conn, "update run set run_state = '" + flag + "' where run_id = " + run_id, 'update')
    
#     return

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

transitioned_future_forecast_master = spark.read.csv(f"/mnt/disk/staging_data/test/{domain}/transitioned_raw_future_forecast_master_{domain}.csv",inferSchema=True,header=True)

# COMMAND ----------

#old line code
#npf_master = spark.read.jdbc(url=url, table="npf_master", properties=properties)
# new line code below
npf_master = spark.read.csv(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_npf_master_{domain_id}.csv",inferSchema=True,header=True)
npf_master = npf_master.filter(npf_master.domain_id==domain)

# COMMAND ----------

npf_master = npf_master.withColumn('keys', concat_ws('@', npf_master.domain_id, npf_master.org_unit_id, npf_master.channel_id, npf_master.product_id))
npf_master = npf_master.select([lower('status').alias('status'), 'keys'])
final_data = transitioned_future_forecast_master.join(npf_master, transitioned_future_forecast_master.key==npf_master.keys, "inner")
final_data = final_data.filter((final_data.status=='active'))
final_data = final_data.drop('keys', 'status')

# COMMAND ----------

final_data = final_data.withColumn("org_unit_id", split(col("key"), "@").getItem(1)).withColumn("channel_id", split(col("key"), "@").getItem(2)).withColumn("product_id", split(col("key"), "@").getItem(3))
final_data = final_data.drop("key")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check if reference master is uploaded or not

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

dbutils.widgets.text("ref_master_flag", "","")
ref_master_flag = dbutils.widgets.get("ref_master_flag")
ref_master_flag = int(ref_master_flag)
print ("ref_master_flag:",ref_master_flag)

# COMMAND ----------

# MAGIC %md
# MAGIC ####To get input and forecast level parameters

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

filtered_future_forecast_master = final_data.toPandas()

# COMMAND ----------

#Read the heirarchy master for reverse hierarchy
d_file = spark.read.jdbc(url=url, table= "hierarchy", properties = properties)
d_file.createOrReplaceTempView('d_file')
d_file = spark.sql("SELECT * from d_file where domain_id = '{}'".format(domain))
d_file = d_file.toPandas()

# COMMAND ----------

try:
    pipeline_flag = 0
    if ref_master_flag == 1 :
        #Reading reference master 
        reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{domain}') as reference_master",properties = properties)
        print("The count of reference_master",reference_master.count())
        if reference_master.count() == 0:
            raise Exception
        reference_master_split = reference_master.toPandas()
        p_file = d_file[d_file['hierarchy_type'] == 'product'].copy()
        c_file = d_file[d_file['hierarchy_type'] == 'channel'].copy()
        o_file = d_file[d_file['hierarchy_type'] == 'org_unit'].copy()

        p_file = generate_hierarchy(p_file, 'product')
        c_file = generate_hierarchy(c_file, 'channel')
        o_file = generate_hierarchy(o_file, 'org_unit')

        #Reference master filtering
        hierarchy_master = p_file[p_file[f'product_{product_forecast_level}'].isin(reference_master_split['product_id'])]
        filtered_future_forecast_master = filtered_future_forecast_master[filtered_future_forecast_master['product_id'].isin(hierarchy_master[f'product_{product_input_level}'])]
        hierarchy_master = c_file[c_file[f'channel_{channel_forecast_level}'].isin(reference_master_split['channel_id'])]
        filtered_future_forecast_master = filtered_future_forecast_master[filtered_future_forecast_master['channel_id'].isin(hierarchy_master[f'channel_{channel_input_level}'])]
        hierarchy_master = o_file[o_file[f'org_unit_{org_unit_forecast_level}'].isin(reference_master_split['org_unit_id'])]
        filtered_future_forecast_master = filtered_future_forecast_master[filtered_future_forecast_master['org_unit_id'].isin(hierarchy_master[f'org_unit_{org_unit_input_level}'])]
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
# MAGIC ## Aggregation

# COMMAND ----------

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

    if 'product_id' not in df.columns:
        df[["org_unit_id", "channel_id", "product_id"]] = df.key.str.split("@", expand=True)
    
    hm = generate_hierarchy(prod_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{product_input_level}',f'{suffix}_{product_output_level}']]
    df = pd.merge(df, hm, left_on=['product_id'], right_on=f'{suffix}_{int(product_input_index)}', how='left')
    df.drop(columns={'product_id',f'{suffix}_{int(product_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(product_output_index)}': 'product_id'}, inplace=True)
    df['tf'] = df['tf'].astype(float)
    df['bf'] = df['bf'].astype(float)
    df = df.groupby(['domain_id', 'org_unit_id','channel_id','product_id','period'])['tf','bf'].sum().reset_index()
#     df = df[['domain_id', 'period','tf','bf']]
    return df

# COMMAND ----------

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

    if 'channel_id' not in df.columns:
        df[["org_unit_id", "channel_id", "product_id"]] = df.key.str.split("@", expand=True)
    
    hm = generate_hierarchy(chann_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{channel_input_level}',f'{suffix}_{channel_output_level}']]
    df = pd.merge(df, hm, left_on=['channel_id'], right_on=f'{suffix}_{int(channel_input_index)}', how='left')
    df.drop(columns={'channel_id',f'{suffix}_{int(channel_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(channel_output_index)}': 'channel_id'}, inplace=True)
    df['tf'] = df['tf'].astype(float)
    df['bf'] = df['bf'].astype(float)
    df = df.groupby(['domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period'])['tf', 'bf'].sum().reset_index()
#     df = df[['domain_id','period','tf','bf']]
    return df

# COMMAND ----------

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

    if 'org_unit_id' not in df.columns:
        df[["org_unit_id", "channel_id", "product_id"]] = df.key.str.split("@", expand=True)
    
    hm = generate_hierarchy(org_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{org_unit_input_level}',f'{suffix}_{org_unit_output_level}']]
    df = pd.merge(df, hm, left_on=['org_unit_id'], right_on=f'{suffix}_{int(org_unit_input_index)}', how='left')
    df.drop(columns={'org_unit_id',f'{suffix}_{int(org_unit_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(org_unit_output_index)}': 'org_unit_id'}, inplace=True)
    df['tf'] = df['tf'].astype(float)
    df['bf'] = df['bf'].astype(float)
    df = df.groupby(['domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period'])['tf', 'bf'].sum().reset_index()
#     df = df[['domain_id','period','tf','bf']]
    return df

# COMMAND ----------

def future_forecast_agg_function(df1, Heirarchy, product_output_level, channel_output_level, org_unit_output_level, product_input_level,
                                 channel_input_level, org_unit_input_level):
    df = df1.reset_index(drop=True)
    df['product_id'] = df['product_id'].astype(str)

    if product_input_level != product_output_level:
        level_difference = product_output_level - product_input_level
        if level_difference > 0:
            df = product_aggregator(df, Heirarchy, product_input_level, product_output_level)
        elif level_difference < 0:
            print("Product Level aggregation is not required")

    if channel_input_level != channel_output_level:
        level_difference = channel_output_level - channel_input_level
        if level_difference > 0:
            df = channel_aggregator(df, Heirarchy, channel_input_level, channel_output_level)
        elif level_difference < 0:
            print("Channel level aggregation is not required")

    if org_unit_input_level != org_unit_output_level:
        level_difference = org_unit_output_level - org_unit_input_level
        if level_difference > 0:
            df = org_unit_aggregator(df, Heirarchy, org_unit_input_level, org_unit_output_level)
        elif level_difference < 0:
            print("Org_Unit level aggregation is not required")
    df['key'] = df['org_unit_id'] + '@' + df['channel_id'] + '@' + df['product_id']
    df = df[['domain_id', 'key', 'period', 'tf', 'bf']]
    df['tf'] = np.round(df['tf'], 0)
    df['bf'] = np.round(df['bf'], 0)
    df.drop_duplicates(keep='first', inplace=True)
    df = df.reset_index(drop=True)
    return df

# COMMAND ----------

future_forecast_master1 = future_forecast_agg_function(filtered_future_forecast_master, d_file, product_output_level=product_forecast_level, channel_output_level=channel_forecast_level,
                                      org_unit_output_level=org_unit_forecast_level, product_input_level=product_input_level,
                                      channel_input_level=channel_input_level,
                                      org_unit_input_level=org_unit_input_level)

# COMMAND ----------

future_forecast_master1.count()

# COMMAND ----------

# run_id = 15
future_forecast_master1['run_id'] = run_id

# COMMAND ----------

if ref_master_flag == 1 :
    reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{domain}') as   reference_master",properties = properties)
    reference_master = reference_master.toPandas()
    reference_master['key']  = reference_master['org_unit_id'] + '@' + reference_master['channel_id'] + '@' + reference_master['product_id']
    future_forecast_master1 = future_forecast_master1[future_forecast_master1['key'].isin(reference_master['key'])]

# COMMAND ----------

future_forecast_master1.display()

# COMMAND ----------

outSchema2 = StructType([StructField('domain_id',StringType(),True),
                       StructField('key',StringType(),True),
                       StructField('period',StringType(),True),
                       StructField('tf',FloatType(),True),
                       StructField('bf',FloatType(),True),
                       StructField('run_id',StringType(),True)])

# COMMAND ----------

final_data3 = spark.createDataFrame(data=future_forecast_master1, schema=outSchema2)

# COMMAND ----------

db_ingestion(final_data3, 'future_forecast_master', 'append')
