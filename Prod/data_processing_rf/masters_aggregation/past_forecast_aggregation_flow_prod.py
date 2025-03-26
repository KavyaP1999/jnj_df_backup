# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, lower, col, split
from pyspark.sql import functions as f
from pyspark.sql import *
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import numpy as np
import pandas as pd
import traceback

# COMMAND ----------

ingestion_flag = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ####Get run_id and domain_id and other parameters 

# COMMAND ----------

dbutils.widgets.text("run_id", "","")
run_id = dbutils.widgets.get("run_id")
print ("run_id:",run_id)

dbutils.widgets.text("domain_id", "","")
domain_id = dbutils.widgets.get("domain_id")
print ("domain_id:",domain_id)

dbutils.widgets.text("ref_master_flag", "","")
ref_master_flag = dbutils.widgets.get("ref_master_flag")
ref_master_flag = int(ref_master_flag)
print ("ref_master_flag:",ref_master_flag)

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

# MAGIC %run /Shared/Prod/configuration_prodV2

# COMMAND ----------

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

# # new changes for NPF
# transitioned_past_forecast_master = spark.read.csv(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_past_forecast_master_{domain_id}.csv",inferSchema=True,header=True)
# #npf_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM npf_master WHERE domain_id = '{domain_id}') as npf_master_data", properties = properties)
# npf_master = spark.read.csv(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_npf_master_{domain_id}.csv",inferSchema=True,header=True)
# d_file = spark.read.jdbc(url=url, table= f"(SELECT * FROM hierarchy WHERE domain_id = '{domain_id}') as hierarchy_df", properties = properties)

# COMMAND ----------

# Old changes reverted back for NPF
transitioned_past_forecast_master = spark.read.csv(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_past_forecast_master_{domain_id}.csv",inferSchema=True,header=True)
npf_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM npf_master WHERE domain_id = '{domain_id}') as npf_master_data", properties = properties)
d_file = spark.read.jdbc(url=url, table= f"(SELECT * FROM hierarchy WHERE domain_id = '{domain_id}') as hierarchy_df", properties = properties)

# COMMAND ----------

npf_master = npf_master.withColumn('keys', concat_ws('@', npf_master.domain_id, npf_master.org_unit_id, npf_master.channel_id, npf_master.product_id))
npf_master = npf_master.select([lower('status').alias('status'), 'keys'])
final_data = transitioned_past_forecast_master.join(npf_master, transitioned_past_forecast_master.key==npf_master.keys, "inner")
final_data = final_data.filter((final_data.status=='active'))
final_data = final_data.drop('keys', 'status')
final_data = final_data.withColumn("org_unit_id", split(col("key"), "@").getItem(1)).withColumn("channel_id", split(col("key"), "@").getItem(2)).withColumn("product_id", split(col("key"), "@").getItem(3))
filtered_past_forecast_master = final_data.drop("key")
filtered_past_forecast_master = filtered_past_forecast_master.withColumn('bf_m00',filtered_past_forecast_master.bf_m00.cast(FloatType())).withColumn('bf_m01',filtered_past_forecast_master.bf_m01.cast(FloatType())).withColumn('bf_m02',filtered_past_forecast_master.bf_m02.cast(FloatType())).withColumn('bf_m03',filtered_past_forecast_master.bf_m03.cast(FloatType())).withColumn('tf_m00',filtered_past_forecast_master.tf_m00.cast(FloatType())).withColumn('tf_m01',filtered_past_forecast_master.tf_m01.cast(FloatType())).withColumn('tf_m02',filtered_past_forecast_master.tf_m02.cast(FloatType())).withColumn('tf_m03',filtered_past_forecast_master.tf_m03.cast(FloatType()))

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

d_file = d_file.toPandas()

# COMMAND ----------

p_file = generate_hierarchy(d_file[d_file['hierarchy_type'] == 'product'], 'product')
p_file = spark.createDataFrame(p_file)

# COMMAND ----------

try:
    if ref_master_flag == 1 :
        #Reading reference master 
        reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{domain_id}') as reference_master",properties = properties)
        print("The count of reference_master",reference_master.count())

#         p_file = d_file.filter(d_file.hierarchy_type == 'product')
#         c_file = d_file.filter(d_file.hierarchy_type == 'channel')
#         o_file = d_file.filter(d_file.hierarchy_type == 'org_unit')
        
        p_file = generate_hierarchy(d_file[d_file['hierarchy_type'] == 'product'], 'product')
        c_file = generate_hierarchy(d_file[d_file['hierarchy_type'] == 'channel'], 'channel')
        o_file = generate_hierarchy(d_file[d_file['hierarchy_type'] == 'org_unit'], 'org_unit')

        p_file = spark.createDataFrame(p_file)
        c_file = spark.createDataFrame(c_file)
        o_file = spark.createDataFrame(o_file)
        #Reference master filtering
#         l = set([data[0] for data in reference_master.select('product_id').collect()])
        l = reference_master.select('product_id').rdd.flatMap(lambda x:x).collect()
        hierarchy_master = p_file.filter(col(f'product_{product_forecast_level}').isin(l))
#         l = set([data[0] for data in hierarchy_master.select(col(f'product_{product_input_level}')).collect()])
        l = hierarchy_master.select(col(f'product_{product_input_level}')).rdd.flatMap(lambda x:x).collect()
        filtered_past_forecast_master = filtered_past_forecast_master.filter(filtered_past_forecast_master.product_id.isin(l))
        
#         l = set([data[0] for data in reference_master.select('channel_id').collect()])
        l = reference_master.select('channel_id').rdd.flatMap(lambda x:x).collect()
        hierarchy_master = c_file.filter(col(f'channel_{channel_forecast_level}').isin(l))
#         l = set([data[0] for data in hierarchy_master.select(col(f'channel_{channel_input_level}')).collect()])
        l = hierarchy_master.select(col(f'channel_{channel_input_level}')).rdd.flatMap(lambda x:x).collect()
        filtered_past_forecast_master = filtered_past_forecast_master.filter(filtered_past_forecast_master.channel_id.isin(l))
        
#         l = list(set([data[0] for data in reference_master.select('org_unit_id').collect()]))
        l = reference_master.select('org_unit_id').rdd.flatMap(lambda x:x).collect()
        hierarchy_master = o_file.filter(col(f'org_unit_{org_unit_forecast_level}').isin(l))
#         l = set([data[0] for data in hierarchy_master.select(col(f'org_unit_{org_unit_input_level}')).collect()])
        l = hierarchy_master.select(col(f'org_unit_{org_unit_input_level}')).rdd.flatMap(lambda x:x).collect()
        filtered_past_forecast_master = filtered_past_forecast_master.filter(filtered_past_forecast_master.org_unit_id.isin(l))
except:
    traceback.print_exc()
    10/0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregation

# COMMAND ----------

def product_aggregator(df1, Heirarchy, product_input_level, product_output_level):
    hm = Heirarchy.select(col(f'product_{product_input_level}'),col(f'product_{product_output_level}'))
    df = df1.join(hm, df1.product_id == hm[f'product_{int(product_input_level)}'], how='left').drop('product_id',f'product_{int(product_input_level)}').withColumnRenamed(f'product_{int(product_output_level)}','product_id')
    df = df.withColumn('bf_m00',col('bf_m00').cast(FloatType())).withColumn('bf_m01',col('bf_m01').cast(FloatType())).withColumn('bf_m02',col('bf_m02').cast(FloatType())).withColumn('bf_m03',col('bf_m03').cast(FloatType())).withColumn('tf_m00',col('tf_m00').cast(FloatType())).withColumn('tf_m01',col('tf_m01').cast(FloatType())).withColumn('tf_m02',col('tf_m02').cast(FloatType())).withColumn('tf_m03',col('tf_m03').cast(FloatType()))

    df = df.groupBy("domain_id",'org_unit_id','channel_id','product_id','period').sum('bf_m00', 'bf_m01','bf_m02','bf_m03','tf_m00','tf_m01','tf_m02','tf_m03').withColumnRenamed('sum(bf_m00)','bf_m00').withColumnRenamed('sum(bf_m01)','bf_m01').withColumnRenamed('sum(bf_m02)','bf_m02').withColumnRenamed('sum(bf_m03)','bf_m03').withColumnRenamed('sum(tf_m00)','tf_m00').withColumnRenamed('sum(tf_m01)','tf_m01').withColumnRenamed('sum(tf_m02)','tf_m02').withColumnRenamed('sum(tf_m03)','tf_m03')
    df = df.dropDuplicates()
    return df

# COMMAND ----------

def channel_aggregator(df1, Heirarchy, channel_input_level, channel_output_level):
    hm = Heirarchy.select(col(f'channel_{channel_input_level}'),col(f'channel_{channel_output_level}'))
    df = df1.join(hm, df1.channel_id == hm[f'channel_{int(channel_input_level)}'], how='left').drop('channel_id',f'channel_{int(channel_input_level)}').withColumnRenamed(f'channel_{int(channel_output_level)}','channel_id')
    df = df.withColumn('bf_m00',col('bf_m00').cast(FloatType())).withColumn('bf_m01',col('bf_m01').cast(FloatType())).withColumn('bf_m02',col('bf_m02').cast(FloatType())).withColumn('bf_m03',col('bf_m03').cast(FloatType())).withColumn('tf_m00',col('tf_m00').cast(FloatType())).withColumn('tf_m01',col('tf_m01').cast(FloatType())).withColumn('tf_m02',col('tf_m02').cast(FloatType())).withColumn('tf_m03',col('tf_m03').cast(FloatType()))
    df = df.groupBy("domain_id",'org_unit_id','channel_id','product_id','period').sum('bf_m00', 'bf_m01','bf_m02','bf_m03','tf_m00','tf_m01','tf_m02','tf_m03').withColumnRenamed('sum(bf_m00)','bf_m00').withColumnRenamed('sum(bf_m01)','bf_m01').withColumnRenamed('sum(bf_m02)','bf_m02').withColumnRenamed('sum(bf_m03)','bf_m03').withColumnRenamed('sum(tf_m00)','tf_m00').withColumnRenamed('sum(tf_m01)','tf_m01').withColumnRenamed('sum(tf_m02)','tf_m02').withColumnRenamed('sum(tf_m03)','tf_m03')
    df = df.dropDuplicates()
    return df

# COMMAND ----------

def org_unit_aggregator(df1, Heirarchy, org_unit_input_level, org_unit_output_level):
    hm = Heirarchy.select(col(f'org_unit_{org_unit_input_level}'),col(f'org_unit_{org_unit_output_level}'))
    df = df1.join(hm, df1.org_unit_id == hm[f'org_unit_{int(org_unit_input_level)}'], how='left').drop('org_unit_id',f'org_unit_{int(org_unit_input_level)}').withColumnRenamed(f'org_unit_{int(org_unit_output_level)}','org_unit_id')
    df = df.withColumn('bf_m00',col('bf_m00').cast(FloatType())).withColumn('bf_m01',col('bf_m01').cast(FloatType())).withColumn('bf_m02',col('bf_m02').cast(FloatType())).withColumn('bf_m03',col('bf_m03').cast(FloatType())).withColumn('tf_m00',col('tf_m00').cast(FloatType())).withColumn('tf_m01',col('tf_m01').cast(FloatType())).withColumn('tf_m02',col('tf_m02').cast(FloatType())).withColumn('tf_m03',col('tf_m03').cast(FloatType()))
    df = df.groupBy("domain_id",'org_unit_id','channel_id','product_id','period').sum('bf_m00', 'bf_m01','bf_m02','bf_m03','tf_m00','tf_m01','tf_m02','tf_m03').withColumnRenamed('sum(bf_m00)','bf_m00').withColumnRenamed('sum(bf_m01)','bf_m01').withColumnRenamed('sum(bf_m02)','bf_m02').withColumnRenamed('sum(bf_m03)','bf_m03').withColumnRenamed('sum(tf_m00)','tf_m00').withColumnRenamed('sum(tf_m01)','tf_m01').withColumnRenamed('sum(tf_m02)','tf_m02').withColumnRenamed('sum(tf_m03)','tf_m03')
    df = df.dropDuplicates()
    return df

# COMMAND ----------

def past_forecast_agg_function(df, Heirarchy, product_output_level, channel_output_level, org_unit_output_level, product_input_level=0,
                         channel_input_level=0, org_unit_input_level=0):   
    print('Past Forecast Aggregation Started')

    if product_input_level != product_output_level:
        level_difference = product_output_level - product_input_level
        if level_difference > 0:
            p_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'product'], 'product')
            p_file = spark.createDataFrame(p_file)
            df = product_aggregator(df, p_file, product_input_level, product_output_level)
        elif level_difference < 0:
            print("Product Level aggregation is not required")

    if channel_input_level != channel_output_level:
        level_difference = channel_output_level - channel_input_level
        if level_difference > 0:
            c_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'channel'], 'channel')
            c_file = spark.createDataFrame(c_file)
            df = channel_aggregator(df, c_file, channel_input_level, channel_output_level)
        elif level_difference < 0:
            print("Channel level aggregation is not required")

    if org_unit_input_level != org_unit_output_level:
        level_difference = org_unit_output_level - org_unit_input_level
        if level_difference > 0:
            o_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit'], 'org_unit')
            o_file = spark.createDataFrame(o_file)
            df = org_unit_aggregator(df, o_file, org_unit_input_level, org_unit_output_level)
        elif level_difference < 0:
            print("Org_Unit level aggregation is not required")
    df = df.withColumn('key',concat_ws('@',df.org_unit_id,df.channel_id,df.product_id))
    df = df.select('domain_id', 'key', 'period', 'bf_m00', 'bf_m01', 'bf_m02', 'bf_m03', 'tf_m00', 'tf_m01', 'tf_m02','tf_m03') 
    df = df.withColumn('bf_m00',df.bf_m00.cast(FloatType())).withColumn('bf_m01',df.bf_m01.cast(FloatType())).withColumn('bf_m02',df.bf_m02.cast(FloatType())).withColumn('bf_m03',df.bf_m03.cast(FloatType())).withColumn('tf_m00',df.tf_m00.cast(FloatType())).withColumn('tf_m01',df.tf_m01.cast(FloatType())).withColumn('tf_m02',df.tf_m02.cast(FloatType())).withColumn('tf_m03',df.tf_m03.cast(FloatType()))
    df = df.select('domain_id','key','period',f.round('bf_m00',0).alias('bf_m00'),f.round('bf_m01',0).alias('bf_m01'),f.round('bf_m02',0).alias('bf_m02'),f.round('bf_m03',0).alias('bf_m03'),f.round('tf_m00',0).alias('tf_m00'),f.round('tf_m01',0).alias('tf_m01'),f.round('tf_m02',0).alias('tf_m02'),f.round('tf_m03',0).alias('tf_m03'))
    
    return df

# COMMAND ----------

past_forecast_master1 = past_forecast_agg_function(filtered_past_forecast_master, d_file, product_output_level=product_forecast_level, channel_output_level=channel_forecast_level,
                                      org_unit_output_level=org_unit_forecast_level, product_input_level=product_input_level,
                                      channel_input_level=channel_input_level,
                                      org_unit_input_level=org_unit_input_level)

# COMMAND ----------

past_forecast_master1 = past_forecast_master1.withColumn('run_id',f.lit(run_id))

# COMMAND ----------

if ref_master_flag == 1 :
    reference_master = reference_master.withColumn('key',concat_ws('@',reference_master.org_unit_id,reference_master.channel_id,reference_master.product_id))
    l = set([data[0] for data in reference_master.select('key').collect()])
    past_forecast_master1 = past_forecast_master1.filter(col('key').isin(l))

# COMMAND ----------

if ingestion_flag ==1:
    db_ingestion(past_forecast_master1, 'past_forecast_master', 'append')
else:
    print("Ingestion not done")
