# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import time
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from collections import defaultdict
from pyspark.sql.functions import concat, col, lit, concat_ws,split
import pyspark.sql.functions as F
from pyspark.sql.functions import when

# COMMAND ----------

# MAGIC %md
# MAGIC Reading configuration from configuration file

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

dbutils.widgets.text("domain_id", "","")
domain_id = dbutils.widgets.get("domain_id")

# COMMAND ----------

# MAGIC %md
# MAGIC Reading Masters from db

# COMMAND ----------

raw_future_forecast_master = spark.read.jdbc(url = url, table = f"(SELECT * FROM raw_future_forecast_master WHERE domain_id = '{domain_id}') as raw_future_forecast_master", properties = properties)
raw_nts_master = spark.read.jdbc(url = url, table = f"(SELECT * FROM raw_nts_master WHERE domain_id = '{domain_id}') as raw_nts_master", properties = properties)
raw_past_forecast_master = spark.read.jdbc(url = url, table = f"(SELECT * FROM raw_past_forecast_master WHERE domain_id = '{domain_id}') as raw_past_forecast_master", properties = properties)
raw_history_master = spark.read.jdbc(url = url, table = f"(SELECT * FROM raw_history_master WHERE domain_id = '{domain_id}') as raw_history_master", properties = properties)
data_correction_master = spark.read.jdbc(url = url, table = f"(SELECT * FROM data_correction WHERE domain_id = '{domain_id}') as data_correction_master", properties = properties)
transition_final_master = spark.read.jdbc(url = url, table = f"(SELECT * FROM transition_final_master WHERE domain_id = '{domain_id}') as transition_final_master", properties = properties)
raw_npf_master = spark.read.jdbc(url = url, table = f"(SELECT * FROM npf_master WHERE domain_id = '{domain_id}') as raw_npf_master", properties = properties)

# COMMAND ----------

transition_final_master = transition_final_master.toPandas() 
dictionary_old_final_sku = dict(zip(transition_final_master['old_sku'], transition_final_master['final_sku'])) #dictionary of old aku and final sku for masters

# COMMAND ----------

# ADDED NPF 
raw_npf_master = raw_npf_master.withColumn('key',concat_ws('@',raw_npf_master.domain_id,raw_npf_master.org_unit_id,raw_npf_master.channel_id,raw_npf_master.product_id))
#Replace values from Dictionary
#raw_npf_master.drop("org_unit_id","channel_id","product_id")
#raw_npf_master = raw_npf_master.dropDuplicates("key")
raw_npf_master = raw_npf_master.replace(dictionary_old_final_sku,subset=['key'])
raw_npf_master =raw_npf_master.withColumn('key2',concat_ws('@',raw_npf_master.domain_id,raw_npf_master.org_unit_id,raw_npf_master.channel_id,raw_npf_master.product_id))

                       


# COMMAND ----------

#new npf
df1 = raw_npf_master.withColumn('domain_id', split(raw_npf_master['key'], '@').getItem(0)) \
       .withColumn('org_unit_id', split(raw_npf_master['key'], '@').getItem(1)) \
       .withColumn('channel_id', split(raw_npf_master['key'], '@').getItem(2))\
       .withColumn('product_id', split(raw_npf_master['key'], '@').getItem(3))


# COMMAND ----------

df1 = df1.withColumn("status",
                     when(col("status")=='Active', 1)
                     .when(col("status")=='NPI', 2)
                     .when(col("status")=='Inactive', 3)
                     .otherwise(4))


# COMMAND ----------

from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
df1.withColumn("status",col("status").cast(IntegerType()))


# COMMAND ----------

df3 = df1.groupBy("org_unit_id","channel_id","product_id","domain_id").min("status")


# COMMAND ----------

# df3.display()

# COMMAND ----------

df4 = df3.withColumnRenamed("min(status)","status")

# COMMAND ----------

 df5 = df4.withColumn("status",
                     when(col("status")== 1, 'Active')
                     .when(col("status")==2, 'NPI')
                     .when(col("status")==3, 'Inactive')
                     .otherwise(4))

# COMMAND ----------

from pyspark.sql.functions import col, when
final = df5.orderBy(when(col("status") == "Active", 1)
           .when(col("status") == "NPI", 2)
           .when(col("status") == "Inactive", 3))

# COMMAND ----------

# final.display()

# COMMAND ----------

#final.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/npf_master.csv")

# COMMAND ----------

##new code npf
final.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_npf_master_{domain_id}.csv") 

# COMMAND ----------

#final.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/npf_master.csv")

# COMMAND ----------

#final.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/npf_master_{domain_id}.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC Transition raw_future_forecast_master

# COMMAND ----------

raw_future_forecast_master = raw_future_forecast_master.withColumn('key',concat_ws('@',raw_future_forecast_master.domain_id,raw_future_forecast_master.key))
#Replace values from Dictionary
raw_future_forecast_master = raw_future_forecast_master.replace(dictionary_old_final_sku,subset=['key'])
raw_future_forecast_master = raw_future_forecast_master.groupby('domain_id','key','period').agg(F.sum('bf').alias('bf'),F.sum('tf').alias('tf'))

# COMMAND ----------

raw_future_forecast_master.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_future_forecast_master_{domain_id}.csv") 

# COMMAND ----------

# MAGIC %md
# MAGIC Transition Raw_Nts_Master

# COMMAND ----------

raw_nts_master = raw_nts_master.withColumn('key',concat_ws('@',raw_nts_master.domain_id,raw_nts_master.key))
#Replace values from Dictionary
raw_nts_master = raw_nts_master.replace(dictionary_old_final_sku,subset=['key'])
raw_nts_master = raw_nts_master.groupby('domain_id','key').agg(F.sum('m1').alias('m1'),F.sum('m2').alias('m2'),F.sum('m3').alias('m3'),
                                                                                      F.sum('m4').alias('m4'),F.sum('m5').alias('m5'),F.sum('m6').alias('m6'),
                                                                                      F.sum('m7').alias('m7'),F.sum('m8').alias('m8'),F.sum('m9').alias('m9'),
                                                                                      F.sum('m10').alias('m10'),F.sum('m11').alias('m11'),F.sum('m12').alias('m12'))

# COMMAND ----------

raw_nts_master.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_nts_master_{domain_id}.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC Transition Raw_Past_Forecast_Master

# COMMAND ----------

raw_past_forecast_master = raw_past_forecast_master.withColumn('key',concat_ws('@',raw_past_forecast_master.domain_id,raw_past_forecast_master.key))
#Replace values from Dictionary
raw_past_forecast_master = raw_past_forecast_master.replace(dictionary_old_final_sku,subset=['key'])
raw_past_forecast_master = raw_past_forecast_master.groupby('domain_id','key','period').agg(F.sum('bf_m00').alias('bf_m00'),F.sum('bf_m01').alias('bf_m01'),
                                                                                           F.sum('bf_m02').alias('bf_m02'),F.sum('bf_m03').alias('bf_m03'),
                                                                                           F.sum('tf_m00').alias('tf_m00'),F.sum('tf_m01').alias('tf_m01'),
                                                                                           F.sum('tf_m02').alias('tf_m02'),F.sum('tf_m03').alias('tf_m03'))

# COMMAND ----------

raw_past_forecast_master.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_past_forecast_master_{domain_id}.csv") 

# COMMAND ----------

# MAGIC %md
# MAGIC Transition Raw_History_Master

# COMMAND ----------

raw_history_master = raw_history_master.withColumn('key',concat_ws('@',raw_history_master.domain_id,raw_history_master.org_unit_id,raw_history_master.channel_id,
                                                                  raw_history_master.product_id))
#Replace values from Dictionary
raw_history_master = raw_history_master.replace(dictionary_old_final_sku,subset=['key'])
raw_history_master = raw_history_master.select(['domain_id','key','run_state','period','segmentation','model_id','promotion','cdh','forecast','expost_sales'])
split_col = split(raw_history_master['key'], '@')
raw_history_master = raw_history_master.withColumn('domain_id', split_col.getItem(0)) \
   .withColumn('org_unit_id', split_col.getItem(1)) \
    .withColumn('channel_id', split_col.getItem(2)) \
    .withColumn('product_id', split_col.getItem(3))
# raw_history_master = raw_history_master.na.fill(value=0,subset=['forecast'])
raw_history_master = raw_history_master.groupby('domain_id','run_state', 'org_unit_id','channel_id','product_id','period','segmentation','model_id','promotion').agg(F.sum('cdh').alias('cdh'),F.sum('expost_sales').alias('expost_sales'))

# COMMAND ----------

raw_history_master = raw_history_master.withColumn("forecast", lit(None).cast(StringType()))

# COMMAND ----------

raw_history_master.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_history_master_{domain_id}.csv") 

# COMMAND ----------

# MAGIC %md
# MAGIC Transition Data_Correction_Master

# COMMAND ----------

data_correction_master = data_correction_master.withColumn('key',concat_ws('@',data_correction_master.domain_id,data_correction_master.org_unit_id,data_correction_master.channel_id,
                                                                  data_correction_master.product_id))

data_correction_master = data_correction_master.replace(dictionary_old_final_sku,subset=['key'])
data_correction_master = data_correction_master.select(['key','period','historical_sale','action','to_period'])
split_col = split(data_correction_master['key'], '@')
data_correction_master = data_correction_master.withColumn('domain_id', split_col.getItem(0)) \
   .withColumn('org_unit_id', split_col.getItem(1)) \
    .withColumn('channel_id', split_col.getItem(2)) \
    .withColumn('product_id', split_col.getItem(3))

# COMMAND ----------

data_correction_master = data_correction_master.withColumn('to_period',F.regexp_replace('to_period','M0',''))
data_correction_master = data_correction_master.withColumn('to_period',F.col('to_period').cast(IntegerType()))  #Type casting from string to integer as per schema definition
data_correction_master = data_correction_master.na.fill(value = 0,subset = ['to_period'])
data_correction_master = data_correction_master.groupBy('domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period').agg(F.sum("historical_sale").alias("historical_sale"),F.collect_set('action').alias('action'),
F.max("to_period").alias('to_period'))
data_correction_master = data_correction_master.withColumn("action", F.array_join("action", ","))
data_correction_master = data_correction_master.withColumn('to_period',F.col('to_period').cast(StringType()))
data_correction_master = data_correction_master.withColumn('historical_sale',F.col('historical_sale').cast(IntegerType()))
data_correction_master = data_correction_master.withColumn("to_period", F.regexp_replace(F.col("to_period") ,  "(\\d{4})(\\d{2})" , "$1M0$2" ))
data_correction_master = data_correction_master.withColumn('to_period', F.when(F.col('to_period') == '0', '').otherwise(F.col('to_period')))

# COMMAND ----------

data_correction_master.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_data_correction_{domain_id}.csv") 
