# Databricks notebook source
import sys
import os
import pandas as pd
import numpy as np
import random
import time
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.functions as func
from collections import defaultdict
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import concat, col, lit, concat_ws,split
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType, DoubleType,DecimalType
from pyspark.sql.functions import expr

# COMMAND ----------

dbutils.widgets.text("domain_id", "","")
domain_id = dbutils.widgets.get("domain_id")
print ("domain_id:",domain_id)

# COMMAND ----------

# MAGIC %run /Shared/Prod/configuration_prodV2

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)

# COMMAND ----------

#Renaming the columns
df = df.withColumnRenamed("Country","org_unit_id") \
    .withColumnRenamed("Planning Hierarchy 2","product_id")\
    .withColumnRenamed("Demand Stream","channel_id")

# COMMAND ----------

df = df.withColumn("org_unit_id",col("org_unit_id").cast(StringType())) \
    .withColumn("product_id",col("product_id").cast(StringType())) \
    .withColumn("channel_id",col("channel_id").cast(StringType()))

# COMMAND ----------

new_list = [col for col in df.columns if col.startswith('20')]

for col_name in new_list:
    df = df.withColumn(col_name, col(col_name).cast(DecimalType(20,10)))

# COMMAND ----------

filtered_col = new_list + ["org_unit_id","channel_id","product_id"]
print(filtered_col)

# COMMAND ----------

df = df.select(*filtered_col)

# COMMAND ----------

new_df = spark.createDataFrame(new_list, StringType())
new_df = new_df.withColumn("value", F.regexp_replace(F.col("value") ,  "(\\d{4})(\\d{2})" , "$1M0$2" ))

# COMMAND ----------

new_names = new_df.select("value").rdd.flatMap(lambda x: x).collect()
new_names = new_names + ["org_unit_id","channel_id","product_id"]
df = df.toDF(*new_names)

# COMMAND ----------

#df.registerTempTable("df_Temp")
df.createOrReplaceTempView ('df_Temp')

# COMMAND ----------

new_list = [col for col in df.columns if col.startswith('20')]

l = []
for i in range(len(new_list)):
    l.append("'{}'".format(new_list[i]) + "," + new_list[i])
fin = ', '.join(l)


# COMMAND ----------

print(fin)

# COMMAND ----------

#Unpivoting The Columns
n = len(new_list)
df_raw = spark.sql("SELECT org_unit_id,product_id,channel_id, stack({0},{1}) AS (period, historical_sale) FROM df_Temp".format(n,fin))

# COMMAND ----------

#Adding a run_state='I3'
df_raw = df_raw.withColumn("run_state",lit("I3"))
#Period Column
df_raw = df_raw.withColumn("period1", F.regexp_replace(F.col("period") ,  "(\\d{4})(\\d{2})" , "$1M0$2" ))
#Dropping the "period" columns
df_raw = df_raw.drop("period")
#Renaming the 'period1' to 'period'
df_raw = df_raw.withColumnRenamed("period1","period")
# Replacing  ' ' with '_' 
df_raw = df_raw.withColumn('channel_id', regexp_replace('channel_id', ' ', '_')).withColumn('org_unit_id', regexp_replace('org_unit_id', ' ', '_')).withColumn('product_id', regexp_replace('product_id', ' ', '_'))
#Adding a columns forecast,segmentation,model_id,promotion
df_raw = df_raw.withColumn("forecast",lit(None).cast(StringType())).withColumn("segmentation",lit(None).cast(StringType())).withColumn("model_id",lit(None).cast(StringType())).withColumn("promotion",lit(None).cast(StringType()))

# COMMAND ----------

#Group By and aggregating the historical_sale
df_raw = df_raw.groupBy("run_state","org_unit_id","channel_id","product_id","period",'forecast','segmentation',"model_id","promotion").agg(F.sum('historical_sale').alias('historical_sale'))
df_raw = df_raw.fillna(0, subset=['historical_sale'])
#Rounding the historical_sale
df_raw = df_raw.withColumn("historical_sale", F.round(df_raw["historical_sale"], 0)).withColumn('domain_id',lit(domain_id))
df_raw = df_raw.select('domain_id','run_state','org_unit_id','channel_id','product_id','period','historical_sale','forecast','segmentation','model_id','promotion')

# COMMAND ----------

# DBTITLE 1,DB Ingestion
def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

db_ingestion(df_raw, 'raw_run_data', 'append')
