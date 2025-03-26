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

spark = SparkSession.builder.appName("demo").getOrCreate()

# COMMAND ----------

# MAGIC %run /Shared/Prod/configuration_prodV2

# COMMAND ----------

df = spark.read.csv(f"/mnt/customerdata/{env}/Incremental_Load/{domain_id}/future_forecast_master.csv",inferSchema=True,header=True)

# COMMAND ----------

df = df.withColumn("org_unit_id",col("org_unit_id").cast(StringType())) \
    .withColumn("product_id",col("product_id").cast(StringType())) \
    .withColumn("channel_id",col("channel_id").cast(StringType()))\
    .withColumn("forecast_type",col("forecast_type").cast(StringType()))

# COMMAND ----------

#Triming org_unit_id,channel_id,product_id column
df = df.withColumn("org_unit_id", trim(df.org_unit_id))
df = df.withColumn("channel_id", trim(df.channel_id))
df = df.withColumn("product_id", trim(df.product_id))

# COMMAND ----------

# Replacing  ' ' with '_' 
df=df.withColumn('channel_id', regexp_replace('channel_id', ' ', '_'))
df=df.withColumn('org_unit_id', regexp_replace('org_unit_id', ' ', '_'))
df=df.withColumn('product_id', regexp_replace('product_id', ' ', '_'))

# COMMAND ----------

#Key
df= df.withColumn('key',concat_ws('@',df.org_unit_id,df.channel_id,df.product_id))

# COMMAND ----------

# Replacing  ' ' with '_' 
df=df.withColumn('key', regexp_replace('key', ' ', '_'))

# COMMAND ----------

#Dropping the "org_unit_id","channel_id","product_id" columns
df=df.drop("org_unit_id","channel_id","product_id")

# COMMAND ----------

new_list = [col for col in df.columns if col.startswith('20')]

for col_name in new_list:
    df = df.withColumn(col_name, col(col_name).cast(DecimalType(20,10)))

# COMMAND ----------

filtered_col = new_list + ["key","forecast_type"]
print(filtered_col)

# COMMAND ----------

df = df.select(*filtered_col)

# COMMAND ----------

new_df = spark.createDataFrame(new_list, StringType())
new_df = new_df.withColumnRenamed("value","value_new")
new_df = new_df.withColumn("value_new", F.regexp_replace(F.col("value_new") ,  "(\\d{4})(\\d{2})" , "$1M0$2" ))

# COMMAND ----------

new_names = new_df.select("value_new").rdd.flatMap(lambda x: x).collect()
new_names = new_names + ["key","forecast_type"]
df = df.toDF(*new_names)

# COMMAND ----------

new_list = [col for col in df.columns if col.startswith('20')]

l = []
for i in range(len(new_list)):
    l.append("'{}'".format(new_list[i]) + "," + new_list[i])
fin = ', '.join(l)

# COMMAND ----------

df.registerTempTable("df_Temp")

# COMMAND ----------

n = len(new_list)

# COMMAND ----------

df_future=spark.sql("SELECT key, forecast_type, stack({0},{1}) AS (period, value) FROM df_Temp".format(n,fin))

# COMMAND ----------

#Converting a 'forecast_type' column into lower case
df_future = df_future.withColumn("forecast_type",func.lower(func.col("forecast_type")))


# COMMAND ----------

#Pivot
df_pivot =df_future.groupBy("key","period").pivot("forecast_type").sum("value")


# COMMAND ----------

df_pivot = df_pivot.fillna(0, subset=['bf','tf'])

# COMMAND ----------

#Adding a domain
df_pivot=df_pivot.withColumn("domain_id",lit(domain_id))

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

db_ingestion(df_pivot, 'raw_future_forecast_master', 'append')
