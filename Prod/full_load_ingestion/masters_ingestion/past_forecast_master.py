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


# COMMAND ----------

dbutils.widgets.text("domain_id", "","")
domain_id = dbutils.widgets.get("domain_id")
print ("domain_id:",domain_id)

# COMMAND ----------

# MAGIC %run /Shared/Prod/configuration_prodV2

# COMMAND ----------

df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/past_forecast_master.csv",inferSchema=True,header=True)

# COMMAND ----------

df = df.withColumn("org_unit_id",col("org_unit_id").cast(StringType())) \
    .withColumn("product_id",col("product_id").cast(StringType())) \
    .withColumn("channel_id",col("channel_id").cast(StringType()))\
    .withColumn("period",col("period").cast(StringType()))\
    .withColumn("bf_m00",col("bf_m00").cast(DecimalType(20,10))) \
    .withColumn("bf_m01",col("bf_m01").cast(DecimalType(20,10))) \
    .withColumn("bf_m02",col("bf_m02").cast(DecimalType(20,10))) \
    .withColumn("bf_m03",col("bf_m03").cast(DecimalType(20,10))) \
    .withColumn("tf_m00", col("tf_m00").cast(DecimalType(20,10))) \
    .withColumn("tf_m01",col("tf_m01").cast(DecimalType(20,10))) \
    .withColumn("tf_m02",col("tf_m02").cast(DecimalType(20,10))) \
    .withColumn("tf_m03",col("tf_m03").cast(DecimalType(20,10)))

# COMMAND ----------

# Trimming the column names
col_ls = []
for coln in df.columns:
    tmp = "".join(coln.rstrip().lstrip())
    col_ls.append(tmp)
df = df.toDF(*col_ls)

# COMMAND ----------

#Renaming the columns
df2 = df.withColumnRenamed("Country","org_unit_id") \
    .withColumnRenamed("Planning Hierarchy 2","product_id")\
    .withColumnRenamed("Demand Stream","channel_id")

# COMMAND ----------

#Triming org_unit_id,channel_id,product_id column
df2 = df2.withColumn("org_unit_id", trim(df2.org_unit_id)).withColumn("channel_id", trim(df2.channel_id)).withColumn("product_id", trim(df2.product_id))

# COMMAND ----------

# Replacing  ' ' with '_' 
df2 = df2.withColumn('channel_id', regexp_replace('channel_id', ' ', '_')).withColumn('org_unit_id', regexp_replace('org_unit_id', ' ', '_')).withColumn('product_id', regexp_replace('product_id', ' ', '_'))

# COMMAND ----------

#Period Column
df2=df2.withColumn("period", F.regexp_replace(F.col("period") ,  "(\\d{4})(\\d{2})" , "$1M0$2" ))

# COMMAND ----------

#Key
df2= df2.withColumn('key',concat_ws('@',df2.org_unit_id,df2.channel_id,df2.product_id))

# COMMAND ----------

#Dropping the "org_unit_id","channel_id","product_id" columns
df_new=df2.drop("org_unit_id","channel_id","product_id")

# COMMAND ----------

#Aggregation
df_new=df_new.groupBy("key","period").sum("bf_m00","bf_m01","bf_m02","bf_m03","tf_m00","tf_m01","tf_m02","tf_m03")

# COMMAND ----------

#Adding a domain
df_new=df_new.withColumn("domain_id",lit(domain_id))

# COMMAND ----------

#Renaming the columns
df_new = df_new.withColumnRenamed("sum(bf_m00)","bf_m00") \
      .withColumnRenamed("sum(bf_m01)","bf_m01")\
      .withColumnRenamed("sum(bf_m02)","bf_m02")\
      .withColumnRenamed("sum(bf_m03)","bf_m03")\
      .withColumnRenamed("sum(tf_m00)","tf_m00")\
      .withColumnRenamed("sum(tf_m01)","tf_m01")\
      .withColumnRenamed("sum(tf_m02)","tf_m02")\
      .withColumnRenamed("sum(tf_m03)","tf_m03")

# COMMAND ----------

# DBTITLE 1,Replacing The Null values
df_new = df_new.fillna(0, subset=['bf_m00','bf_m01','bf_m02','bf_m03','tf_m00','tf_m01','tf_m02','tf_m03'])

# COMMAND ----------

df_new = df_new.select('domain_id','key','period','bf_m00','bf_m01','bf_m02','bf_m03','tf_m00','tf_m01','tf_m02','tf_m03')

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

# DBTITLE 1,DB Ingestion
def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

db_ingestion(df_new, 'raw_past_forecast_master', 'append')
