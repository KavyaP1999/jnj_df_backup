# Databricks notebook source
import sys
import os
import pandas as pd
import numpy as np
import random
import time
import pyspark
import pyspark.sql.functions as F
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

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/nts_master.csv",inferSchema=True,header=True)

# COMMAND ----------

df = df.withColumn("org_unit_id",col("org_unit_id").cast(StringType())) \
    .withColumn("product_id",col("product_id").cast(StringType())) \
    .withColumn("channel_id",col("channel_id").cast(StringType()))\
    .withColumn("m1",col("m1").cast(DecimalType(20,10))) \
    .withColumn("m2",col("m2").cast(DecimalType(20,10))) \
    .withColumn("m3",col("m3").cast(DecimalType(20,10))) \
    .withColumn("m4",col("m4").cast(DecimalType(20,10))) \
    .withColumn("m5", col("m5").cast(DecimalType(20,10))) \
    .withColumn("m6",col("m6").cast(DecimalType(20,10))) \
    .withColumn("m7",col("m7").cast(DecimalType(20,10))) \
    .withColumn("m8",col("m8").cast(DecimalType(20,10)))\
    .withColumn("m9",col("m9").cast(DecimalType(20,10))) \
    .withColumn("m10",col("m10").cast(DecimalType(20,10))) \
    .withColumn("m11",col("m11").cast(DecimalType(20,10))) \
    .withColumn("m12",col("m12").cast(DecimalType(20,10)))

# COMMAND ----------

df = df.fillna(0, subset=['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12'])

# COMMAND ----------

# Trimming the column names
col_ls = []
for coln in df.columns:
    tmp = "".join(coln.rstrip().lstrip())
    col_ls.append(tmp)
df = df.toDF(*col_ls)

# COMMAND ----------

#Triming org_unit_id,channel_id,product_id column
df = df.withColumn("org_unit_id", trim(df.org_unit_id)).withColumn("channel_id", trim(df.channel_id)).withColumn("product_id", trim(df.product_id))

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

#Adding a domain
df=df.withColumn("domain_id",lit(domain_id))

# COMMAND ----------

#Dropping the "org_unit_id","channel_id","product_id" columns
df_new=df.drop("org_unit_id","channel_id","product_id")

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

db_ingestion(df_new, 'raw_nts_master', 'append')
