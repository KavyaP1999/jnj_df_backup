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
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType, DoubleType


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

df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/npf_master.csv",inferSchema=True,header=True)
df = df.withColumn('org_unit_id',F.col('org_unit_id').cast(StringType())).withColumn('product_id',F.col('product_id').cast(StringType()))

# COMMAND ----------

# Trimming the column names
col_ls = []
for coln in df.columns:
    tmp = "".join(coln.rstrip().lstrip())
    col_ls.append(tmp)
df = df.toDF(*col_ls)

# COMMAND ----------

df = df.withColumnRenamed("Org_unit_id","org_unit_id") \
    .withColumnRenamed("Channel_id","channel_id") \
    .withColumnRenamed("Product_id","product_id") \
    .withColumnRenamed("Status","status")

# COMMAND ----------

#Triming channel_id,product_id org_unit_id column
df = df.withColumn("org_unit_id", trim(df.org_unit_id))
df = df.withColumn("channel_id", trim(df.channel_id))
df = df.withColumn("product_id", trim(df.product_id))

# COMMAND ----------

# Replacing  ' ' with '_' 
df=df.withColumn('channel_id', regexp_replace('channel_id', ' ', '_'))
df=df.withColumn('org_unit_id', regexp_replace('org_unit_id', ' ', '_'))
df=df.withColumn('product_id', regexp_replace('product_id', ' ', '_'))


# COMMAND ----------

#Adding a domain
df=df.withColumn("domain_id",lit(domain_id))

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

db_ingestion(df, 'npf_master', 'append')
