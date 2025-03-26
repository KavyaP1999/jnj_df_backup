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

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/transition_master.csv",inferSchema=True,header=True)

# COMMAND ----------

# Trimming the column names
col_ls = []
for coln in df.columns:
    tmp = "".join(coln.rstrip().lstrip())
    col_ls.append(tmp)
df = df.toDF(*col_ls)

# COMMAND ----------

# Replacing  ' ' with '_' 
df=df.withColumn('channel_id', regexp_replace('channel_id', ' ', '_'))
df=df.withColumn('org_unit_id', regexp_replace('org_unit_id', ' ', '_'))

# COMMAND ----------

#Trim for org_unit_id and channel_id
df = df.withColumn("org_unit_id", trim(df.org_unit_id)).withColumn("channel_id", trim(df.channel_id))

# COMMAND ----------

#Adding a domain
df=df.withColumn("domain_id",lit(domain_id))

# COMMAND ----------

df = df.select('domain_id','org_unit_id','channel_id','old_sku','new_sku')

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

db_ingestion(df, 'transition', 'append')
