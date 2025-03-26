# Databricks notebook source
import traceback
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
from functools import reduce
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType, DoubleType
from pyspark.sql.functions import expr

# COMMAND ----------

dbutils.widgets.text("domain_id", "","")
domain_id = dbutils.widgets.get("domain_id")
print ("domain_id:",domain_id)

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

df = spark.read.csv(f"/mnt/customerdata/{env}/Incremental_Load/{domain_id}/npf_master.csv",inferSchema=True,header=True)

# COMMAND ----------

# Trimming the column names
col_ls = []
for coln in df.columns:
    tmp = "".join(coln.rstrip().lstrip())
    col_ls.append(tmp)
df = df.toDF(*col_ls)

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

db_ingestion(df, 'npf_master', 'append')
