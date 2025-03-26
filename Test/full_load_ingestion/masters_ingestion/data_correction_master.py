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

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

data_correction = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/data_correction.csv",inferSchema=True,header=True)

# COMMAND ----------

data_correction = data_correction.withColumn("org_unit_id",col("org_unit_id").cast(StringType())) \
    .withColumn("product_id",col("product_id").cast(StringType())) \
    .withColumn("channel_id",col("channel_id").cast(StringType()))\
    .withColumn("period",col("period").cast(StringType()))\
    .withColumn("historical_sale",col("historical_sale").cast(IntegerType())) \
    .withColumn("action",col("action").cast(StringType()))\
    .withColumn("to_period",col("to_period").cast(StringType()))

# COMMAND ----------

# Replacing  ' ' with '_' 
data_correction=data_correction.withColumn('channel_id', regexp_replace('channel_id', ' ', '_'))
data_correction=data_correction.withColumn('org_unit_id', regexp_replace('org_unit_id', ' ', '_'))
data_correction=data_correction.withColumn('product_id', regexp_replace('product_id', ' ', '_'))

# COMMAND ----------

# For Period Column
data_correction = data_correction.withColumn("period", F.regexp_replace(F.col("period") ,  "(\\d{4})(\\d{2})" , "$1M0$2" ))

# COMMAND ----------

# For to_period Column
data_correction = data_correction.withColumn("to_period", F.regexp_replace(F.col("to_period") ,  "(\\d{4})(\\d{2})" , "$1M0$2" ))

# COMMAND ----------

#Adding a domain_id
data_correction=data_correction.withColumn("domain_id",lit(domain_id))

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

db_ingestion(data_correction, 'data_correction', 'append')
