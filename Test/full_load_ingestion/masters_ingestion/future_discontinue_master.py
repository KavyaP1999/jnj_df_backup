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

df= spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/future_discontinue_master.csv",inferSchema=True,header=True)

# COMMAND ----------

# Replacing  ' ' with '_' 
df=df.withColumn('channel_id', regexp_replace('channel_id', ' ', '_'))
df=df.withColumn('org_unit_id', regexp_replace('org_unit_id', ' ', '_'))
df=df.withColumn('product_id', regexp_replace('product_id', ' ', '_'))


# COMMAND ----------

# For Period Column
df = df.withColumn("period", F.regexp_replace(F.col("period") ,  "(\\d{4})(\\d{2})" , "$1M0$2" ))

# COMMAND ----------

df = df.withColumn('key',concat_ws('@',df.org_unit_id,df.channel_id,df.product_id))

# COMMAND ----------

#Replacing the ' ' with '_'
df=df.withColumn('key', regexp_replace('key', ' ', '_'))

# COMMAND ----------

#Adding a domain
df=df.withColumn("domain_id",lit(domain_id))

# COMMAND ----------

df_new=df.drop("org_unit_id","channel_id","product_id")

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

db_ingestion(df_new, 'future_discontinue_master', 'append')
