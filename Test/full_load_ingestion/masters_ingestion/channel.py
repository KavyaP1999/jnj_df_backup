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

df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)

# COMMAND ----------

#Renaming  "DemandStream" to "channel_desc"
df_new=df.withColumnRenamed("Demand Stream","channel_desc")
df_new=df_new.select(col("channel_desc"))
df_new=df_new.withColumn('channel_id',df_new.channel_desc)
# Replacing  ' ' with '_' 
df_new=df_new.withColumn('channel_id', regexp_replace('channel_id', ' ', '_'))
#Adding a domain
df_new=df_new.withColumn("domain_id",lit(domain_id))
#Adding a level_id
df_new=df_new.withColumn("level_id",lit('0'))
df_new=df_new.dropDuplicates(['channel_desc',"channel_id"])
df_new = df_new.select('domain_id','level_id','channel_id','channel_desc')

# COMMAND ----------

# DBTITLE 1,DB Ingestion
def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

db_ingestion(df_new, 'channel', 'append')
