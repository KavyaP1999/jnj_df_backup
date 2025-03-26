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

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

channel_db =  spark.read.jdbc(url=url, table="channel", properties=properties)
channel_db = channel_db.filter(channel_db.domain_id==domain_id)
channel_db = channel_db.limit(10)

# COMMAND ----------

df = spark.read.csv(f"/mnt/customerdata/{env}/Incremental_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)
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
channel_2 = df_new.select('domain_id','level_id','channel_id','channel_desc')

# COMMAND ----------

channel_2.display()

# COMMAND ----------

#Retruns the records which are there in incremental new data but not in full load exisiting data
channel_incr = channel_2.subtract(channel_db)

# COMMAND ----------

channel_incr.display()

# COMMAND ----------

l = set([data[0] for data in channel_incr.select('channel_id').collect()])

# COMMAND ----------

#Return the product_ids that doesn't exist in incremental load.
df_not_in_incr = channel_db.filter(~channel_db.channel_id.isin(l))

# COMMAND ----------

df_not_in_incr.display()

# COMMAND ----------

final_data = df_not_in_incr.union(channel_incr)
#final_data.display()

# COMMAND ----------

final_data.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/incremental_staging_data/{domain_id}_channel_master.csv") 
