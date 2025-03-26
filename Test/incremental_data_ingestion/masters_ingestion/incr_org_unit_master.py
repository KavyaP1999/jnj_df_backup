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

org_unit_db =  spark.read.jdbc(url=url, table="org_unit", properties=properties)
org_unit_db = org_unit_db.filter(org_unit_db.domain_id==domain_id)
# org_unit_db = org_unit_db.limit(10)

# COMMAND ----------

#Reading the source file
df = spark.read.csv(f"/mnt/customerdata/{env}/Incremental_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)
#Renaming the columns 'Country' and 'Country Description'
df_country = df.withColumnRenamed("Country","org_unit_id") \
    .withColumnRenamed("Country Description","org_unit_desc")
df_country=df_country.select('org_unit_id', 'org_unit_desc')
#Adding a level_id=0
df_country=df_country.withColumn("level_id",lit(0))
#Renaming the columns 'Cluster' and 'Cluster Description'
df1_cluster = df.withColumnRenamed("Cluster","org_unit_id") \
    .withColumnRenamed("Cluster Description","org_unit_desc")
df1_cluster=df1_cluster.select('org_unit_id', 'org_unit_desc')
#Adding a level_id=1
df1_cluster=df1_cluster.withColumn("level_id",lit(1))
#Renaming the columns 'Region' and 'Region Description'
df2_region = df.withColumnRenamed("Region","org_unit_id") \
    .withColumnRenamed("Region Description","org_unit_desc")
df2_region=df2_region.select('org_unit_id', 'org_unit_desc')
#Adding a level_id=2
df2_region=df2_region.withColumn("level_id",lit(2))
unionDF = (df_country.union(df1_cluster)).union(df2_region)
unionDF=unionDF.dropDuplicates(['org_unit_id','org_unit_desc','level_id'])
# Replacing  ' ' with '_' 
unionDF=unionDF.withColumn('org_unit_id', regexp_replace('org_unit_id', ' ', '_')).withColumn("domain_id",lit(domain_id))
org_unit_2 = unionDF.select('domain_id','level_id','org_unit_id','org_unit_desc')    

# COMMAND ----------

org_unit_2.display()

# COMMAND ----------

#Retruns the records which are there in incremental new data but not in full load exisiting data
org_unit_incr = org_unit_2.subtract(org_unit_db)

# COMMAND ----------

org_unit_incr.display()

# COMMAND ----------

l = set([data[0] for data in org_unit_incr.select('org_unit_id').collect()])

# COMMAND ----------

#Return the product_ids that doesn't exist in incremental load.
df_not_in_incr = org_unit_db.filter(~org_unit_db.org_unit_id.isin(l))

# COMMAND ----------

df_not_in_incr.display()

# COMMAND ----------

final_data = df_not_in_incr.union(org_unit_incr)
#final_data.display()

# COMMAND ----------

final_data.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/incremental_staging_data/{domain_id}_org_unit_master.csv") 
