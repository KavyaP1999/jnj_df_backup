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

#Reading the source file
df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)

# COMMAND ----------

# DBTITLE 1,Country
#Renaming the columns 'Country' and 'Country Description'
df_country = df.withColumnRenamed("Country","org_unit_id") \
    .withColumnRenamed("Country Description","org_unit_desc")
    

# COMMAND ----------

df_country=df_country.select('org_unit_id', 'org_unit_desc')

# COMMAND ----------

#Adding a level_id=0
df_country=df_country.withColumn("level_id",lit(0))

# COMMAND ----------

# DBTITLE 1,Cluster
#Renaming the columns 'Cluster' and 'Cluster Description'
df1_cluster = df.withColumnRenamed("Cluster","org_unit_id") \
    .withColumnRenamed("Cluster Description","org_unit_desc")

# COMMAND ----------

df1_cluster=df1_cluster.select('org_unit_id', 'org_unit_desc')

# COMMAND ----------

#Adding a level_id=1
df1_cluster=df1_cluster.withColumn("level_id",lit(1))

# COMMAND ----------

# DBTITLE 1,Region
#Renaming the columns 'Region' and 'Region Description'
df2_region = df.withColumnRenamed("Region","org_unit_id") \
    .withColumnRenamed("Region Description","org_unit_desc")

# COMMAND ----------

df2_region=df2_region.select('org_unit_id', 'org_unit_desc')

# COMMAND ----------

#Adding a level_id=2
df2_region=df2_region.withColumn("level_id",lit(2))

# COMMAND ----------

# DBTITLE 1,Union
unionDF = (df_country.union(df1_cluster)).union(df2_region)

# COMMAND ----------

# DBTITLE 1,Dropping Duplicates
unionDF=unionDF.dropDuplicates(['org_unit_id','org_unit_desc','level_id'])

# COMMAND ----------

# Replacing  ' ' with '_' 
unionDF=unionDF.withColumn('org_unit_id', regexp_replace('org_unit_id', ' ', '_')).withColumn("domain_id",lit(domain_id))

# COMMAND ----------

unionDF = unionDF.select('domain_id','level_id','org_unit_id','org_unit_desc')

# COMMAND ----------

# DBTITLE 1,DB Ingestion
def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

db_ingestion(unionDF, 'org_unit', 'append')
