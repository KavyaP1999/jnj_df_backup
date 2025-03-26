# Databricks notebook source
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

df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)

# COMMAND ----------

# Trimming the column names
col_ls = []
for coln in df.columns:
    tmp = "".join(coln.rstrip().lstrip())
    col_ls.append(tmp)
df = df.toDF(*col_ls)

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 2
df_ph2 = df.withColumnRenamed("Planning Hierarchy 2","product_id").withColumnRenamed("Planning Hierarchy 2 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(0))

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 3
df_ph3 = df.withColumnRenamed("Planning Hierarchy 3","product_id").withColumnRenamed("Planning Hierarchy 3 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(1))

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 7
df_ph7 = df.withColumnRenamed("Planning Hierarchy 7","product_id").withColumnRenamed("Planning Hierarchy 7 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(5))

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 6
df_ph6 = df.withColumnRenamed("Planning Hierarchy 6","product_id").withColumnRenamed("Planning Hierarchy 6 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(4))

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 5
df_ph5 = df.withColumnRenamed("Planning Hierarchy 5","product_id").withColumnRenamed("Planning Hierarchy 5 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(3))

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 4
df_ph4 = df.withColumnRenamed("Planning Hierarchy 4","product_id").withColumnRenamed("Planning Hierarchy 4 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(2))

# COMMAND ----------

# DBTITLE 1,Union
df_union = (df_ph7.union(df_ph6).union(df_ph5).union(df_ph4).union(df_ph3)).union(df_ph2)

# COMMAND ----------

#Adding a columns
df_union=df_union.withColumn("seasonality",lit("non-seasonal")).withColumn("status",lit("ACTIVE")).withColumn("touched",lit("TOUCHLESS"))

# COMMAND ----------

# Replacing  ' ' with '_' 
df_union=df_union.withColumn('product_id', regexp_replace('product_id', ' ', '_'))#.withColumn('product_id', regexp_replace('product_id', '-', '_'))

# COMMAND ----------

df_union=df_union.dropDuplicates(['product_id','product_desc','level_id','seasonality','status','touched'])

# COMMAND ----------

#Adding a domain
df_union=df_union.withColumn("domain_id",lit(domain_id))

# COMMAND ----------

df_union = df_union.select('domain_id','product_id','level_id','product_desc','touched','seasonality','status')

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

db_ingestion(df_union, 'product', 'append')
