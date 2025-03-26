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

df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)

# COMMAND ----------

# Trimming the column names
col_ls = []
for coln in df.columns:
    tmp = "".join(coln.rstrip().lstrip())
    col_ls.append(tmp)
df = df.toDF(*col_ls)

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 7
df_ph7 = df.withColumnRenamed("Planning Hierarchy 7","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 7 Description","description").select('hierarchy_value', 'description')\
.withColumn("level_id",lit(5))\
.withColumn("parent_value",lit(None).cast(StringType()))

df_ph7 = df_ph7.select('parent_value', 'hierarchy_value', 'description', 'level_id')

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 6
df_ph6 = df.withColumnRenamed("Planning Hierarchy 6","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 6 Description","description")\
    .withColumnRenamed("Planning Hierarchy 7","parent_value").select('parent_value','hierarchy_value','description').withColumn("level_id",lit(4))

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 5
df_ph5 = df.withColumnRenamed("Planning Hierarchy 5","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 5 Description","description")\
    .withColumnRenamed("Planning Hierarchy 6","parent_value").select('parent_value','hierarchy_value','description').withColumn("level_id",lit(3))

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 4
df_ph4 = df.withColumnRenamed("Planning Hierarchy 4","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 4 Description","description")\
    .withColumnRenamed("Planning Hierarchy 5","parent_value").select('parent_value','hierarchy_value','description').withColumn("level_id",lit(2))

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 3
df_ph3 = df.withColumnRenamed("Planning Hierarchy 3","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 3 Description","description")\
    .withColumnRenamed("Planning Hierarchy 4","parent_value").select('parent_value','hierarchy_value','description').withColumn("level_id",lit(1))

# COMMAND ----------

# DBTITLE 1,Planning Hierarchy 2
df_ph2 = df.withColumnRenamed("Planning Hierarchy 2","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 2 Description","description")\
    .withColumnRenamed("Planning Hierarchy 3","parent_value").select('parent_value','hierarchy_value','description').withColumn("level_id",lit(0))

# COMMAND ----------

# DBTITLE 1,Union-Product
df_product = (df_ph7.union(df_ph6).union(df_ph5).union(df_ph4).union(df_ph3)).union(df_ph2)

# COMMAND ----------

#Adding a columns hierarchy_type
df_product=df_product.withColumn("hierarchy_type",lit("product"))

# COMMAND ----------

df_product = df_product.dropDuplicates(['hierarchy_value','description','level_id','parent_value'])
df_product = df_product.select('hierarchy_type','hierarchy_value','parent_value','level_id','description')

# COMMAND ----------

# DBTITLE 1,Channel
 #Renaming the columns
df_channel = df.withColumnRenamed("Demand Stream","description").select('description').withColumn("level_id",lit(0))\
.withColumn("parent_value",lit(None).cast(StringType())).withColumn("hierarchy_type",lit("channel"))

# COMMAND ----------

# df_channel=df_channel.withColumn('description', regexp_replace('description', ' ', ''))

# COMMAND ----------

df_channel = df_channel.withColumn("hierarchy_value",lit(df_channel.description))

# COMMAND ----------

df_channel=df_channel.dropDuplicates(['hierarchy_value',"description",'level_id','parent_value'])
df_channel = df_channel.select('hierarchy_type','hierarchy_value','parent_value','level_id','description')

# COMMAND ----------

# DBTITLE 1,Region
#Renaming the columns
df_region = df.withColumnRenamed("Region","hierarchy_value")\
              .withColumnRenamed("Region Description","description").select('hierarchy_value','description').withColumn("level_id",lit(2))\
.withColumn("parent_value",lit(None).cast(StringType()))

df_region = df_region.select('hierarchy_value', 'description', 'parent_value', 'level_id')

# COMMAND ----------

# DBTITLE 1,Cluster
#Renaming the columns
df_cluster = df.withColumnRenamed("Cluster","hierarchy_value")\
               .withColumnRenamed("Cluster Description","description")\
               .withColumnRenamed("Region","parent_value").select('hierarchy_value','description','parent_value')\
.withColumn("level_id",lit(1))

# COMMAND ----------

# DBTITLE 1,Country
#Renaming the columns
df_country = df.withColumnRenamed("Country","hierarchy_value")\
               .withColumnRenamed("Country Description","description")\
               .withColumnRenamed("Cluster","parent_value").select('hierarchy_value','description','parent_value').withColumn("level_id",lit(0))

# COMMAND ----------

# DBTITLE 1,Union 2
df_org_unit = (df_cluster.union(df_region)).union(df_country)

# COMMAND ----------

#Adding a hierarchy_type column
df_org_unit=df_org_unit.withColumn("hierarchy_type",lit('org_unit'))

# COMMAND ----------

df_org_unit=df_org_unit.dropDuplicates(['hierarchy_value',"description",'level_id','parent_value'])
df_org_unit = df_org_unit.select('hierarchy_type','hierarchy_value','parent_value','level_id','description')

# COMMAND ----------

# DBTITLE 1,Union
df_union = (df_product.union(df_channel)).union(df_org_unit)

# COMMAND ----------

# Replacing  ' ' with '_' 
df_union=df_union.withColumn('hierarchy_value', regexp_replace('hierarchy_value', ' ', '_')).withColumn('hierarchy_value', regexp_replace('hierarchy_value', '-', '_'))
df_union=df_union.withColumn('parent_value', regexp_replace('parent_value', ' ', '_')).withColumn('parent_value', regexp_replace('parent_value', '-', '_'))
# df_union=df_union.withColumn('description', regexp_replace('description', '-', '_'))

# COMMAND ----------

#Adding a domain
df_union=df_union.withColumn("domain_id",lit(domain_id))

# COMMAND ----------

# drop duplicates
df_union=df_union.dropDuplicates(['hierarchy_value','level_id','parent_value','hierarchy_type','domain_id'])

# COMMAND ----------

df_union = df_union.select('domain_id','hierarchy_type','hierarchy_value','parent_value','level_id','description')

# COMMAND ----------

# MAGIC %md
# MAGIC Removed at the end

# COMMAND ----------

# DBTITLE 1,DB Ingestion
def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

db_ingestion(df_union, 'hierarchy', 'append')
