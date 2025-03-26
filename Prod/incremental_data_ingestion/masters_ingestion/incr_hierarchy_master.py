# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, lower, col, split
from pyspark.sql import functions as f
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
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

# MAGIC %run /Shared/Prod/configuration_prodV2

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

hierarchy_db =  spark.read.jdbc(url=url, table="hierarchy", properties=properties)
hierarchy_db = hierarchy_db.filter(hierarchy_db.domain_id==domain_id)

# COMMAND ----------

df = spark.read.csv(f"/mnt/customerdata/{env}/Incremental_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)
# Trimming the column names
col_ls = []
for coln in df.columns:
    tmp = "".join(coln.rstrip().lstrip())
    col_ls.append(tmp)
df = df.toDF(*col_ls)
df_ph7 = df.withColumnRenamed("Planning Hierarchy 7","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 7 Description","description").select('hierarchy_value', 'description')\
.withColumn("level_id",lit(5))\
.withColumn("parent_value",lit(None).cast(StringType()))
df_ph7 = df_ph7.select('parent_value', 'hierarchy_value', 'description', 'level_id')
df_ph6 = df.withColumnRenamed("Planning Hierarchy 6","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 6 Description","description")\
    .withColumnRenamed("Planning Hierarchy 7","parent_value").select('parent_value','hierarchy_value','description').withColumn("level_id",lit(4))
df_ph5 = df.withColumnRenamed("Planning Hierarchy 5","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 5 Description","description")\
    .withColumnRenamed("Planning Hierarchy 6","parent_value").select('parent_value','hierarchy_value','description').withColumn("level_id",lit(3))
df_ph4 = df.withColumnRenamed("Planning Hierarchy 4","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 4 Description","description")\
    .withColumnRenamed("Planning Hierarchy 5","parent_value").select('parent_value','hierarchy_value','description').withColumn("level_id",lit(2))
df_ph3 = df.withColumnRenamed("Planning Hierarchy 3","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 3 Description","description")\
    .withColumnRenamed("Planning Hierarchy 4","parent_value").select('parent_value','hierarchy_value','description').withColumn("level_id",lit(1))
df_ph2 = df.withColumnRenamed("Planning Hierarchy 2","hierarchy_value") \
    .withColumnRenamed("Planning Hierarchy 2 Description","description")\
    .withColumnRenamed("Planning Hierarchy 3","parent_value").select('parent_value','hierarchy_value','description').withColumn("level_id",lit(0))
df_product = (df_ph7.union(df_ph6).union(df_ph5).union(df_ph4).union(df_ph3)).union(df_ph2)
#Adding a columns hierarchy_type
df_product=df_product.withColumn("hierarchy_type",lit("product"))
df_product = df_product.dropDuplicates(['hierarchy_value','description','level_id','parent_value'])
df_product = df_product.select('hierarchy_type','hierarchy_value','parent_value','level_id','description')
 #Renaming the columns
df_channel = df.withColumnRenamed("Demand Stream","description").select('description').withColumn("level_id",lit(0))\
.withColumn("parent_value",lit(None).cast(StringType())).withColumn("hierarchy_type",lit("channel"))
df_channel = df_channel.withColumn("hierarchy_value",lit(df_channel.description))
df_channel=df_channel.dropDuplicates(['hierarchy_value',"description",'level_id','parent_value'])
df_channel = df_channel.select('hierarchy_type','hierarchy_value','parent_value','level_id','description')
#Renaming the columns
df_region = df.withColumnRenamed("Region","hierarchy_value")\
              .withColumnRenamed("Region Description","description").select('hierarchy_value','description').withColumn("level_id",lit(2))\
.withColumn("parent_value",lit(None).cast(StringType()))

df_region = df_region.select('hierarchy_value', 'description', 'parent_value', 'level_id')
#Renaming the columns
df_cluster = df.withColumnRenamed("Cluster","hierarchy_value")\
               .withColumnRenamed("Cluster Description","description")\
               .withColumnRenamed("Region","parent_value").select('hierarchy_value','description','parent_value')\
.withColumn("level_id",lit(1))
#Renaming the columns
df_country = df.withColumnRenamed("Country","hierarchy_value")\
               .withColumnRenamed("Country Description","description")\
               .withColumnRenamed("Cluster","parent_value").select('hierarchy_value','description','parent_value').withColumn("level_id",lit(0))
df_org_unit = (df_cluster.union(df_region)).union(df_country)
#Adding a hierarchy_type column
df_org_unit=df_org_unit.withColumn("hierarchy_type",lit('org_unit'))
df_org_unit=df_org_unit.dropDuplicates(['hierarchy_value',"description",'level_id','parent_value'])
df_org_unit = df_org_unit.select('hierarchy_type','hierarchy_value','parent_value','level_id','description')
df_union = (df_product.union(df_channel)).union(df_org_unit)
# Replacing  ' ' with '_' 
df_union=df_union.withColumn('hierarchy_value', regexp_replace('hierarchy_value', ' ', '_')).withColumn('hierarchy_value', regexp_replace('hierarchy_value', '-', '_'))
df_union=df_union.withColumn('parent_value', regexp_replace('parent_value', ' ', '_')).withColumn('parent_value', regexp_replace('parent_value', '-', '_'))
# df_union=df_union.withColumn('description', regexp_replace('description', '-', '_'))
#Adding a domain
df_union=df_union.withColumn("domain_id",lit(domain_id))
# drop duplicates
df_union=df_union.dropDuplicates(['hierarchy_value','level_id','parent_value','hierarchy_type','domain_id'])
hierarchy_2 = df_union.select('domain_id','hierarchy_type','hierarchy_value','parent_value','level_id','description')

# COMMAND ----------

#Retruns the records which are there in incremental new data but not in full load exisiting data
hierarchy_incr = hierarchy_2.subtract(hierarchy_db)

# COMMAND ----------

l = set([data[0] for data in hierarchy_incr.select('hierarchy_value').collect()])

# COMMAND ----------

#Return the product_ids that doesn't exist in incremental load.
df_not_in_incr = hierarchy_db.filter(~hierarchy_db.hierarchy_value.isin(l))

# COMMAND ----------

final_data = df_not_in_incr.union(hierarchy_incr)

# COMMAND ----------

final_data.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/incremental_staging_data/{domain_id}_hierarchy_master.csv") 
