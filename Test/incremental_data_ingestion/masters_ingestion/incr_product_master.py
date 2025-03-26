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

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

product_db =  spark.read.jdbc(url=url, table="product", properties=properties)
product_db = product_db.filter(product_db.domain_id==domain_id)

# COMMAND ----------

df = spark.read.csv(f"/mnt/customerdata/{env}/Incremental_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)
# Trimming the column names
col_ls = []
for coln in df.columns:
    tmp = "".join(coln.rstrip().lstrip())
    col_ls.append(tmp)
df = df.toDF(*col_ls)
df_ph2 = df.withColumnRenamed("Planning Hierarchy 2","product_id").withColumnRenamed("Planning Hierarchy 2 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(0))
df_ph3 = df.withColumnRenamed("Planning Hierarchy 3","product_id").withColumnRenamed("Planning Hierarchy 3 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(1))
df_ph7 = df.withColumnRenamed("Planning Hierarchy 7","product_id").withColumnRenamed("Planning Hierarchy 7 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(5))
df_ph6 = df.withColumnRenamed("Planning Hierarchy 6","product_id").withColumnRenamed("Planning Hierarchy 6 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(4))
df_ph5 = df.withColumnRenamed("Planning Hierarchy 5","product_id").withColumnRenamed("Planning Hierarchy 5 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(3))
df_ph4 = df.withColumnRenamed("Planning Hierarchy 4","product_id").withColumnRenamed("Planning Hierarchy 4 Description","product_desc").select('product_id', 'product_desc').withColumn("level_id",lit(2))
df_union = (df_ph7.union(df_ph6).union(df_ph5).union(df_ph4).union(df_ph3)).union(df_ph2)
#Adding a columns
df_union=df_union.withColumn("seasonality",lit("non-seasonal")).withColumn("status",lit("ACTIVE")).withColumn("touched",lit("TOUCHLESS"))
# Replacing  ' ' with '_' 
df_union=df_union.withColumn('product_id', regexp_replace('product_id', ' ', '_'))#.withColumn('product_id', regexp_replace('product_id', '-', '_'))
df_union=df_union.dropDuplicates(['product_id','product_desc','level_id','seasonality','status','touched'])
#Adding a domain
df_union=df_union.withColumn("domain_id",lit(domain_id))
product_2 = df_union.select('domain_id','product_id','level_id','product_desc','touched','seasonality','status')

# COMMAND ----------

#Retruns the records which are there in incremental new data but not in full load exisiting data
product_incr = product_2.subtract(product_db)

# COMMAND ----------

l = set([data[0] for data in product_incr.select('product_id').collect()])

# COMMAND ----------

#Return the product_ids that doesn't exist in incremental load.
df_not_in_incr = product_db.filter(~product_db.product_id.isin(l))

# COMMAND ----------

final_data = df_not_in_incr.union(product_incr)
# final_data.display()

# COMMAND ----------

final_data.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/incremental_staging_data/{domain_id}_product_master.csv") 
