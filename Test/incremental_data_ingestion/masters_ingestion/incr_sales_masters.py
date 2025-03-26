# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import numpy as np
import pandas as pd
import traceback
import pandas as pd
import numpy as np
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.functions as func
from pyspark.sql.functions import regexp_replace
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

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

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
df_union=df_union.withColumn('hierarchy_value', regexp_replace('hierarchy_value', ' ', '_'))
df_union=df_union.withColumn('parent_value', regexp_replace('parent_value', ' ', '_'))
#Adding a domain
df_union=df_union.withColumn("domain_id",lit(domain_id))
# drop duplicates
df_union=df_union.dropDuplicates(['hierarchy_value','level_id','parent_value','hierarchy_type','domain_id'])
hierarchy_2 = df_union.select('domain_id','hierarchy_type','hierarchy_value','parent_value','level_id','description')

# COMMAND ----------

#Retruns the records which are there in incremental new data but not in full load exisiting data
hierarchy_incr = hierarchy_2.subtract(hierarchy_db)
hierarchy_incr = hierarchy_incr.withColumn('key',concat_ws('~',hierarchy_incr.hierarchy_value,hierarchy_incr.level_id))
hierarchy_val = hierarchy_incr.select(f.collect_list('key')).first()[0]

# COMMAND ----------

hierarchy_db_temp = hierarchy_db.withColumn('key',concat_ws('~',hierarchy_db.hierarchy_value,hierarchy_db.level_id))

# COMMAND ----------

# l = set([data[0] for data in hierarchy_incr.select('hierarchy_value').collect()])
#level_ids = set([data[0] for data in hierarchy_incr.select('level_id').collect()])
#hierarchy_val = hierarchy_incr.select(f.collect_list('key')).first()[0]

# COMMAND ----------

#Return the product_ids that doesn't exist in incremental load.
# df_not_in_incr = hierarchy_db.filter(~hierarchy_db.hierarchy_value.isin(l))
df_not_in_incr = hierarchy_db_temp.filter((~hierarchy_db_temp.key.isin(hierarchy_val)))

# COMMAND ----------

final_data_temp = df_not_in_incr.union(hierarchy_incr)
final_data = final_data_temp.drop('key')

# COMMAND ----------

#saving in blob stroage
final_data.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/incremental_staging_data/{domain_id}_hierarchy_master.csv") 

# COMMAND ----------

# MAGIC %md
# MAGIC ##product

# COMMAND ----------

product_data = final_data.filter(final_data.hierarchy_type == 'product').withColumnRenamed('hierarchy_value','product_id').withColumnRenamed('description','product_desc').drop('hierarchy_type','parent_value')

# COMMAND ----------

#Adding a columns
product_data=product_data.withColumn("seasonality",lit("non-seasonal")).withColumn("status",lit("ACTIVE")).withColumn("touched",lit("TOUCHLESS"))
# Replacing  ' ' with '_' 
product_data=product_data.withColumn('product_id', regexp_replace('product_id', ' ', '_'))#.withColumn('product_id', regexp_replace('product_id', '-', '_'))
product_data=product_data.dropDuplicates(['product_id','product_desc','level_id','seasonality','status','touched'])
#Adding a domain
product_data=product_data.withColumn("domain_id",lit(domain_id))
product_data = product_data.select('domain_id','product_id','level_id','product_desc','touched','seasonality','status')

# COMMAND ----------

#db ingestion
db_ingestion(product_data, 'product', 'append')

# COMMAND ----------

# MAGIC %md
# MAGIC #org_unit

# COMMAND ----------

org_unit_data = final_data.filter(final_data.hierarchy_type == 'org_unit').withColumnRenamed('hierarchy_value','org_unit_id').withColumnRenamed('description','org_unit_desc').drop('hierarchy_type','parent_value')

# COMMAND ----------

#db ingestion
db_ingestion(org_unit_data, 'org_unit', 'append')

# COMMAND ----------

# MAGIC %md
# MAGIC ##channel

# COMMAND ----------

channel_data = final_data.filter(final_data.hierarchy_type == 'channel').withColumnRenamed('hierarchy_value','channel_id').withColumnRenamed('description','channel_desc').drop('hierarchy_type','parent_value')

# COMMAND ----------

#db ingestion 
db_ingestion(channel_data, 'channel', 'append')
