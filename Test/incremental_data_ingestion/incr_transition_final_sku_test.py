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

pd.set_option('display.max_columns', None)

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

def find_final_sku(graph, src):
    if src in graph:
        for neighbor in graph[src]:
            yield from find_final_sku(graph, neighbor)
    else:
        yield src

def old_new_sku_mapping(domain_id, transition_master):

    if len(transition_master) > 0:
        transition_master['old_sku'] = transition_master['domain_id'] + '@' + transition_master['org_unit_id'] \
                                    + '@' + transition_master['channel_id'] + '@' + transition_master['old_sku']
        transition_master['new_sku'] = transition_master['domain_id'] + '@' + transition_master['org_unit_id'] \
                                    + '@' + transition_master['channel_id'] + '@' + transition_master['new_sku']
        graph = defaultdict(list)
        new_skus = set()

        for old_sku, new_sku in zip(transition_master.old_sku, transition_master.new_sku):
            graph[old_sku].append(new_sku)
            new_skus.add(new_sku)
        transition_master['final_sku'] = [next(find_final_sku(graph, x)) for x in graph]
        print(transition_master)
    return transition_master

# COMMAND ----------

transition = spark.read.jdbc(url=url, table="transition_final_master", properties=properties)
transition = transition.filter(transition.domain_id==domain_id)

# COMMAND ----------

transition_df = transition.toPandas()

# COMMAND ----------

df = spark.read.csv(f"/mnt/customerdata/{env}/Incremental_Load/{domain_id}/transition_master.csv",inferSchema=True,header=True)

# COMMAND ----------

# Trimming the column names
col_ls = []
for coln in df.columns:
    tmp = "".join(coln.rstrip().lstrip())
    col_ls.append(tmp)
df = df.toDF(*col_ls)
# Replacing  ' ' with '_' 
df=df.withColumn('channel_id', regexp_replace('channel_id', ' ', '_'))
df=df.withColumn('org_unit_id', regexp_replace('org_unit_id', ' ', '_'))
#Trim for org_unit_id and channel_id
df = df.withColumn("org_unit_id", trim(df.org_unit_id)).withColumn("channel_id", trim(df.channel_id))
#Adding a domain
df=df.withColumn("domain_id",lit(domain_id))
df = df.select('domain_id','org_unit_id','channel_id','old_sku','new_sku')
df = df.withColumn("new_sku",col("new_sku").cast(StringType()))# Changed into string recently(5/10/22)
df = df.withColumn("old_sku",col("old_sku").cast(StringType()))#(5/10/22)

# COMMAND ----------

incr_transition = df.toPandas()

# COMMAND ----------

incr_transitioned_df = old_new_sku_mapping(domain_id, incr_transition)

# COMMAND ----------

if len(incr_transitioned_df) > 0 & len(transition_df) > 0 :
    result1 = pd.merge(transition_df, incr_transitioned_df, how = "left", left_on=['final_sku'],right_on=['old_sku'], indicator = True)
    result1['final_sku'] = result1['final_sku_y'].combine_first(result1['final_sku_x'])
    result1 = result1.rename(columns = {'domain_id_x':'domain_id','org_unit_id_x':'org_unit_id','channel_id_x':'channel_id','old_sku_x':'old_sku', 'new_sku_x':'new_sku'})
    result1 =result1[['domain_id','org_unit_id','channel_id','old_sku', 'new_sku', 'final_sku']]
    result2 = pd.merge(incr_transitioned_df, result1, how="left", left_on=['final_sku'],right_on=['old_sku'])
    result2['final_sku'] = result2['final_sku_y'].combine_first(result2['final_sku_x'])
    result2 = result2.rename(columns = {'domain_id_x':'domain_id','org_unit_id_x':'org_unit_id','channel_id_x':'channel_id','old_sku_x':'old_sku', 'new_sku_x':'new_sku'})
    result2 =result2[['domain_id','org_unit_id','channel_id','old_sku', 'new_sku', 'final_sku']] 
    final_transition_master = pd.concat([result1, result2], ignore_index = True)
elif len(incr_transitioned_df) > 0 & len(transition_df) == 0:
    final_transition_master = incr_transitioned_df
elif len(incr_transitioned_df) == 0 & len(transition_df) > 0: 
    final_transition_master = transition_df
else:
    final_transition_master = incr_transitioned_df
    

# COMMAND ----------

if len(final_transition_master) > 0:
    transitioned_master = spark.createDataFrame(final_transition_master)
else :
    columns = StructType([StructField('domain_id',
                                  StringType(), True),
                    StructField('org_unit_id',
                                StringType(), True),
                    StructField('channel_id',
                                StringType(), True),
                    StructField('old_sku',
                                StringType(), True),
                    StructField('new_sku',
                                StringType(), True),
                    StructField('final_sku',
                                StringType(), True)])
    transitioned_master = spark.createDataFrame(data = final_transition_master,
                           schema = columns)

# COMMAND ----------

transitioned_master.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/{env}/{domain_id}/incremental_staging_data/{domain_id}_transition_master.csv") 
