# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import time
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from collections import defaultdict

# COMMAND ----------

dbutils.widgets.text("domain_id", "","")
domain = dbutils.widgets.get("domain_id")
print ("domain_id:",domain)

# COMMAND ----------

pd.set_option('display.max_columns', None)

# COMMAND ----------

# MAGIC %run /Shared/Prod/configuration_prodV2

# COMMAND ----------

# transition_master = spark.read.csv("/mnt/demo/EMEA_TECA/transition.csv", inferSchema=True, header=True)
# transition_df = transition_master.toPandas()
# convert_dict = {'old_sku':str, 'new_sku':str}
# transition_df = transition_df.astype(convert_dict)

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

transition = spark.read.jdbc(url=url, table= "transition", properties = properties)
transition.createOrReplaceTempView('transition')
transition = spark.sql("SELECT * from transition where domain_id = '{}'".format(domain))

# COMMAND ----------

transition_df = transition.toPandas()

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

transitioned_df = old_new_sku_mapping(domain, transition_df)

# COMMAND ----------

if len(transitioned_df) > 0:
    transitioned_master = spark.createDataFrame(transitioned_df)
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
    transitioned_master = spark.createDataFrame(data = transitioned_df,
                           schema = columns)

# COMMAND ----------

transitioned_master.write.jdbc(url=url, table= "transition_final_master", mode="append", properties = properties)

# COMMAND ----------

# transition_final_master = spark.read.jdbc(url=url, table= "transition_final_master", properties = properties)

# COMMAND ----------

#transition_final_master.display()
