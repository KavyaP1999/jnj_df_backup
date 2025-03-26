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
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType, DoubleType
from pyspark.sql.functions import expr

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

dbutils.widgets.text("domain_id", "","")
domain_id = dbutils.widgets.get("domain_id")
print ("domain_id:",domain_id)

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

df = spark.read.csv(f"/mnt/disk/staging_data/{env}/{domain_id}/incremental_staging_data/{domain_id}_org_unit_master.csv",inferSchema=True,header=True)

# COMMAND ----------

# DBTITLE 1,data-ingestion
def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

db_ingestion(df, 'org_unit', 'append')
