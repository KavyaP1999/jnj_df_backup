# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, lower, col

# COMMAND ----------

dbutils.widgets.text("domain_id", "","")
domain = dbutils.widgets.get("domain_id")
print ("domain_id:",domain)

# COMMAND ----------

spark = SparkSession.builder.appName("demo").getOrCreate()

# COMMAND ----------

# MAGIC %run /Shared/Prod/configuration_prodV2

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

transitioned_raw_history_master = spark.read.csv(f"/mnt/disk/staging_data/prod/{domain}/transitioned_raw_history_master_{domain}.csv",inferSchema=True,header=True)

# COMMAND ----------

#Old code reverted for NPF
npf_master = spark.read.jdbc(url=url, table="npf_master", properties=properties)
npf_master = npf_master.filter(npf_master.domain_id==domain)

# COMMAND ----------

# #New code readed from transitioned_npf_master

# #npf_master = spark.read.jdbc(url=url, table="npf_master", properties=properties)
# npf_master = spark.read.csv(f"/mnt/disk/staging_data/{env}/{domain}/transitioned_raw_npf_master_{domain}.csv",inferSchema=True,header=True)

# npf_master = npf_master.filter(npf_master.domain_id==domain)

# COMMAND ----------

npf_master = npf_master.select(['org_unit_id', 'channel_id', 'product_id', lower('status').alias('status')])
npf_master = npf_master.withColumn('keys', concat_ws('~', npf_master.org_unit_id, npf_master.channel_id, npf_master.product_id))
npf_master = npf_master.drop('org_unit_id', 'channel_id', 'product_id')
transitioned_raw_history_master = transitioned_raw_history_master.withColumn('keys', concat_ws('~', transitioned_raw_history_master.org_unit_id, transitioned_raw_history_master.channel_id, transitioned_raw_history_master.product_id))
final_data = transitioned_raw_history_master.join(npf_master, transitioned_raw_history_master.keys==npf_master.keys, "inner")
final_data = final_data.filter((final_data.status=='active') | (final_data.status=='npi'))
final_data = final_data.drop('keys', 'status')

# COMMAND ----------

final_data.count()

# COMMAND ----------

final_data.coalesce(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/disk/staging_data/prod/{domain}/filtered_active_npi_raw_history_master_{domain}.csv") 
