# Databricks notebook source
# MAGIC %md
# MAGIC ###To get domain_id

# COMMAND ----------

dbutils.widgets.text("domain_id", "","")
Domain = dbutils.widgets.get("domain_id")
print ("domain_id:",Domain)

# COMMAND ----------

# MAGIC %run /Shared/Prod/configuration_prodV2

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

raw_history_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM raw_run_data WHERE domain_id = '{Domain}' and  run_state = 'I3') as raw_history_master", properties = properties)

# COMMAND ----------

raw_history_master = raw_history_master.withColumn('expost_sales', raw_history_master.historical_sale)

# COMMAND ----------

raw_history_master = raw_history_master.withColumnRenamed("historical_sale", "cdh")

# COMMAND ----------

raw_history_master.display()

# COMMAND ----------

db_ingestion(raw_history_master, 'raw_history_master', 'append')
