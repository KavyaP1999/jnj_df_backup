# Databricks notebook source
import json

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

run_id = spark.read.jdbc(url=url, table= f"(SELECT run_id FROM run WHERE create_ts <= DATEADD(M, -2, GETDATE())) as run_id", properties = properties)
run_ids = run_id.select('run_id').rdd.flatMap(lambda x: x).collect()
print(run_ids)

# COMMAND ----------

return_json = json.dumps(run_ids)    
dbutils.notebook.exit(return_json)

# COMMAND ----------


