# Databricks notebook source
import pyspark
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import date_format
from datetime import datetime

# COMMAND ----------



# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

month_num = datetime.now().month
month_num = str(month_num)
datetime_object = datetime.strptime(month_num, "%m")
month_name = datetime_object.strftime("%B")
print(month_name)

# COMMAND ----------

run_id = spark.read.jdbc(url=url, table= f"(SELECT run_id FROM run WHERE create_ts <= DATEADD(M, -2, GETDATE())) as run_id", properties = properties)

# COMMAND ----------

run_ids = run_id.select('run_id').rdd.flatMap(lambda x: x).collect()
print(run_ids)

# COMMAND ----------

table_names = ('run','run_parameter','run_data','run_data_all_models','future_forecast_master','past_forecast_master','nts_master','agg_history_master','exception_output','event_log','brp_product_master','approve_forecast','reference_master','model_reference','model_selection_scores','proportion','normalization')
for i in table_names:
        for j in run_ids:
            df = spark.read.jdbc(url=url, table= f"(SELECT * FROM {i} WHERE run_id = {j}) as run_id", properties = properties)
            df.repartition(1).write.format("com.databricks.spark.csv").option("header","true").mode("OverWrite").save(f"/mnt/backup/Backup/{env}/{month_name}/{j}/{i}.csv") 

# COMMAND ----------

return_json = json.dumps(run_ids)    
dbutils.notebook.exit(return_json)
