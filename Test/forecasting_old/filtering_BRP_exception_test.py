# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

ingestion = 1

# COMMAND ----------

dbutils.widgets.text('run_id', '', 'run_id')
dbutils.widgets.text('domain_id', '', 'domain_id')
# dbutils.widgets.text('model_select', '', 'model_select')
# dbutils.widgets.text('run_mode', '', 'run_mode')

run_id = dbutils.widgets.get("run_id")
run_id = str(run_id)
domain_id = dbutils.widgets.get("domain_id")
# model_select = dbutils.widgets.get("model_select")
# run_mode = dbutils.widgets.get("run_mode")

print(run_id)
print(domain_id)
# print(model_select)
# print(run_mode)

# COMMAND ----------

def run_parameter_data(run_id,url,properties):
    
    parameter = spark.read.jdbc(url=url,table=f"(select parameter_id,run_value from run_parameter where run_id = '{run_id}') as run_value",properties=properties)
    parameter = parameter.toPandas()
   
    parameter = dict(zip(parameter['parameter_id'],parameter['run_value']))
    
    return parameter

# COMMAND ----------

parameters = run_parameter_data(run_id,url,properties)
model_select = parameters[22]
run_mode = parameters[25]
print(model_select)
print(run_mode)

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)
    
    return

# COMMAND ----------

# DBTITLE 1,Reading the data
filename = f'run_data_all_model_{run_id}.parquet'
folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
run_all_df = spark.read.parquet(folderPath, inferSchema = True, header= True)

if model_select == 'dynamic':
    filename = f'model_selection_scores_{run_id}.parquet'
    folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
    model_scoring_df = spark.read.parquet(folderPath,inferSchema = True, header= True)

else:
    pass

# COMMAND ----------

temp_period = run_all_df.sort(run_all_df.key,run_all_df.period).filter(run_all_df.forecast.isNotNull()).first()['period']

# COMMAND ----------

run_all_df.count()

# COMMAND ----------

if run_mode == 'MR_forecast':
    if ingestion == 1:
        run_all_df1 = run_all_df.drop('key','post_forecast_adjustment','model','level_shift')
        db_ingestion(run_all_df1, 'run_data', 'append')
        run_all_df1 = run_all_df1.na.drop(subset=["forecast"])
        db_ingestion(run_all_df1, 'run_data_all_models', 'append')
    else:
        print('Ingestion not done')
        run_data_all_model_pd = run_data_all_model1.toPandas()
    
elif model_select == 'dynamic':
    if ingestion == 1:
        db_ingestion(model_scoring_df, 'model_selection_scores', 'append')
        run_all_df1 = run_all_df.drop('key','post_forecast_adjustment','model','level_shift')
        run_all_df1 = run_all_df1.na.drop(subset=["forecast"])
        db_ingestion(run_all_df1, 'run_data_all_models', 'append')        
    else:
        print('Ingestion not done')
        model_scoring_df_pd = model_scoring_df.toPandas() 

# COMMAND ----------

# DBTITLE 1,Filtering best model from the run_all_models
if model_select == 'dynamic':
    model_scoring_df = model_scoring_df.withColumn('model_id', regexp_replace(model_scoring_df.model_id, '_n', ''))
    model_scoring_df = model_scoring_df.withColumn('keys', concat_ws('@', model_scoring_df.key, model_scoring_df.model_id))
    model_scoring_grouped = model_scoring_df.groupBy('key').agg({'score': 'min', 'keys':'first'}).withColumnRenamed('first(keys)', 'keys')
    key_list = model_scoring_grouped.toPandas()['keys'].values.tolist()
    run_all_df = run_all_df.withColumn('model_id', regexp_replace(run_all_df.model_id, '_n', ''))
    run_all_df = run_all_df.withColumn('key', concat_ws('@', run_all_df.org_unit_id, run_all_df.channel_id,run_all_df.product_id,run_all_df.model_id))
    filtered_df = run_all_df.filter(run_all_df.key.isin(key_list))
    filtered_df = filtered_df.drop('key','model','post_forecast_adjustment')
#     filtered_df = filtered_df.withColumn('key', concat_ws('@', filtered_df.org_unit_id, filtered_df.channel_id,filtered_df.product_id))
else:
    pass

# COMMAND ----------

# DBTITLE 1,writing in run_data
if run_mode == 'MR_forecast':
    filename = f'run_data_{run_id}.parquet'
    folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
    if ingestion == 1:
        run_all_df1.repartition(128).write.mode('overwrite').parquet(folderPath)
    else:
        print('Ingestion not done')
        run_all_df1_pd = run_all_df1.toPandas()
    

elif model_select == 'dynamic':
    filename = f'run_data_{run_id}.parquet'
    folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
    if ingestion == 1:
        filtered_df.repartition(128).write.mode('overwrite').parquet(folderPath)
        db_ingestion(filtered_df, 'run_data', 'append')
    else:
        print('Ingestion not done')
        filtered_df_pd = filtered_df.toPandas()
elif model_select == 'static' and run_mode != 'MR_forecast':
#     run_all_df = run_all_df.drop('key','post_forecast_adjustment','model')
    run_all_df = run_all_df.drop('key','post_forecast_adjustment','model','level_shift')
#     run_all_df = run_all_df.na.drop(subset=["forecast"])
    filename = f'run_data_{run_id}.parquet'
    folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
    if ingestion == 1:
        run_all_df.repartition(128).write.mode('overwrite').parquet(folderPath)
        db_ingestion(run_all_df, 'run_data', 'append')

    else:
        print('Ingestion not done')
        run_all_df_pd = run_all_df.toPandas()    
else:
    pass

# COMMAND ----------

# DBTITLE 1,BRP Product master
# MAGIC %run /Shared/Test/forecasting/brp_product_master_generator_test

# COMMAND ----------

parameter_values = run_parameter_data(run_id,url,properties)
brp_product_master = brp_product_master(run_id, parameter_values,url,properties)

# COMMAND ----------

brp_product_master_schema = StructType([StructField('run_id',StringType(),True),
                            StructField('domain_id',StringType(),True),
                            StructField('key',StringType(),True),
                            StructField('planner',StringType(),True),
                            StructField('Affiliate',StringType(),True),
                            StructField('p7',StringType(),True),
                            StructField('p6',StringType(),True),
                            StructField('p5',StringType(),True),
                            StructField('p4',StringType(),True),
                            StructField('p3',StringType(),True),
                            StructField('p2',StringType(),True),
                            StructField('product_id',StringType(),True),
                            StructField('channel',StringType(),True)                            
                            ])

# COMMAND ----------

brp_product_master = spark.createDataFrame(data = brp_product_master,schema = brp_product_master_schema)

# COMMAND ----------

# brp_product_master.display()

# COMMAND ----------

# DBTITLE 1,writing in brp_product_master
filename = f'brp_product_master_{run_id}.parquet'
folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
if ingestion == 1:
    brp_product_master.repartition(128).write.mode('overwrite').parquet(folderPath)
    db_ingestion(brp_product_master, 'brp_product_master', 'append')
    
else:
    print('Ingestion not done')

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting/exception_test
# MAGIC

# COMMAND ----------

parameter_values = run_parameter_data(run_id,url,properties)
forecast_length = int(parameter_values[24])
run_id = int(run_id)
final_df = get_decisionTree_df(run_id,forecast_length,domain_id)

# COMMAND ----------

final_df = spark.createDataFrame(final_df)

# COMMAND ----------

# DBTITLE 1,writing as a parquet
filename = f'exception_output_{run_id}.parquet'
folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
if ingestion == 1:
    final_df.repartition(128).write.mode('overwrite').parquet(folderPath)
    db_ingestion(final_df, 'exception_output', 'append')
else:
    print('Ingestion not done')

# COMMAND ----------

# filename = f'brp_product_master_{run_id}.parquet'
# folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
# df = spark.read.parquet(folderPath,inferSchema = True, header = True)
# df.display()


# COMMAND ----------

import json
temp_period = json.dumps(temp_period)
dbutils.notebook.exit(temp_period)
