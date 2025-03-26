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

def save_output1(df,excel_path,status):
    file_name = excel_path
    lim = 1000000
    rows = df.count()
    
    def get_col_name(col): # col is 1 based
        excelCol = str()
        div = col
        while div:
            (div, mod) = divmod(div-1, 26) # will return (x, 0 .. 25)
            excelCol = chr(mod + 65) + excelCol

        return excelCol
    
    last_cell = get_col_name(len(df.columns)) + str(rows + 1)


    if rows > lim:
        files = [x for x in range(0,rows,lim)]
        files.append(rows)
        for i in range(len(files)):
            if i < len(files)-1:
                df.withColumn('index',monotonically_increasing_id()).where(col("index").between(files[i],files[i+1]-1)).drop('index')\
                .write.format("com.crealytics.spark.excel")\
                .option("dataAddress",f"{status}_split_{str(i+1)}!A1:{last_cell}")\
                .option("header", "true")\
                .mode("append")\
                .save(file_name)
    else:
        df.write.format("com.crealytics.spark.excel")\
        .option("dataAddress",f"{status}!A1:{last_cell}")\
        .option("header", "true")\
        .mode("overwrite")\
        .save(file_name)

    return file_name

# COMMAND ----------

def jnj_output_blob_ingestion():
    
    spark.conf.set(
      "fs.azure.account.key.jnjmddevstgoutput.blob.core.windows.net",
      'KchlUa8Hr3r7lUYzoAmKfxa3SJIDuA0EExFlFnl+oONzOTSJnUiJ0SRkTBc4EFu5aoMxHb+QOPT8bHJZaQJReQ==')
    container_path = 'wasbs://jnj-md-project-output-files@jnjmddevstgoutput.blob.core.windows.net'
    output_path = container_path+f"/{env}/{domain_id}/Run_ID_{run_id}/"  
    
    return output_path

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
working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_run_data_all_model.parquet"
run_all_df = spark.read.parquet(working_path, inferSchema = True, header= True)


if model_select == 'dynamic':
    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_model_selection_scores.parquet"
    model_scoring_df = spark.read.parquet(working_path,inferSchema = True, header= True)

else:
    pass

# COMMAND ----------

temp_period = run_all_df.sort(run_all_df.key,run_all_df.period).filter(run_all_df.forecast.isNotNull()).first()['period']

# COMMAND ----------

if run_mode == 'MR_forecast':
    if ingestion == 1:
        run_all_df1 = run_all_df.drop('key','post_forecast_adjustment','model','level_shift')
        working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F3_run_data.parquet"
        run_all_df1.repartition(128).write.mode('overwrite').parquet(working_path)

        run_all_df1 = run_all_df1.na.drop(subset=["forecast"])
        db_ingestion(run_all_df1, 'run_data_all_models', 'append')
        working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F3_run_data_all_models.parquet"
        run_all_df1.repartition(128).write.mode('overwrite').parquet(working_path)

    else:
        print('Ingestion not done')
        run_data_all_model_pd = run_data_all_model1.toPandas()
    
elif model_select == 'dynamic':
    if ingestion == 1:
        run_all_df1 = run_all_df.drop('key','post_forecast_adjustment','model','level_shift')
        run_all_df1 = run_all_df1.na.drop(subset=["forecast"])
        db_ingestion(run_all_df1, 'run_data_all_models', 'append')
        working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F3_run_data_all_models.parquet"
        run_all_df1.repartition(128).write.mode('overwrite').parquet(working_path)
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
else:
    pass

# COMMAND ----------

# DBTITLE 1,writing in run_data
if run_mode == 'MR_forecast':

    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F3_run_data.parquet"
    if ingestion == 1:
        run_all_df1.repartition(128).write.mode('overwrite').parquet(working_path)
    else:
        print('Ingestion not done')
        run_all_df1_pd = run_all_df1.toPandas()
    

elif model_select == 'dynamic':

    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F3_run_data.parquet"

    if ingestion == 1:
        filtered_df.repartition(128).write.mode('overwrite').parquet(working_path)
        db_ingestion(filtered_df, 'run_data', 'append')
        excel_path = f'/mnt/jnj_output_file/{env}/{domain_id}/Run_ID_{run_id}/F3_{run_id}.xlsx'
        save_output1(filtered_df, excel_path, 'F3')
    else:
        print('Ingestion not done')
        filtered_df_pd = filtered_df.toPandas()
elif model_select == 'static' and run_mode != 'MR_forecast':
    run_all_df = run_all_df.drop('key','post_forecast_adjustment','model','level_shift')
    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F3_run_data.parquet"

    if ingestion == 1:
        run_all_df.repartition(128).write.mode('overwrite').parquet(working_path)
        db_ingestion(run_all_df, 'run_data', 'append')
    else:
        print('Ingestion not done')
        run_all_df_pd = run_all_df.toPandas()    
else:
    pass

# COMMAND ----------

# DBTITLE 1,BRP Product master
# MAGIC %run /Shared/Test/forecasting_rf/brp_product_master_generator_test

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

# DBTITLE 1,writing in brp_product_master
working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_brp_product_master.parquet"

if ingestion == 1:
    brp_product_master.repartition(128).write.mode('overwrite').parquet(working_path)
    db_ingestion(brp_product_master, 'brp_product_master', 'append')
    
else:
    print('Ingestion not done')

# COMMAND ----------

# MAGIC %run /Shared/Test/forecasting_rf/exception_test
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
working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_exception_output.parquet"
if ingestion == 1:
    final_df.repartition(128).write.mode('overwrite').parquet(working_path)
    db_ingestion(final_df, 'exception_output', 'append')
else:
    print('Ingestion not done')

# COMMAND ----------

import json
temp_period = json.dumps(temp_period)
dbutils.notebook.exit(temp_period)
