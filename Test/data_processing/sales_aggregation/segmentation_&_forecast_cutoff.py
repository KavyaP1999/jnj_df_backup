# Databricks notebook source
# DBTITLE 1,Libraries
import numpy as np
import pandas as pd
import os,sys
import pyspark
from scipy import optimize
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType, DoubleType,DecimalType
import traceback

# COMMAND ----------

# DBTITLE 1,Running the configuration file
# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

#%run /Shared/Test/Databasewrapper_py_test

# COMMAND ----------

# DBTITLE 1,JDBC connection
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

# def update_flag(db_obj:object,db_conn:object,run_id,flag):
    
#     db_obj.execute_query(db_conn, "update run set run_state = '" + flag + "' where run_id = " + str(run_id), 'update')
    
#     return

# COMMAND ----------

def run_parameter_data(run_id,url,properties):
    
    parameter = spark.read.jdbc(url=url,table=f"(select parameter_id,run_value from run_parameter where run_id = '{run_id}') as run_value",properties=properties)
    parameter = parameter.toPandas()
   
    parameter = dict(zip(parameter['parameter_id'],parameter['run_value']))
    
    return parameter

# COMMAND ----------

dbutils.widgets.text('run_id', '', 'run_id')
run_id = dbutils.widgets.get("run_id")
run_id = int(run_id)

dbutils.widgets.text('domain_id', '', 'domain_id')
domain_id= dbutils.widgets.get("domain_id")
dbutils.widgets.text('filtermode', '', 'filtermode')
filtermode= dbutils.widgets.get("filtermode")

dbutils.widgets.text('run_mode', '', 'run_mode')
run_mode= dbutils.widgets.get("run_mode")
dbutils.widgets.text('filtermode', '', 'filtermode')
filtermode= dbutils.widgets.get("filtermode")
dbutils.widgets.text('forecast_length', '', 'forecast_length')
forecast_length= dbutils.widgets.get("forecast_length")
forecast_length = int(forecast_length)
dbutils.widgets.text('input_period', '', 'input_period')
input_period= dbutils.widgets.get("input_period")

# COMMAND ----------

# DBTITLE 1,Reading aggregation file from the blob storage
# filename = f'DP3_run_data_{run_id}.csv'
# folderPath=f"/mnt/disk/staging_data/test/{domain_id}/" + filename
# agg_dis_agg = spark.read.csv(folderPath,inferSchema=True,header=True)

working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP3_Int_data.parquet"
agg_dis_agg = spark.read.parquet(working_path,inferSchema=True,header=True)

# COMMAND ----------

# DBTITLE 1,Prep for calculate segmentation
agg_dis_agg_1 = agg_dis_agg.withColumn("key",concat_ws("~","org_unit_id","channel_id","product_id"))
agg_dis_agg_1 = agg_dis_agg_1.withColumn("org_unit_id", agg_dis_agg_1["org_unit_id"].cast('string'))
agg_dis_agg_1 = agg_dis_agg_1.withColumn("product_id", agg_dis_agg_1["product_id"].cast('string'))
agg_dis_agg_1 = agg_dis_agg_1.withColumn("channel_id", agg_dis_agg_1["channel_id"].cast('string'))

# COMMAND ----------

# agg_dis_agg_1.select(countDistinct('key')).show()

# COMMAND ----------

# DBTITLE 1,Defining schema
#for the segmentation and forecastability output
outSchema = StructType([StructField('domain_id',StringType(),True),
#                         StructField('run_state',StringType(),True),
                        StructField('org_unit_id',StringType(),True),
                        StructField('channel_id',StringType(),True),
                        StructField('product_id',StringType(),True),
                        StructField('period',StringType(),True),
                        StructField('historical_sale',IntegerType(),True),
                        StructField('forecast',IntegerType(),True),
                        StructField('segmentation',StringType(),True),
#                         StructField('model_id',StringType(),True),
#                         StructField('promotion',StringType(),True),
                        StructField('key',StringType(),True),
                       ])
# For the final df right before adding to the DB
finalschema = StructType([StructField('domain_id',StringType(),True),
                        StructField('run_id',IntegerType(),True),
                        StructField('run_state',StringType(),True),
                        StructField('org_unit_id',StringType(),True),
                        StructField('channel_id',StringType(),True),
                        StructField('product_id',StringType(),True),
                        StructField('period',StringType(),True),
                        StructField('historical_sale',DecimalType(20,10),True),
                        StructField('forecast',DecimalType(20,10),True),
                        StructField('segmentation',StringType(),True),
                        StructField('model_id',StringType(),True),
                        StructField('promotion',StringType(),True)
                       ])

# COMMAND ----------

# DBTITLE 1,Calculate Segmentation function()
@F.pandas_udf(outSchema, F.PandasUDFType.GROUPED_MAP)
def calculate_segmentation(df):
    df = df.copy()
    df = df.sort_values(by = 'period')
    period_val = str(df['period'].values[0])
    
    if run_mode == 'poc':
        df, df_test = df[0:-forecast_length], df[-forecast_length:df.shape[0]]
        
    new_prod_cut_off = 0
    if str(period_val).find('D') != -1:
        new_prod_cut_off = 366
    if str(period_val).find('W') != -1:
        new_prod_cut_off = 53
    if str(period_val).find('M') != -1:
        new_prod_cut_off = 12
    historical_sale = np.array(df['historical_sale'])
    segmentation = ''
    ratios = 0
    if historical_sale.shape[0] > new_prod_cut_off:
        total_counts = historical_sale.shape[0]
        non_zero_counts = historical_sale[np.nonzero(historical_sale)].shape[0]
        first_counts = 0
        for i in historical_sale:
            if i == 0:
                first_counts = first_counts + 1
            else:
                break

        if (total_counts - first_counts) > new_prod_cut_off:
            segmentation = 'OldProduct'
            ratios = non_zero_counts / (total_counts - first_counts)
        else:
            historical_sale = historical_sale[-new_prod_cut_off:]
            total_counts = historical_sale.shape[0]
            non_zero_counts = historical_sale[np.nonzero(historical_sale)].shape[0]
            first_counts = 0
            segmentation = 'NewProduct'
            ratios = (non_zero_counts - first_counts) / (total_counts - first_counts)
    else:
        total_counts = historical_sale.shape[0]
        non_zero_counts = historical_sale[np.nonzero(historical_sale)].shape[0]
        first_counts = 0
        segmentation = 'NewProduct'
        ratios = (non_zero_counts - first_counts) / (total_counts - first_counts)

    if segmentation == 'OldProduct':
        if ratios >= .75:
            segmentation = "Regular"
        if .75 > ratios >= .20:
            segmentation = "Intermittent"
        if ratios < .20:
            segmentation = "Sparse"
    elif segmentation == 'NewProduct':
        if ratios >= .75:
            segmentation = "NewProductRegular"
        if .75 > ratios >= .20:
            segmentation = "NewProductIntermittent"
        if ratios < .20:
            segmentation = "NewProductSparse"

    df['segmentation'] = segmentation
    
    if run_mode == 'poc':
        df = pd.concat([df, df_test], axis=0, ignore_index=True)
        df['segmentation'] = df.segmentation.fillna(method='ffill')
        
    return df


# COMMAND ----------

# DBTITLE 1,Forecastability Cutoff ()

@F.pandas_udf(outSchema, F.PandasUDFType.GROUPED_MAP)
def product_filter(historical_sales_data):
    historical_sales_data = historical_sales_data.sort_values(by = 'period')
    
    if input_period == 'W':
        forecastability_cutoff = 52
        new_product_cutoff = 26
    elif input_period == 'M':
        forecastability_cutoff = 12
        new_product_cutoff = 6
    elif input_period == 'D':
        forecastability_cutoff = 365
        new_product_cutoff = 182
         
    historical_sales_data1 = historical_sales_data.copy()
#     print(filtermode)
    if filtermode == 'all_products':
        
        historical_sales_data1.sort_values(['key', 'period'], inplace=True)
        
        keys = np.unique(historical_sales_data1['key'])

        filterted_hist = pd.DataFrame()
        sub_data = historical_sales_data1.copy().reset_index()
        sub_data.drop(columns=['index'], inplace=True)
        first_non_zero = sub_data['historical_sale'].ne(0).idxmax()
        if run_mode == 'poc':            
            if ((len(sub_data) - first_non_zero) >= (new_product_cutoff + forecast_length)):
                if (sub_data['historical_sale'][-(forecast_length + forecastability_cutoff):]).sum() > 0:
                    filterted_hist = pd.concat([filterted_hist, sub_data])

        if run_mode == 'forecast' or run_mode == 'MR_forecast':
            
            if ((len(sub_data) - first_non_zero) >= (new_product_cutoff)):
                if (sub_data['historical_sale'][-forecastability_cutoff:]).sum() > 0:
                    
                    filterted_hist = pd.concat([filterted_hist, sub_data])
      
    if filtermode != 'all_products':
        filterted_hist = filterted_hist.drop(['key'], axis=1)
          
    if filterted_hist.empty:
        
        filterted_hist = pd.concat([filterted_hist, sub_data])
        filterted_hist['key'] = 'key_deleted_forecastibility_cutoff'
    
    return filterted_hist


# COMMAND ----------

# reference_master = spark.read.jdbc(url=url, table= "reference_master", properties = properties)
# reference_master.createOrReplaceTempView('reference_master')
# reference_master = spark.sql("SELECT * from reference_master where run_id = '{0}' AND domain_id = '{1}'".format(run_id,domain_id))
# reference_master_split = reference_master.toPandas()

# COMMAND ----------

# df_filtered_pd = df_filtered.toPandas()
# df_filtered_pd['key'] = df_filtered_pd['org_unit_id'] + '@' + df_filtered_pd['channel_id'] + '@' + df_filtered_pd['product_id']
# df_filtered_pd = df_filtered_pd[df_filtered_pd['key'].isin(reference_master_split['key'])]

# COMMAND ----------

# df_filtered_pd = df_filtered_pd[df_filtered_pd['key'].isin(reference_master_split['key'])]
# df_filtered_pd.drop(columns=['key'], inplace=True)



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

# db_obj = DatabaseWrapper()
# db_conn = db_obj.connect()

try:
    pipeline_flag = 0
    df_seg = agg_dis_agg_1.groupby("key").apply(calculate_segmentation)
    df_filtered  = df_seg.groupby("key").apply(product_filter)
    
    df_filtered = df_filtered.filter((df_filtered.key != "key_deleted_forecastibility_cutoff"))
    df_filtered = df_filtered.withColumn('run_id',lit(run_id).cast(IntegerType()))
    df_filtered = df_filtered.withColumn('run_state',lit('DP3'))
    df_filtered = df_filtered.withColumn('model_id',lit(None).cast(StringType()))
    df_filtered = df_filtered.withColumn('promotion',lit(None).cast(StringType()))
    df_filtered = df_filtered.withColumn('forecast', lit(None).cast(StringType()))
    df_filtered = df_filtered.withColumn("historical_sale", df_filtered["historical_sale"].cast('decimal(20,10)'))
    df_filtered = df_filtered.withColumn("forecast", df_filtered["forecast"].cast('decimal(20,10)'))
    
    df_filtered = df_filtered.select('domain_id', 'run_id', 'run_state', 'org_unit_id', 'channel_id','product_id', 'period', 'historical_sale', 'forecast', 'segmentation','model_id','promotion')
    df_filtered = sqlContext.createDataFrame(df_filtered.collect(), finalschema)

    # INGESTING TO THE DB
    db_ingestion(df_filtered, 'run_data', 'append')
    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP3_run_data.parquet"
    df_filtered.repartition(1).write.mode('overwrite').parquet(working_path)
#     excel_path = f'/mnt/jnj_output_file/{env}/{domain_id}/Run_ID_{run_id}/DP3_{run_id}.xlsx'
#     save_output1(df_filtered,excel_path,'DP3')
#     parameters = run_parameter_data(run_id,url,properties)
#     status = parameters[7]
#     update_flag(db_obj,db_conn,run_id,status)
#     db_obj.close(db_conn)
    print("Aggregation Segmentation and forecastability cutoff successful")
    
except:
    traceback.print_exc()
    10/0
#     traceback.print_exc()
#     parameters = run_parameter_data(run_id,url,properties)
#     status = parameters[8]
#     update_flag(db_obj,db_conn,run_id,status)
#     db_obj.close(db_conn)
#     pipeline_flag = 1
#     print("Aggregation Segmentation and forecastability cutoff failed")

# COMMAND ----------

# if pipeline_flag == 0:
#     print("Success")
# else:
#     print("failure")
#     10/0

# COMMAND ----------

# df_seg.select(countDistinct('key')).show()

# COMMAND ----------

# df_filtered_tmp = df_filtered
# df_filtered_tmp = df_filtered_tmp.withColumn("key",concat_ws("~","org_unit_id","channel_id","product_id"))
# df_filtered_tmp.select(countDistinct('key')).show()
