# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, lower, col, split
from pyspark.sql import functions as f
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import sys
import os
import pandas as pd
import numpy as np
import random
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import logging
import pandas as pd
import pandas as pd
from pandas import DataFrame
from numpy.random import RandomState
from sklearn.base import BaseEstimator, TransformerMixin
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType, DoubleType
from pyspark.sql.functions import when, col, size, split
from pyspark.sql.functions import pandas_udf, PandasUDFType 
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import first
from pyspark.sql.functions import col, pandas_udf
import math as m
import warnings
import traceback
warnings.filterwarnings('ignore')
from pyspark.sql.types import LongType
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

ingestion_flag = 1

# COMMAND ----------

dbutils.widgets.text("run_id", "","")
run_id = dbutils.widgets.get("run_id")
print ("run_id:",run_id)

dbutils.widgets.text("domain_id", "","")
domain_id = dbutils.widgets.get("domain_id")
print ("domain_id:",domain_id)

# COMMAND ----------

dbutils.widgets.text("org_unit_input_level", "","")
org_unit_input_level = dbutils.widgets.get("org_unit_input_level")
org_unit_input_level = int(org_unit_input_level)

dbutils.widgets.text("channel_input_level", "","")
channel_input_level = dbutils.widgets.get("channel_input_level")
channel_input_level = int(channel_input_level)

dbutils.widgets.text("product_input_level", "","")
product_input_level = dbutils.widgets.get("product_input_level")
product_input_level = int(product_input_level)

print("org_unit_input_level:",org_unit_input_level)
print("channel_input_level:",channel_input_level)
print("product_input_level:",product_input_level)

# COMMAND ----------

dbutils.widgets.text("org_unit_forecast_level", "","")
org_unit_forecast_level = dbutils.widgets.get("org_unit_forecast_level")
org_unit_forecast_level = int(org_unit_forecast_level)

dbutils.widgets.text("channel_forecast_level", "","")
channel_forecast_level = dbutils.widgets.get("channel_forecast_level")
channel_forecast_level = int(channel_forecast_level)

dbutils.widgets.text("product_forecast_level", "","")
product_forecast_level = dbutils.widgets.get("product_forecast_level")
product_forecast_level = int(product_forecast_level)

print("org_unit_forecast_level:",org_unit_forecast_level)
print("channel_forecast_level:",channel_forecast_level)
print("product_forecast_level:",product_forecast_level)

# COMMAND ----------

dbutils.widgets.text("org_unit_output_level", "","")
org_unit_output_level = dbutils.widgets.get("org_unit_output_level")
org_unit_output_level = int(org_unit_output_level)

dbutils.widgets.text("channel_output_level", "","")
channel_output_level = dbutils.widgets.get("channel_output_level")
channel_output_level = int(channel_output_level)

dbutils.widgets.text("product_output_level", "","")
product_output_level = dbutils.widgets.get("product_output_level")
product_output_level = int(product_output_level)

print("org_unit_forecast_level:",org_unit_output_level)
print("channel_forecast_level:",channel_output_level)
print("product_forecast_level:",product_output_level)

# COMMAND ----------

dbutils.widgets.text("period_input_level", "","")
period_input_level = dbutils.widgets.get("period_input_level")


dbutils.widgets.text("period_forecast_level", "","")
period_forecast_level = dbutils.widgets.get("period_forecast_level")


dbutils.widgets.text("period_output_level", "","")
period_output_level = dbutils.widgets.get("period_output_level")

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

rounded_schema = StructType([StructField('domain_id',StringType(),True),
                        StructField('org_unit_id',StringType(),True),
                        StructField('channel_id',StringType(),True),
                        StructField('product_id',StringType(),True),
                        StructField('period',StringType(),True),
                        StructField('historical_sale',IntegerType(),True),
                        StructField('forecast',IntegerType(),True),
                        StructField('model_id',StringType(),True),
                        StructField('promotion',StringType(),True),
                        StructField('segmentation',StringType(),True)
                       ])

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

future_discontinue_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM future_discontinue_master WHERE domain_id = '{domain_id}') AS rd", properties = properties)
df_hierarchy = spark.read.jdbc(url=url, table= f"(SELECT * FROM hierarchy WHERE domain_id = '{domain_id}') AS hm", properties = properties)
df_hierarchy_new=df_hierarchy.toPandas()

# COMMAND ----------

future_discontinue_master.display()

# COMMAND ----------

@F.pandas_udf(rounded_schema, F.PandasUDFType.GROUPED_MAP)
def rounding_logic(df1):
    df_final = pd.DataFrame()

    hist_df1 = df1[df1['forecast'].isna()].copy()
    df1 = df1[df1['forecast'] >= 0].copy()
    
    df1.reset_index(drop=True, inplace=True)

    df1["cum_values"] = df1["forecast"].cumsum()
    rounded = []
    
    for i in range(df1.index.start, df1.index.stop):
        vl = float(df1["cum_values"][i]) - np.sum(rounded)
        rn = np.round(vl)
        rounded.append(rn)

    df1.loc[:, "forecast"] = rounded

    df_final = pd.concat([df_final, hist_df1, df1])
        
    df_final.drop(columns=["psuedo_key", "cum_values"], inplace=True)
    df_final = df_final[['domain_id','org_unit_id','channel_id','product_id','period','historical_sale',
                         'forecast','model_id','promotion','segmentation']]
    df_final.drop_duplicates(inplace=True)

    return df_final

# COMMAND ----------



# COMMAND ----------

def create_keys(df):
    df = df.withColumn('org_unit_id',df.org_unit_id.cast(StringType())).withColumn('channel_id',df.channel_id.cast(StringType())).withColumn('product_id',df.product_id.cast(StringType()))
    df = df.withColumn('keys',concat_ws('~',df.org_unit_id,df.channel_id,df.product_id))
    keys = df.select('keys').distinct()
    return df

# COMMAND ----------

def generate_hierarchy(data, suffix='product'):
    data = data[data['hierarchy_type'] == suffix]
    data = data[['hierarchy_value', 'parent_value', 'level_id']].copy()
    split_df = {}
    for i in np.unique(data.level_id):
        split_df[i] = data[data['level_id'] == i].copy()
    df = pd.DataFrame()
    df = split_df[list(split_df.keys())[-1]][['hierarchy_value']].copy()
    df.rename(columns={'hierarchy_value': f'{suffix}_{list(split_df.keys())[-1]}'}, inplace=True)
    for i in list(split_df.keys())[-2::-1]:
        data = split_df[i][['hierarchy_value', 'parent_value']].copy()
        data.rename(columns={'hierarchy_value': f'{suffix}_{i}'}, inplace=True)
        df = pd.merge(df, data, left_on=f'{suffix}_{int(i)+1}', right_on='parent_value', how='inner')
        df.drop(columns={'parent_value'}, inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# COMMAND ----------

def product_aggregator(df1, Heirarchy, product_input_level, product_output_level):

    hm = Heirarchy.select(col(f'product_{product_input_level}'),col(f'product_{product_output_level}'))
    df = df1.join(hm, df1.product_id == hm[f'product_{int(product_input_level)}'], how='left').drop('product_id',f'product_{int(product_input_level)}').withColumnRenamed(f'product_{int(product_output_level)}','product_id')
    df = df.withColumn('historical_sale',df.historical_sale.cast(IntegerType())).withColumn('forecast',df.forecast.cast(IntegerType()))
    df = df.groupBy("domain_id",'org_unit_id','channel_id','product_id','period').sum('historical_sale', 'forecast').withColumnRenamed('sum(historical_sale)','historical_sale').withColumnRenamed('sum(forecast)','forecast')
    return df

def channel_aggregator(df1, Heirarchy, channel_input_level, channel_output_level):

    hm = Heirarchy.select(col(f'channel_{channel_input_level}'),col(f'channel_{channel_output_level}'))
    df = df1.join(hm, df1.channel_id == hm[f'channel_{int(channel_input_level)}'], how='left').drop('channel_id',f'channel_{int(channel_input_level)}').withColumnRenamed(f'channel_{int(channel_output_level)}','channel_id')
    df = df.withColumn('historical_sale',df.historical_sale.cast(IntegerType())).withColumn('forecast',df.forecast.cast(IntegerType()))
    df = df.groupBy("domain_id",'org_unit_id','product_id','channel_id','period').sum('historical_sale', 'forecast').withColumnRenamed('sum(historical_sale)','historical_sale').withColumnRenamed('sum(forecast)','forecast')
    return df

def org_unit_aggregator(df1, Heirarchy, org_unit_input_level, org_unit_output_level):

    hm = Heirarchy.select(col(f'org_unit_{org_unit_input_level}'),col(f'org_unit_{org_unit_output_level}'))
    df = df1.join(hm, df1.org_unit_id == hm[f'org_unit_{int(org_unit_input_level)}'], how='left').drop('org_unit_id',f'org_unit_{int(org_unit_input_level)}').withColumnRenamed(f'org_unit_{int(org_unit_output_level)}','org_unit_id')
    df = df.withColumn('historical_sale',df.historical_sale.cast(IntegerType())).withColumn('forecast',df.forecast.cast(IntegerType()))
    df = df.groupBy("domain_id",'org_unit_id','product_id','channel_id','period').sum('historical_sale', 'forecast').withColumnRenamed('sum(historical_sale)','historical_sale').withColumnRenamed('sum(forecast)','forecast')
    return df

# COMMAND ----------

def product_dis_aggregator(df1, Heirarchy, product_input_level, product_output_level):

    hm = Heirarchy.select(col(f'product_{product_input_level}'),col(f'product_{product_output_level}'))
    df = df1.join(hm, df1.product_id == hm[f'product_{int(product_input_level)}'], how='left')
    df = df.drop('product_id',f'product_{int(product_input_level)}').withColumnRenamed(f'product_{int(product_output_level)}','product_id')
    return df

def channel_dis_aggregator(df1, Heirarchy, channel_input_level, channel_output_level):

    hm = Heirarchy.select(col(f'channel_{channel_input_level}'),col(f'channel_{channel_output_level}'))
    df = df1.join(hm, df1.channel_id == hm[f'channel_{int(channel_input_level)}'], how='left').drop('channel_id',f'channel_{int(channel_input_level)}').withColumnRenamed(f'channel_{int(channel_output_level)}','channel_id')
    return df

def org_unit_dis_aggregator(df1, Heirarchy, org_unit_input_level, org_unit_output_level):

    hm = Heirarchy.select(col(f'org_unit_{org_unit_input_level}'),col(f'org_unit_{org_unit_output_level}'))
    df = df1.join(hm, df1.org_unit_id == hm[f'org_unit_{int(org_unit_input_level)}'], how='left').drop('org_unit_id',f'org_unit_{int(org_unit_input_level)}').withColumnRenamed(f'org_unit_{int(org_unit_output_level)}','org_unit_id')
    return df

# COMMAND ----------

def create_dataframe(df2, ratio_table):
    ratio_table = ratio_table.withColumn('ratio',ratio_table.ratio.cast(FloatType()))
    dfx = df2.join(ratio_table.select('org_unit_id','channel_id','product_id','ratio'), on =['org_unit_id','channel_id','product_id'] , how='left')
    dfx = dfx.withColumn('forecast_new',dfx.forecast * dfx.ratio).drop("ratio",'forecast').withColumnRenamed("forecast_new","forecast") 
    return dfx

# COMMAND ----------

def agg_dis_agg_function(df, Heirarchy,product_output_level, channel_output_level, org_unit_output_level, product_input_level,channel_input_level, org_unit_input_level):

    df = df.withColumn('product_id',df.product_id.cast(StringType()))

    if product_input_level != product_output_level:
        level_difference = product_output_level - product_input_level
        suffix = 'product'

        if level_difference > 0:
            p_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'product'], 'product')
            p_file = spark.createDataFrame(p_file)
            df = product_aggregator(df, p_file, product_input_level, product_output_level)

        elif level_difference < 0:
            p_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'product'], 'product')
            p_file = spark.createDataFrame(p_file)
            df = product_dis_aggregator(df, p_file, product_input_level, product_output_level)
                
    if channel_input_level != channel_output_level:
        level_difference = channel_output_level - channel_input_level
        suffix = 'channel'
        
        if level_difference > 0:
            
            c_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'channel'], 'channel')
            p_file = spark.createDataFrame(p_file)
            df = channel_aggregator(df, c_file, channel_input_level, channel_output_level)
            
        elif level_difference < 0:
            c_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'channel'], 'channel')
            c_file = spark.createDataFrame(c_file)
            df = channel_dis_aggregator(df, c_file, channel_input_level, channel_output_level)

    if org_unit_input_level != org_unit_output_level:
        level_difference = org_unit_output_level - org_unit_input_level
        suffix = 'org_unit'
        if level_difference > 0:
            o_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit'], 'org_unit')
            o_file = spark.createDataFrame(o_file)
            df = org_unit_aggregator(df, o_file, org_unit_input_level, org_unit_output_level)
        elif level_difference < 0:
            o_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit'], 'org_unit')
            o_file = spark.createDataFrame(o_file)
            df = org_unit_dis_aggregator(df, o_file, org_unit_input_level, org_unit_output_level)
    
    df = df.withColumn("historical_sale", round(df.historical_sale, 0)).dropDuplicates()
    df = df.withColumn('historical_sale',df.historical_sale.cast(FloatType())).withColumn('forecast',df.forecast.cast(FloatType()))    
    return (df)

# COMMAND ----------

def final_agg(df_final):
    if (product_input_level == product_output_level) and (channel_input_level == channel_output_level) and (org_unit_input_level == org_unit_output_level):
        pass
   
  
    elif (product_input_level > product_output_level) or (channel_input_level > channel_output_level) or (org_unit_input_level > org_unit_output_level):
        pass
        print("Input level cannot be greater than output level")
    elif (product_input_level < product_output_level) or (channel_input_level < channel_output_level) or (org_unit_input_level < org_unit_output_level):
        df_final = agg_dis_agg_function(df_final,df_hierarchy_new, product_output_level, channel_output_level, org_unit_output_level,
                                        product_input_level, channel_input_level, org_unit_input_level)
        df_final = df_final.withColumn('promotion',lit(''))
        df_final = df_final.withColumn('segmentation',lit(''))
        df_final = df_final.withColumn('model_id',lit(''))


    df_final = df_final.withColumn('forecast',df_final.forecast.cast(FloatType()))
   
    df_final = df_final.sort(col("org_unit_id"),col("channel_id"),col('product_id'),col('period'))

    return df_final

# COMMAND ----------

def fetch_run_data(run_id):
    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F3_run_data.parquet"
    run_data = spark.read.format('parquet').load(working_path,inferSchema=True)
    return run_data

# COMMAND ----------

def fetch_original_data(fetch_status):
    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_{fetch_status}_run_data.parquet"
    run_data1 = spark.read.format('parquet').load(working_path,inferSchema=True)
    #run_data1 = run_data1.toPandas()
    
    return run_data1

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)
    
    return

# COMMAND ----------

try:
    pipeline_flag = 0

    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F3_run_data.parquet"
    df3 = spark.read.parquet(working_path, inferSchema = True, header= True)

    
    if (product_forecast_level == product_input_level) and (channel_forecast_level == channel_input_level) and (org_unit_forecast_level == org_unit_input_level):
        print('Disaggregation not required')
        fetch_status = 'F3'
        df2 = fetch_original_data(fetch_status)
        if 'keys' in df2.columns:
            print('original_sales_data Already has keys')
        else:
            df2 = create_keys(df2)
            #df2 = df2.reset_index(drop=False)
        df2 = df2.withColumn('promotion',lit(''))
        df2 = df2.withColumn('segmentation',lit(''))
        df2 = df2.withColumn('model_id',lit(''))

    else:
        print('Disaggregation started')

        ratio_table = spark.read.jdbc(url=url, table= f"(SELECT * FROM proportion WHERE run_id = '{run_id}' and domain_id = '{domain_id}') as ratio_table", properties = properties)
        ratio_table = ratio_table.select('run_id', 'domain_id', 'org_unit_id', 'channel_id', 'product_id', 'ratio')
        
        #below line added to remove duplicates
        ratio_table = ratio_table.dropDuplicates()


        if ratio_table.count()>0:

            df = agg_dis_agg_function(df3,df_hierarchy_new ,product_input_level, channel_input_level, org_unit_input_level,
                                                   product_forecast_level, channel_forecast_level, org_unit_forecast_level)

            print('agg_dis_agg_function call completed')
            df = df.na.drop(subset=["forecast"])
            print('Ratio_table dataframe disagg creation started')
            df = create_dataframe(df, ratio_table)
            df = df.withColumn("keys",concat_ws("~","org_unit_id","channel_id","product_id"))
            print('Ratio_table dataframe disagg creation completed')
            df = df.na.drop(subset=["forecast"])
            df = df.sort(col("keys"),col("period"))
            df2 = df.withColumn('promotion',lit(''))
            df2 = df.withColumn('segmentation',lit(''))
            df2 = df.withColumn('model_id',lit(''))
            
    df_final = df2.na.drop(subset=['product_id', 'channel_id', 'org_unit_id', 'domain_id'])
    df_final = df_final.withColumn('forecast',df_final.forecast.cast(FloatType()))
    df_final = df_final.sort(col("org_unit_id"),col("channel_id"),col('product_id'),col('period'))
    df_final = final_agg(df_final)
    df_final = df_final.drop('keys')

#     # Ingesting in raw disagg data db
    print("Disaggregation completed, rounding logic started")
    df_final = df_final.withColumn("psuedo_key",concat_ws("~","org_unit_id","channel_id","product_id"))
    df_final = df_final.groupby("psuedo_key").apply(rounding_logic)
    print('rounding logic completed, future discontinue master started')
    if future_discontinue_master.count() == 0:
        print("future discontinue master is empty.\n future discontinue master is not applied")

    if future_discontinue_master.count() > 0:
        df_final = df_final.withColumn("key",concat_ws("~","org_unit_id","channel_id","product_id"))
        
        dffd = df_final.join(future_discontinue_master.select('key','period').withColumnRenamed('period','period_y'),on = 'key',how = 'left')

        dffd = dffd.withColumn('forecast',when((col("period") > col('period_y')),0).otherwise(col('forecast')))
    
        df_final  = dffd.select('domain_id', 'org_unit_id', 'channel_id','product_id', 'period', 'historical_sale', 'forecast', 'segmentation','model_id','promotion')
        
    print('future discontinue master completed')
    status = 'F5'
    df_final = df_final.withColumn('run_id',lit(run_id).cast(IntegerType()))
    df_final = df_final.withColumn('run_state',lit(status))
    df_final = df_final.select('domain_id', 'run_id', 'run_state', 'org_unit_id', 'channel_id','product_id', 'period', 'historical_sale', 'forecast', 'segmentation','model_id','promotion')

    
    
    
    print('Ingestion started')
    

    if ingestion_flag == 1:
        working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F5_run_data.parquet"
        df_final.repartition(1).write.mode('overwrite').parquet(working_path)
        #df_final = df_final.dropDuplicates()
        #print("Distinct count: "+str(df2.count()))
        

#         excel_path = f'/mnt/jnj_output_file/{env}/{domain_id}/Run_ID_{run_id}/F5_{run_id}.xlsx'
#         save_output1(df_final,excel_path,'F5')
        
    else:
        df_final_pd = df_final.toPandas()

    print('disaggregation sucessful')
    
except:
    traceback.print_exc()
    10/0

# COMMAND ----------


