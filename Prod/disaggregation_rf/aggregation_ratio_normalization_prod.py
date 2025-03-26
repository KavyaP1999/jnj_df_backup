# Databricks notebook source
ingestion_flag = 1 #ingestion in DB & blob storage if 0 no ingestion

# COMMAND ----------

# DBTITLE 1,Libraries
import numpy as np
import pandas as pd
import os,sys,re
import pyspark
from scipy import optimize
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
import datetime
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType, DoubleType,DecimalType
import traceback
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# DBTITLE 1,Running the configuration file
# MAGIC %run /Shared/Prod/configuration_prodV2

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

# DBTITLE 1,Reading the parameters
dbutils.widgets.text('run_id', '', 'run_id')
run_id= str(dbutils.widgets.get("run_id"))
#run_id =572

dbutils.widgets.text('domain_id', '', 'domain_id')
domain_id= dbutils.widgets.get("domain_id")
#domain_id = 'EMEA_PT'

#dbutils.widgets.text('forecast_period', '', 'forecast_period')
#input_period= dbutils.widgets.get("forecast_period")

#forecast_period =36 # commented out for check

dbutils.widgets.text('run_mode', '', 'run_mode')
run_mode= dbutils.widgets.get("run_mode")
print(run_mode)
#run_mode = 'MR'

dbutils.widgets.text('proportion_length', '', 'proportion_length')
proportion_length= int(dbutils.widgets.get("proportion_length"))

#below line added
#proportion_length = 9
ratio_weeks = int(proportion_length)

dbutils.widgets.text('input_period', '', 'input_period')
input_period= dbutils.widgets.get("input_period")
print(input_period)

dbutils.widgets.text("org_unit_input_level", "","")
org_unit_input_level = int(dbutils.widgets.get("org_unit_input_level"))

#org_unit_input_level = 0 # need to remove hardcoded

dbutils.widgets.text("channel_input_level", "","")# 2 lines comment for now need to uncomment
channel_input_level = int(dbutils.widgets.get("channel_input_level"))

#channel_input_level = 0# giving hard coded value

dbutils.widgets.text("product_input_level", "","")
product_input_level = int(dbutils.widgets.get("product_input_level"))
#product_input_level = 0

dbutils.widgets.text("org_unit_forecast_level", "","")
org_unit_forecast_level = int(dbutils.widgets.get("org_unit_forecast_level"))

#org_unit_forecast_level = 2 # hardcoded

dbutils.widgets.text("channel_forecast_level", "","")# need to uncomment these 2 lines
channel_forecast_level = int(dbutils.widgets.get("channel_forecast_level"))

#channel_forecast_level = 0 # hard coded one

dbutils.widgets.text("product_forecast_level", "","")
product_forecast_level = int(dbutils.widgets.get("product_forecast_level"))
#product_forecast_level =1 # hard coded values

#dbutils.widgets.text("ref_master_flag", "","")#uncomment these 2 lines later
#ref_master_flag = int(dbutils.widgets.get("ref_master_flag"))
#ref_master_flag = 1

org_unit_normalization_level = org_unit_forecast_level
channel_normalization_level = channel_forecast_level
product_normalization_level = product_forecast_level

# COMMAND ----------

# DBTITLE 1,Reading the data
df_hierarchy = spark.read.jdbc(url=url, table= f"(SELECT * FROM hierarchy WHERE domain_id = '{domain_id}') AS hm", properties = properties)
df_hierarchy_new=df_hierarchy.toPandas()

working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP1_run_data.parquet"
df_rundata = spark.read.format('parquet').load(working_path,inferSchema=True,header=True)

#changed below code with run_mode parameter instead of ref_master_flag
#if ref_master_flag == 1 :
 #   reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{domain_id}') as   reference_master",properties = properties)


reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{domain_id}') as   reference_master",properties = properties)


# COMMAND ----------

#reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '589' and domain_id = '{domain_id}') as   reference_master",properties = properties)
#reference_master.display()
#if reference_master.count()>0:
    #print("hi")



# COMMAND ----------

df_rundata.display()

# COMMAND ----------

# DBTITLE 1,Generate_hierarchy (creates the hierarchy data in wide format instade of long)
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

# DBTITLE 1,Aggregation functions
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

# DBTITLE 1,The main aggregation and disaggregation function which calls the individual functions and is being called in the dataprocessing
def agg_dis_agg_function(df, Heirarchy, product_output_level, channel_output_level, org_unit_output_level, product_input_level=0,
                         channel_input_level=0, org_unit_input_level=0):
    df = df.withColumn('historical_sale',df.historical_sale.cast(IntegerType())).withColumn('product_id',df.product_id.cast(StringType()))

   
    
    hm = pd.DataFrame()
    if product_input_level != product_output_level:
       
        level_difference = product_output_level - product_input_level
        suffix = 'product'

        if level_difference > 0:
            p_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'product'], 'product')
            hmt = p_file[[f'{suffix}_{product_input_level}',f'{suffix}_{product_output_level}']]
            hmt.rename(columns={f'{suffix}_{int(product_input_level)}': 'heirarchy_value',f'{suffix}_{int(product_output_level)}': 'parent_value'}, inplace=True)
            p_file = spark.createDataFrame(p_file)
            df = product_aggregator(df, p_file, product_input_level, product_output_level)
            
        elif level_difference < 0:
            print("no product aggregation required")
            
        print(hmt.shape)
    
        hm = hm.append(hmt)

    if channel_input_level != channel_output_level:
        level_difference = channel_output_level - channel_input_level
        suffix = 'channel'
        if level_difference > 0:
            c_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'channel'], 'channel')
            hmt = c_file[[f'{suffix}_{channel_input_level}',f'{suffix}_{channel_output_level}']]
            hmt.rename(columns={f'{suffix}_{int(channel_input_level)}': 'heirarchy_value',f'{suffix}_{int(channel_output_level)}': 'parent_value'}, inplace=True)
            c_file = spark.createDataFrame(c_file)
            df = channel_aggregator(df, c_file, channel_input_level, channel_output_level)

        elif level_difference <= 0:
            print("no channel aggregation required")
    
        hm = hm.append(hmt)
    
    if org_unit_input_level != org_unit_output_level:
        level_difference = org_unit_output_level - org_unit_input_level
        suffix = 'org_unit'
        if level_difference > 0:
            o_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit'], 'org_unit')
            hmt = o_file[[f'{suffix}_{org_unit_input_level}',f'{suffix}_{org_unit_output_level}']]
            hmt.rename(columns={f'{suffix}_{int(org_unit_input_level)}': 'heirarchy_value',f'{suffix}_{int(org_unit_output_level)}': 'parent_value'}, inplace=True)
            o_file = spark.createDataFrame(o_file)
            df = org_unit_aggregator(df, o_file, org_unit_input_level, org_unit_output_level)

        elif level_difference < 0:
            print("no org unit aggregation required")
        hm = hm.append(hmt)
    
    df = df.withColumn("historical_sale", round(df.historical_sale, 0)).dropDuplicates()
    df = df.withColumn('historical_sale',df.historical_sale.cast(FloatType())).withColumn('forecast',df.forecast.cast(FloatType()))

    
    hm.drop_duplicates(keep = 'first',inplace = True)
    
    return (df,hm)

# COMMAND ----------

# DBTITLE 1,Calculate_ratios() gives the ratios which will be used for disaggregation
def calculate_ratios(df1, df_original1,hierarchy_map, domain_id, run_mode, run_id, ratio_weeks=52): #changes to 24 instead of 52 weeks
        
    lst = ['product','channel','org_unit']
    for i in lst:
        df = df_original1.withColumnRenamed(f'{i}_id','heirarchy_value').join(hierarchy_map, on = 'heirarchy_value', how='left').drop(hierarchy_map.heirarchy_value).withColumnRenamed('parent_value', f'parent_{i}_id').withColumnRenamed('heirarchy_value', f'{i}_id')
        df_original1 = df
    df = df.withColumn('parent_product_id', when( df.parent_product_id.isNull(), df.product_id).otherwise(df.parent_product_id))
    df = df.withColumn('parent_channel_id', when( df.parent_channel_id.isNull(), df.channel_id).otherwise(df.parent_channel_id))
    df = df.withColumn('parent_org_unit_id', when( df.parent_org_unit_id.isNull(), df.org_unit_id).otherwise(df.parent_org_unit_id))
    
    df = df.withColumnRenamed('historical_sale','historical_sale_original')
    
    df3 = df1
    df3 = df3.withColumnRenamed('org_unit_id','agg_org_unit_id').withColumnRenamed('product_id','agg_product_id').withColumnRenamed('channel_id','agg_channel_id')
    
    df = df.withColumnRenamed('parent_org_unit_id','agg_org_unit_id' ).withColumnRenamed('parent_channel_id','agg_channel_id' ).withColumnRenamed('parent_product_id','agg_product_id' ).join(df3, on =['domain_id','agg_org_unit_id','agg_channel_id','agg_product_id','period'], how = 'left')

    periods = df.select("period").distinct().sort('period').rdd.flatMap(lambda x:x).collect()
    last_period_ratio = periods[-ratio_weeks:len(periods)]
    
    df_temp = df.filter(df.period.isin(last_period_ratio))
    df_temp = df_temp.groupBy('org_unit_id','product_id','channel_id').sum('historical_sale', 'historical_sale_original').withColumnRenamed('sum(historical_sale)','historical_sale').withColumnRenamed('sum(historical_sale_original)','historical_sale_original')
    
    df_temp = df_temp.withColumn('historical_sale', when( df_temp.historical_sale == 0, df_temp.historical_sale + 1).otherwise(df_temp.historical_sale))

    df_temp = df_temp.withColumn('ratio', df_temp['historical_sale_original'] / df_temp['historical_sale'])
    df_temp = df_temp.withColumn('domain_id', lit(domain_id))
    ratio_table = df_temp

    ratio_table = ratio_table.withColumn('run_id', lit(run_id))
    ratio_table = ratio_table.select('run_id', 'domain_id', 'org_unit_id', 'channel_id', 'product_id', 'ratio')
    ratio_table = ratio_table.withColumn('ratio',ratio_table.ratio.cast(FloatType()))
#     ratio_table = spark.createDataFrame(data = ratio_table, schema = ratio_table_Schema)


    return ratio_table

# COMMAND ----------

ratio_table_Schema = StructType([StructField('run_id',StringType(),True),
                        StructField('org_unit_id',StringType(),True),
                        StructField('product_id',StringType(),True),
                        StructField('channel_id',StringType(),True),
                        StructField('domain_id',StringType(),True),
                        StructField('ratio',FloatType(),True)
                       ])

# COMMAND ----------

def get_allocation(input_period):
    allocated_period = re.split('W|D|M|Q', input_period)[1]
    return allocated_period

def get_org(input_data):
    org = re.split('~', input_data)[0]
    return org


def get_channel(input_data):
    channel = re.split('~', input_data)[1]
    return channel


def get_product(input_data):
    product = re.split('~', input_data)[2]
    return product

# COMMAND ----------

outSchema = StructType([StructField('keys',StringType(),True),
                        StructField('period',StringType(),True),
                        StructField('fn_ratio',FloatType(),True),
                        StructField('domain_id',StringType(),True),
                        StructField('org_unit_id',StringType(),True),
                        StructField('channel_id',StringType(),True),
                        StructField('product_id',StringType(),True)])

# COMMAND ----------

# DBTITLE 1,forecast_allocation_ratios
@pandas_udf(outSchema, PandasUDFType.GROUPED_MAP)
def forecast_allocation_ratios(df):
    #df = df1.copy()
#     df['keys'] = df['org_unit_id'].map(str) + '~' + df['channel_id'].map(str) + '~' + df['product_id'].map(str)
#     print(df)
#     df, keys = create_keys(df)
    df['historical_sale'] = df['historical_sale'].astype(float)
    df_allocations = pd.DataFrame()
#     for i in np.unique(df['keys']):
#         df_temp = df[df['keys'] == i]
    df_temp = df.sort_values(by=['period'], ascending=True).reset_index(drop=True)
#     if run_mode == 'poc':
#         df_temp = df_temp[0:-forecast_length]
    df_temp['period'] = df_temp['period'].astype(str)
    df_temp['allocated_period'] = df_temp['period'].apply(get_allocation)
    print(df_temp)
    df_temp = df_temp[['keys', 'allocated_period', 'historical_sale']]
    df_temp['historical_sale'] = df_temp['historical_sale'].astype(float)
    df_temp = df_temp.groupby(['keys', 'allocated_period'])['historical_sale'].mean().reset_index()
    print(df_temp)
    df_allocations = pd.concat(([df_allocations, df_temp]), axis=0, ignore_index=True)
    df_allocations.columns = ['keys', 'period', 'fn_ratio']
    df_allocations.insert(loc=3, column='domain_id', value=domain_id)
    df_allocations['org_unit_id'] = df_allocations['keys'].apply(get_org)
    df_allocations['channel_id'] = df_allocations['keys'].apply(get_channel)
    df_allocations['product_id'] = df_allocations['keys'].apply(get_product)
    # print("Flag -1:", df_allocations.columns)
    # print(df_allocations)
    return df_allocations

# COMMAND ----------

# DBTITLE 1,Forecast normalization data, output schema
ratio_table_Schema = StructType([StructField('run_id',StringType(),True),
                        StructField('org_unit_id',StringType(),True),
                        StructField('product_id',StringType(),True),
                        StructField('channel_id',StringType(),True),
                        StructField('domain_id',StringType(),True),
                        StructField('ratio',FloatType(),True)
                       ])

forecast_normalization_ratio_table2_Schema = StructType([StructField('run_id',StringType(),True),
                        StructField('feature_key_ref',StringType(),True),
                        StructField('feature_value',FloatType(),True),
                        StructField('period',StringType(),True),
                        StructField('domain_id',StringType(),True)
                       ])


# COMMAND ----------

# old block commented out new code is below cell

# try:
    
#     if (product_forecast_level == product_input_level) and (channel_forecast_level == channel_input_level) and (org_unit_forecast_level == org_unit_input_level):
        
#         #if ref_master_flag == 1: changed this line with run_mode parameter
#         if run_mode == 'MR':
#             df_rundata = df_rundata.withColumn('key',concat_ws('@',df_rundata.org_unit_id,df_rundata.channel_id,df_rundata.product_id))
#             reference_master = reference_master.withColumn('key',concat_ws('@',reference_master.org_unit_id,reference_master.channel_id,reference_master.product_id))        
            
#             l = reference_master.select(col('key')).rdd.flatMap(lambda x:x).collect()
#             df_rundata = df_rundata.filter(df_rundata.key.isin(l)).drop('key')
            
#        ## Ingesting in the blob storage
        
#         if ingestion_flag==1:
            
    
#             working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP3_Int_data.parquet"
#             df_rundata1 = df_rundata.drop('run_id','run_state','segmentation','promotion','model_id')
#             df_rundata1.repartition(128).write.mode('overwrite').parquet(working_path)
            
#         else:
#             df_rundata1 = df_rundata1.toPandas()

#     else:
#         agg_dis_agg,hm=agg_dis_agg_function(df_rundata,df_hierarchy_new,product_forecast_level,channel_forecast_level,org_unit_forecast_level,
#                                      product_input_level,channel_input_level, org_unit_input_level)

#         #if ref_master_flag == 1 : # changed this with run_mode line only rest is same
#         if run_mode == 'MR':           
#             agg_dis_agg = agg_dis_agg.withColumn('key',concat_ws('@',agg_dis_agg.org_unit_id,agg_dis_agg.channel_id,agg_dis_agg.product_id))
#             reference_master = reference_master.withColumn('key',concat_ws('@',reference_master.org_unit_id,reference_master.channel_id,reference_master.product_id))        
#             l = reference_master.select(col('key')).rdd.flatMap(lambda x:x).collect()
#             agg_dis_agg = agg_dis_agg.filter(agg_dis_agg.key.isin(l)).drop('key')

#         #Ingesting in the blob storage
#         working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP3_Int_data.parquet"
#         if ingestion_flag == 1:
#             agg_dis_agg.repartition(128).write.mode('overwrite').parquet(working_path)
#         else:
#             agg_dis_agg_pd = agg_dis_agg.toPandas()
        
#         hm = spark.createDataFrame(hm)
#                 # Ratio creation
#         ratio_table = calculate_ratios(agg_dis_agg, df_rundata,hm, domain_id, run_mode, run_id, ratio_weeks)
        
#         if ingestion_flag == 1:
#             db_ingestion(ratio_table, 'proportion', 'append')
#         else:
#             ratio_table = ratio_table.toPandas()

#         print('aggreagtion and ratio creation and normalization table creation successful')

#     historical_sales_data4 = df_rundata
#     historical_sales_data4,hm1 = agg_dis_agg_function(historical_sales_data4,df_hierarchy_new, product_normalization_level,
#                                                               channel_normalization_level, org_unit_normalization_level,
#                                                               product_input_level,
#                                                               channel_input_level, org_unit_input_level)

#     print('Forecast allocation started')
#     historical_sales_data4 = historical_sales_data4.withColumn('org_unit_id',col('org_unit_id').cast(StringType())). \
#     withColumn('channel_id',col('channel_id').cast(StringType())). \
#     withColumn('product_id',col('product_id').cast(StringType()))
#     historical_sales_data4 = historical_sales_data4.withColumn('keys',concat_ws('~',historical_sales_data4.org_unit_id,historical_sales_data4.channel_id,historical_sales_data4.product_id))
#     forecast_normalization_ratio_table = historical_sales_data4.repartition('keys').groupby('keys').apply(forecast_allocation_ratios)
#     forecast_normalization_ratio_table = forecast_normalization_ratio_table.drop_duplicates()
#     forecast_normalization_ratio_table = forecast_normalization_ratio_table.drop('org_unit_id','channel_id','product_id')
#     forecast_normalization_ratio_table = forecast_normalization_ratio_table.withColumnRenamed('keys','feature_key_ref').withColumnRenamed('fn_ratio','feature_value')
#     forecast_normalization_ratio_table = forecast_normalization_ratio_table.withColumn('run_id',lit(run_id))
#     forecast_normalization_ratio_table = forecast_normalization_ratio_table.select('run_id', 'feature_key_ref', 'feature_value', 'period', 'domain_id')
#     forecast_normalization_ratio_table.count()

#     #     DB ingestion normalization table# these 4 lines need to uncomment later
#     #if ingestion_flag==1:
        
#      #   db_ingestion(forecast_normalization_ratio_table, 'normalization', 'append')
        
#     #else:
        
#      #   forecast_normalization_ratio_table_pd = forecast_normalization_ratio_table.toPandas()
    
    
# except:
#     traceback.print_exc()
#     print('aggreagtion or ratio creation or normalization table creation failed')
#     10/0

# COMMAND ----------

try:
    
    if (product_forecast_level == product_input_level) and (channel_forecast_level == channel_input_level) and (org_unit_forecast_level == org_unit_input_level):
        
        #if ref_master_flag == 1: changed this line with run_mode parameter
        #new block change instead of passing value from backend ref master flag--------
        #if run_mode == 'MR':
        if reference_master.count()>0:

            df_rundata = df_rundata.withColumn('key',concat_ws('@',df_rundata.org_unit_id,df_rundata.channel_id,df_rundata.product_id))
            reference_master = reference_master.withColumn('key',concat_ws('@',reference_master.org_unit_id,reference_master.channel_id,reference_master.product_id))        
            
            l = reference_master.select(col('key')).rdd.flatMap(lambda x:x).collect()
            df_rundata = df_rundata.filter(df_rundata.key.isin(l)).drop('key')
            
       ## Ingesting in the blob storage
        
        if ingestion_flag==1:
            
    
            working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP3_Int_data.parquet"
            df_rundata1 = df_rundata.drop('run_id','run_state','segmentation','promotion','model_id')
            df_rundata1.repartition(128).write.mode('overwrite').parquet(working_path)
            
        else:
            df_rundata1 = df_rundata1.toPandas()

    else:
        agg_dis_agg,hm=agg_dis_agg_function(df_rundata,df_hierarchy_new,product_forecast_level,channel_forecast_level,org_unit_forecast_level,
                                     product_input_level,channel_input_level, org_unit_input_level)

        #if ref_master_flag == 1 : # changed this with run count line only rest is same
        if reference_master.count()>0:
                  
            agg_dis_agg = agg_dis_agg.withColumn('key',concat_ws('@',agg_dis_agg.org_unit_id,agg_dis_agg.channel_id,agg_dis_agg.product_id))
            reference_master = reference_master.withColumn('key',concat_ws('@',reference_master.org_unit_id,reference_master.channel_id,reference_master.product_id))        
            l = reference_master.select(col('key')).rdd.flatMap(lambda x:x).collect()
            agg_dis_agg = agg_dis_agg.filter(agg_dis_agg.key.isin(l)).drop('key')

        #Ingesting in the blob storage
        working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP3_Int_data.parquet"
        if ingestion_flag == 1:
            agg_dis_agg.repartition(128).write.mode('overwrite').parquet(working_path)
        else:
            agg_dis_agg_pd = agg_dis_agg.toPandas()
        
        hm = spark.createDataFrame(hm)
                # Ratio creation
        ratio_table = calculate_ratios(agg_dis_agg, df_rundata,hm, domain_id, run_mode, run_id, ratio_weeks)
        
        if ingestion_flag == 1:
            db_ingestion(ratio_table, 'proportion', 'append')
        else:
            ratio_table = ratio_table.toPandas()

        print('aggreagtion and ratio creation and normalization table creation successful')

    historical_sales_data4 = df_rundata
    historical_sales_data4,hm1 = agg_dis_agg_function(historical_sales_data4,df_hierarchy_new, product_normalization_level,
                                                              channel_normalization_level, org_unit_normalization_level,
                                                              product_input_level,
                                                              channel_input_level, org_unit_input_level)

    print('Forecast allocation started')
    historical_sales_data4 = historical_sales_data4.withColumn('org_unit_id',col('org_unit_id').cast(StringType())). \
    withColumn('channel_id',col('channel_id').cast(StringType())). \
    withColumn('product_id',col('product_id').cast(StringType()))
    historical_sales_data4 = historical_sales_data4.withColumn('keys',concat_ws('~',historical_sales_data4.org_unit_id,historical_sales_data4.channel_id,historical_sales_data4.product_id))
    forecast_normalization_ratio_table = historical_sales_data4.repartition('keys').groupby('keys').apply(forecast_allocation_ratios)
    forecast_normalization_ratio_table = forecast_normalization_ratio_table.drop_duplicates()
    forecast_normalization_ratio_table = forecast_normalization_ratio_table.drop('org_unit_id','channel_id','product_id')
    forecast_normalization_ratio_table = forecast_normalization_ratio_table.withColumnRenamed('keys','feature_key_ref').withColumnRenamed('fn_ratio','feature_value')
    forecast_normalization_ratio_table = forecast_normalization_ratio_table.withColumn('run_id',lit(run_id))
    forecast_normalization_ratio_table = forecast_normalization_ratio_table.select('run_id', 'feature_key_ref', 'feature_value', 'period', 'domain_id')
    forecast_normalization_ratio_table.count()

    #     DB ingestion normalization table# these 4 lines need to uncomment later
    #if ingestion_flag==1:
        
       #db_ingestion(forecast_normalization_ratio_table, 'normalization', 'append')
        
    #else:
        
       #forecast_normalization_ratio_table_pd = forecast_normalization_ratio_table.toPandas()
    
    
except:
    traceback.print_exc()
    print('aggreagtion or ratio creation or normalization table creation failed')
    10/0

# COMMAND ----------


