# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType,StringType

# COMMAND ----------

# DBTITLE 1,Generate Hierarchy function - reverse hiearchy
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

def product_aggregator(df, Heirarchy, product_input_level, product_output_level,run_id):
        
    if 'product_id' not in df.columns:
        df = df.withColumn('org_unit_id',split(df['key'],'@').getItem(0)) \
        .withColumn('channel_id',split(df['key'],'@').getItem(1)) \
        .withColumn('product_id',split(df['key'],'@').getItem(2))
    suffix = 'product'
    hm = Heirarchy.select(col(f'product_{product_input_level}'),col(f'product_{product_output_level}'))

    df = df.join(hm, df.product_id == hm[f'{suffix}_{int(product_input_level)}'],how='left')
    df = df.drop('product_id',f'{suffix}_{int(product_input_level)}')
    df = df.withColumnRenamed(f'{suffix}_{int(product_output_level)}','product_id')
    df = df.withColumn('cdh',col('cdh').cast(FloatType())).withColumn('expost_sales',col('expost_sales').cast(FloatType()))
    df = df.groupBy('domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period').agg(sum('cdh').alias('cdh'),sum('expost_sales').alias('expost_sales'))
    return df

# COMMAND ----------

def channel_aggregator(df, Heirarchy, channel_input_level, channel_output_level,run_id):
    
    if 'channel_id' not in df.columns:
         df = df.withColumn('org_unit_id',split(df['key'],'@').getItem(0)) \
        .withColumn('channel_id',split(df['key'],'@').getItem(1)) \
        .withColumn('product_id',split(df['key'],'@').getItem(2))
    
    suffix = 'channel'
    hm = Heirarchy.select(col(f'channel_{channel_input_level}'),col(f'channel_{channel_output_level}'))

    df = df.join(hm, df.channel_id == hm[f'{suffix}_{int(channel_input_level)}'],how='left')
    df = df.drop('channel_id',f'{suffix}_{int(channel_input_level)}')
    df = df.withColumnRenamed(f'{suffix}_{int(channel_output_level)}','channel_id')
    df = df.withColumn('cdh',col('cdh').cast(FloatType())).withColumn('expost_sales',col('expost_sales').cast(FloatType()))
    df = df.groupBy('domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period').agg(sum('cdh').alias('cdh'),sum('expost_sales').alias('expost_sales'))
    
    return df

# COMMAND ----------

def org_unit_aggregator(df, Heirarchy, org_unit_input_level, org_unit_output_level,run_id):
    
    if 'org_unit_id' not in df.columns:
         df = df.withColumn('org_unit_id',split(df['key'],'@').getItem(0)) \
        .withColumn('channel_id',split(df['key'],'@').getItem(1)) \
        .withColumn('product_id',split(df['key'],'@').getItem(2))
    
    suffix = 'org_unit'
    
    hm = Heirarchy.select(col(f'org_unit_{org_unit_input_level}'),col(f'org_unit_{org_unit_output_level}'))

    df = df.join(hm, df.org_unit_id == hm[f'{suffix}_{int(org_unit_input_level)}'],how='left')
    df = df.drop('org_unit_id',f'{suffix}_{int(org_unit_input_level)}')
    df = df.withColumnRenamed(f'{suffix}_{int(org_unit_output_level)}','org_unit_id')
    df = df.withColumn('cdh',col('cdh').cast(FloatType())).withColumn('expost_sales',col('expost_sales').cast(FloatType()))
    df = df.groupBy('domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period').agg(sum('cdh').alias('cdh'),sum('expost_sales').alias('expost_sales'))
    
    return df

# COMMAND ----------

# DBTITLE 1,History master agg function
def history_master_agg_function(df, Heirarchy, product_output_level, channel_output_level, org_unit_output_level, product_input_level,
                                 channel_input_level, org_unit_input_level,run_id):
#     df = df1.reset_index(drop=True)
#     df['product_id'] = df['product_id'].astype(str)
    df = df.withColumn('product_id',col('product_id').cast(StringType()))
    
    if product_input_level != product_output_level:
        level_difference = product_output_level - product_input_level
        if level_difference > 0:
            p_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'product'], 'product')
            p_file = spark.createDataFrame(p_file)
            df = product_aggregator(df, p_file, product_input_level, product_output_level,run_id)
        elif level_difference < 0:
            print("Product Level aggregation is not required")

   
    if channel_input_level != channel_output_level:
        level_difference = channel_output_level - channel_input_level
        if level_difference > 0:
            c_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'channel'], 'channel')
            c_file = spark.createDataFrame(c_file)
            df = channel_aggregator(df, c_file, channel_input_level, channel_output_level,run_id)
        elif level_difference < 0:
            print("Channel level aggregation is not required")

   
    if org_unit_input_level != org_unit_output_level:
        level_difference = org_unit_output_level - org_unit_input_level
        if level_difference > 0:
            o_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit'], 'org_unit')
            o_file = spark.createDataFrame(o_file)
            df = org_unit_aggregator(df, o_file, org_unit_input_level, org_unit_output_level,run_id)
        elif level_difference < 0:
            print("Org_Unit level aggregation is not required")
    df = df.select('domain_id','org_unit_id','channel_id','product_id', 'period', 'cdh', 'expost_sales')
    df = df.drop_duplicates()
    
    return df
