# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

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

# DBTITLE 1,Product aggregator for history master
def product_aggregator(df, Heirarchy, product_input_level, product_output_level):
    product_heirarchy = np.unique(Heirarchy[Heirarchy['hierarchy_type'] == 'product']['level_id'])
    prod_heirarchy = Heirarchy[Heirarchy['hierarchy_type'] == 'product']
    if product_output_level > np.amax(product_heirarchy):
        product_output_level = np.amax(product_heirarchy)
    product_input_index = int(np.where(product_heirarchy == product_input_level)[0])
    product_output_index = int(np.where(product_heirarchy == product_output_level)[0])
    aggregate_levels = [product_input_index,product_output_index]
    
    if 'product_id' not in df.columns:
        df[["org_unit_id", "channel_id", "product_id"]] = df.key.str.split("@", expand=True)
    
    suffix = 'product'
    
    hm = generate_hierarchy(prod_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{product_input_level}',f'{suffix}_{product_output_level}']]
    
    df = pd.merge(df, hm, left_on=['product_id'], right_on=f'{suffix}_{int(product_input_index)}', how='left')
    df.drop(columns={'product_id',f'{suffix}_{int(product_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(product_output_index)}': 'product_id'}, inplace=True)
    df['cdh'] = df['cdh'].astype(float)
    df['expost_sales'] = df['expost_sales'].astype(float)
    df = df.groupby(['domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period'])['cdh', 'expost_sales'].sum().reset_index()
    return(df)

# COMMAND ----------

# DBTITLE 1,Channel aggregator for history master
def channel_aggregator(df, Heirarchy, channel_input_level, channel_output_level):
    channel_heirarchy = np.unique(Heirarchy[Heirarchy['hierarchy_type'] == 'channel']['level_id'])
    chann_heirarchy = Heirarchy[Heirarchy['hierarchy_type'] == 'channel']
    if channel_output_level > np.amax(channel_heirarchy):
        channel_output_level = np.amax(channel_heirarchy)
    channel_input_index = int(np.where(channel_heirarchy == channel_input_level)[0])
    channel_output_index = int(np.where(channel_heirarchy == channel_output_level)[0])
    aggregate_levels = [channel_input_index,channel_output_index]
    
    if 'channel_id' not in df.columns:
        df[["org_unit_id", "channel_id", "product_id"]] = df.key.str.split("@", expand=True)
    
    suffix = 'channel'
    
    hm = generate_hierarchy(chann_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{channel_input_level}',f'{suffix}_{channel_output_level}']]
    
    df = pd.merge(df, hm, left_on=['channel_id'], right_on=f'{suffix}_{int(channel_input_index)}', how='left')
    df.drop(columns={'channel_id',f'{suffix}_{int(channel_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(channel_output_index)}': 'channel_id'}, inplace=True)
    df['cdh'] = df['cdh'].astype(float)
    df['expost_sales'] = df['expost_sales'].astype(float)
    df = df.groupby(['domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period'])['cdh', 'expost_sales'].sum().reset_index()
    return(df)

# COMMAND ----------

# DBTITLE 1,Org_unit aggregator for history master
def org_unit_aggregator(df, Heirarchy, org_unit_input_level, org_unit_output_level):
    org_unit_heirarchy = np.unique(Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit']['level_id'])
    org_heirarchy = Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit']
    if org_unit_output_level > np.amax(org_unit_heirarchy):
        org_unit_output_level = np.amax(org_unit_heirarchy)
    org_unit_input_index = int(np.where(org_unit_heirarchy == org_unit_input_level)[0])
    org_unit_output_index = int(np.where(org_unit_heirarchy == org_unit_output_level)[0])
    aggregate_levels = [org_unit_input_index,org_unit_output_index]
    
    if 'org_unit_id' not in df.columns:
        df[["org_unit_id", "channel_id", "product_id"]] = df.key.str.split("@", expand=True)
    
    suffix = 'org_unit'
    
    hm = generate_hierarchy(org_heirarchy,suffix = suffix)
    hm = hm[[f'{suffix}_{org_unit_input_level}',f'{suffix}_{org_unit_output_level}']]
    
    df = pd.merge(df, hm, left_on=['org_unit_id'], right_on=f'{suffix}_{int(org_unit_input_index)}', how='left')
    df.drop(columns={'org_unit_id',f'{suffix}_{int(org_unit_input_index)}'}, inplace=True)
    df.rename(columns={f'{suffix}_{int(org_unit_output_index)}': 'org_unit_id'}, inplace=True)
    df['cdh'] = df['cdh'].astype(float)
    df['expost_sales'] = df['expost_sales'].astype(float)
    df = df.groupby(['domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period'])['cdh', 'expost_sales'].sum().reset_index()
    return(df)

# COMMAND ----------

# DBTITLE 1,History master agg function
def history_master_agg_function(df1, Heirarchy, product_output_level, channel_output_level, org_unit_output_level, product_input_level,
                                 channel_input_level, org_unit_input_level):
    df = df1.reset_index(drop=True)
    df['product_id'] = df['product_id'].astype(str)
#     domain_id = np.unique(np.array(df1.domain_id))[0]
#     db_obj = DatabaseWrapper()
#     conn = db_obj.connect()
#     Heirarchy = db_obj.execute_query(conn, query=f"SELECT * FROM hierarchy WHERE domain_id = '{domain_id}'")
#     Heirarchy_cols = ['domain_id', 'hierarchy_type', 'hierarchy_value', 'parent_value', 'level_id', 'description']
#     Heirarchy.columns = Heirarchy_cols
    if product_input_level != product_output_level:
        level_difference = product_output_level - product_input_level
        if level_difference > 0:
            df = product_aggregator(df, Heirarchy, product_input_level, product_output_level)
        elif level_difference < 0:
            print("Product Level aggregation is not required")
#     df_p = df[df['product_id'] == 'CM0103084228'].display()

    if channel_input_level != channel_output_level:
        level_difference = channel_output_level - channel_input_level
        if level_difference > 0:
            df = channel_aggregator(df, Heirarchy, channel_input_level, channel_output_level)
        elif level_difference < 0:
            print("Channel level aggregation is not required")
#     df_c = df[df['product_id'] == 'CM0103084228'].display()
    
    if org_unit_input_level != org_unit_output_level:
        level_difference = org_unit_output_level - org_unit_input_level
        if level_difference > 0:
            df = org_unit_aggregator(df, Heirarchy, org_unit_input_level, org_unit_output_level)
        elif level_difference < 0:
            print("Org_Unit level aggregation is not required")
    
#     df_o = df[df['product_id'] == 'CM0103084228'].display()

#     df['run_id'] = run_id
    df = df[['domain_id','org_unit_id','channel_id','product_id', 'period', 'cdh', 'expost_sales']]
    df.drop_duplicates(keep='first', inplace=True)
#     df.drop(['run_state'], inplace = True, axis = 1)
    df = df.reset_index(drop=True)
    return df
