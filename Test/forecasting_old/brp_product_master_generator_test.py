# Databricks notebook source
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

# user = 'dbadmin'
# password = 'Summer@123456789'
# port = '3342'
# server = 'jnj-md-sqldb.public.53da1f976c5f.database.windows.net'
# databaseName = 'jnj_db_dbx_test'
# properties = {
#  "user" : user,
#  "password" : password 
# }
# url = "jdbc:sqlserver://{0}:{1};database={2}".format(server,port,databaseName)

# COMMAND ----------

def run_parameter_data(run_id,url,properties):
    
    parameter = spark.read.jdbc(url=url,table=f"(select parameter_id,run_value from run_parameter where run_id = '{run_id}') as run_parameter" ,properties=properties)
    parameter = parameter.toPandas()
   
    parameter = dict(zip(parameter['parameter_id'],parameter['run_value']))
    
    return parameter

# COMMAND ----------

def fetch_f3_data(run_id: str,domain_id):    
#     status = 'F3'
    filename = f'run_data_all_model_{run_id}.parquet'
    folderPath=f"/dbfs/mnt/disk/staging_data/{env}/{domain_id}/" + filename
    f3_data = spark.read.parquet(folderPath,inferSchema=True,header=True)
    f3_data = f3_data.toPandas()
    return f3_data

# COMMAND ----------

# def fetch_f3_data(run_id: str,url,properties):
    
#     status = 'F3'
#     f3_data = spark.read.jdbc(url=url, table= f"(SELECT * FROM run_data WHERE run_id = '{run_id}' and  run_state = '{status}') as f3_data", properties = properties)
#     f3_data = f3_data.toPandas()
#     return f3_data

# COMMAND ----------

def df_from_database(url,properties):
    
    df = spark.read.jdbc(url=url, table= f"(SELECT domain_id,hierarchy_type,hierarchy_value,parent_value,level_id,description FROM hierarchy) as df" , properties = properties)
    df = df.toPandas()
    return df

# COMMAND ----------

def generate_hierarchy(data, suffix='product'):
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

def fetch_ref_master(run_id: str,url,properties):
    
    df = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}') as df", properties = properties)
    df = df.toPandas()
    return df

# COMMAND ----------

def get_run_user(run_id:str,url,properties):
    
    run_id = str(run_id)
    run = spark.read.jdbc(url=url, table= f"(SELECT * FROM run WHERE run_id = '{run_id}') as run", properties = properties)
    run.createOrReplaceTempView('run')
    user = spark.read.jdbc(url=url, table= f"(SELECT * FROM [user]) as u", properties = properties)
    user.createOrReplaceTempView('user')
    user_name = spark.sql('select user_name from user u join run r on r.user_id=u.id')
    user_name = user_name.toPandas()

    return user_name['user_name'].values[0]

# COMMAND ----------


def brp_product_master(run_id: str, parameters_values,url,properties):
    
    forecast_level_product = int(parameters_values[18])

    #get F3 data 
    f3_df = fetch_f3_data(run_id,domain_id)
#     f3_df = fetch_f3_data(run_id,url,properties)
    master1=f3_df[['run_id','domain_id', 'org_unit_id', 'channel_id', 'product_id']]
    master1.drop_duplicates(inplace=True)

    master1=master1.reset_index(drop=True)
    
    master1['key']=master1['org_unit_id'].astype(str)+'@'+master1['channel_id']+'@'+master1['product_id']
    master1=master1.rename(columns={"org_unit_id": "affiliate"})

    #get hierarchy file
    h_file = df_from_database(url,properties)
    h_file_desc = h_file.copy()

    h_file = h_file[h_file['hierarchy_type'] == 'product']

    master2 = generate_hierarchy(h_file, suffix='product')
    ### get descriptions from hierarchy file
    for i in master2.columns[:]:
        # print(i)
        # print(i.split('_'))
        if int(i.split('_')[1]) < forecast_level_product:
            master2[i] = ''
        result_data=pd.merge(h_file[['hierarchy_value', 'description']], master2, left_on='hierarchy_value', right_on=i, how='right')
        result_data.rename(columns={'description': f'{i}_desc'}, inplace=True)

        result_data.drop(['hierarchy_value'], axis=1, inplace=True)
        master2=result_data.copy()
    result_data=result_data.rename(columns={"product_{}".format(forecast_level_product): "product_id"})
    final_masterdata=pd.merge(master1, result_data[['product_id','product_0_desc','product_1_desc', 'product_2_desc', 'product_3_desc', 'product_4_desc',
        'product_5_desc']],on='product_id', how='left')
    final_masterdata.drop_duplicates(inplace=True)

    final_masterdata=final_masterdata.reset_index(drop=True)

    #For channel desc
    final_masterdata=pd.merge(h_file_desc[['hierarchy_value', 'description']], final_masterdata, left_on='hierarchy_value', right_on='channel_id', how='right')
    final_masterdata=final_masterdata.rename(columns={"description": "channel"})
    final_masterdata.drop(['hierarchy_value','channel_id'], axis=1, inplace=True)

    #For affiliate desc
    forecast_level_org_unit = int(parameters_values[16])
    h_file_desc_aff = h_file_desc[h_file_desc['level_id']==forecast_level_org_unit]
    final_masterdata['affiliate']=final_masterdata['affiliate'].astype(str)
    final_masterdata=pd.merge(h_file_desc_aff[['hierarchy_value', 'description']], final_masterdata, left_on='hierarchy_value', right_on='affiliate', how='right')
    final_masterdata=final_masterdata.rename(columns={"description": "Affiliate"})
    final_masterdata.drop(['hierarchy_value','affiliate'], axis=1, inplace=True)

    final_masterdata=final_masterdata.rename(columns={"product_5_desc": "p7","product_4_desc":"p6","product_3_desc":"p5",
                                                        "product_2_desc":"p4","product_1_desc":"p3","product_0_desc":"p2"})
    final_masterdata['key1']=final_masterdata['Affiliate'].astype(str)+'@'+final_masterdata['channel']+'@'+final_masterdata['product_id']

    final_masterdata = final_masterdata.drop_duplicates().reset_index(drop=True)

    try:
        ref_df = fetch_ref_master(run_id,url,properties)
    except:
        ref_df = pd.DataFrame()

    if len(ref_df)==0:
        user_name = get_run_user(run_id,url,properties)
        final_masterdata['planner'] = user_name
    else:
        ref_df['key'] = ref_df['org_unit_id'] + '@' + ref_df['channel_id'] + '@' + ref_df['product_id']
        final_masterdata = pd.merge(final_masterdata,ref_df[['key','planner']],on=['key'], how='left')

    final_masterdata.drop(['key1'], axis=1, inplace=True)
    final_masterdata=final_masterdata[['run_id','domain_id','key', 'planner', 'Affiliate','p7','p6',
                                       'p5','p4','p3','p2', 'product_id','channel']]


    final_masterdata.drop_duplicates(inplace=True)
    
#     final_masterdata['p2'] == lit('')
   

    return final_masterdata

# COMMAND ----------


