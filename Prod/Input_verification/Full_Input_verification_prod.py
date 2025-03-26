# Databricks notebook source
# dbutils.library.installPyPI("sqlalchemy")
# dbutils.library.installPyPI("pymssql")
# dbutils.library.installPyPI("azure")
# dbutils.library.restartPython()

# COMMAND ----------

import json
import pandas as pd
import numpy as np
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.functions as func
from azure.storage.blob import BlockBlobService
from pyspark.sql.functions import concat, col, lit, concat_ws,split
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import collections

# COMMAND ----------

# DBTITLE 1,Taking Domain Id
dbutils.widgets.text("domain_id", "","")
domain_id = dbutils.widgets.get("domain_id")
print ("domain_id:",domain_id)

# COMMAND ----------

error_log = np.empty(0)

# COMMAND ----------

# MAGIC %run /Shared/Prod/configuration_prodV2

# COMMAND ----------

# DBTITLE 1,Taking File names from azure blob storage and checking the file names
try:    
    from azure.storage.blob import BlockBlobService
    blob_service = BlockBlobService('jnjmddevstgacct','YeytTvTZCBfz7BcZjT4e4MrtsmFI+1g7wEBYkyrZ9W0beN0SaYRMHXNWUiSmX9CcWhjDu/0nhf2R6/PbSQHmWw==')
    blobfile = []
    generator = blob_service.list_blobs('customerdata', prefix=f"{env}/Full_Load/{domain_id}/", delimiter="")
    for blob in generator:
        blobname = blob.name.split('/')[-1]
        blobfile.append(blobname)
        print("\t Blob name: " + blob.name)
    print(blobfile)
    for i in blobfile:
        if 'period.csv' in blobfile:           
            my_list=['data_correction.csv','future_discontinue_master.csv', 'future_forecast_master.csv', 'npf_master.csv', 'nts_master.csv', 'past_forecast_master.csv', 'period.csv', 'sales_master.csv', 'transition_master.csv']
        else:
            my_list=['data_correction.csv','future_discontinue_master.csv', 'future_forecast_master.csv', 'npf_master.csv', 'nts_master.csv', 'past_forecast_master.csv','sales_master.csv', 'transition_master.csv']            
    if all(x in blobfile for x in my_list):
        print ("The lists are identical")
    else :
        10/0
except:
    error = f'error in master file names. File names should be {my_list}'
    error_log = np.append(error_log,error)   

# COMMAND ----------

# DBTITLE 1,Sales_master column name check
try:
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)
    df_col = [col for col in df.columns if not col.startswith('20')]
    my_col=['Region Description', 'Region', 'Cluster', 'Cluster Description', 'Country', 'Country Description', 'Planning Hierarchy 7', 'Planning Hierarchy 7 Description', 'Planning Hierarchy 6', 'Planning Hierarchy 6 Description', 'Planning Hierarchy 5', 'Planning Hierarchy 5 Description', 'Planning Hierarchy 4', 'Planning Hierarchy 4 Description', 'Planning Hierarchy 3', 'Planning Hierarchy 3 Description', 'Planning Hierarchy 2', 'Planning Hierarchy 2 Description', 'Demand Stream']
    if len(df_col) == len(my_col) and collections.Counter(df_col) == collections.Counter(my_col):
        print ("The Column_names are correct")
    else :
        10/0
except:
    error = f'Column_names are incorrect in sales_master. Column names should be {my_col}. NOTE : No leading or trailing spaces in the column names.'
    error_log = np.append(error_log,error)

# COMMAND ----------

# DBTITLE 1,past_forecast_master column name check
try:
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/past_forecast_master.csv",inferSchema=True,header=True)
    df_col =df.columns
    my_col=['org_unit_id','channel_id', 'product_id', 'period', 'bf_m00', 'bf_m01', 'bf_m02', 'bf_m03', 'tf_m00', 'tf_m01', 'tf_m02', 'tf_m03']
    if len(df_col) == len(my_col) and collections.Counter(df_col) == collections.Counter(my_col):
        print ("The Column_names are correct")
    else :
        10/0
except:
    error = f'Column_names are incorrect in past_forecast_master. Column names should be {my_col}. NOTE : No leading or trailing spaces in the column names.'
    error_log = np.append(error_log,error)        

# COMMAND ----------

# DBTITLE 1,nts_master column name check
try:
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/nts_master.csv",inferSchema=True,header=True)
    df_col =df.columns
    my_col=['org_unit_id', 'channel_id', 'product_id', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']
    if len(df_col) == len(my_col) and collections.Counter(df_col) == collections.Counter(my_col):
        print ("The Column_names are correct")
    else :
        10/0
except:
    error = f'Column_names are incorrect in nts_master. Column names should be {my_col}. NOTE : No leading or trailing spaces in the column names.'
    error_log = np.append(error_log,error)

# COMMAND ----------

# DBTITLE 1,future_forecast_master column name check
try:
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/future_forecast_master.csv",inferSchema=True,header=True)
    df_col = [col for col in df.columns if not col.startswith('20')]
    my_col=['org_unit_id', 'channel_id', 'product_id', 'forecast_type']
    if len(df_col) == len(my_col) and collections.Counter(df_col) == collections.Counter(my_col):
        print ("The Column_names are correct")
    else :
        10/0
except:
    error = f'Column_names are incorrect in future_forecast_master. Column names should be {my_col}. NOTE : No leading or trailing spaces in the column names.'
    error_log = np.append(error_log,error)    

# COMMAND ----------

# DBTITLE 1,data_correction column name check
try:
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/data_correction.csv",inferSchema=True,header=True)
    df_col =df.columns
    my_col=['org_unit_id', 'channel_id', 'product_id', 'period', 'historical_sale', 'action', 'to_period']
    if len(df_col) == len(my_col) and collections.Counter(df_col) == collections.Counter(my_col):
        print ("The Column_names are correct")
    else :
        10/0
except:
    error = f'Column_names are incorrect in data_correction. Column names should be {my_col}. NOTE : No leading or trailing spaces in the column names.'
    error_log = np.append(error_log,error)          

# COMMAND ----------

# DBTITLE 1,transition_master column name check
try:
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/transition_master.csv",inferSchema=True,header=True)
    df_col =df.columns
    my_col=['org_unit_id', 'channel_id', 'old_sku', 'new_sku']
    if len(df_col) == len(my_col) and collections.Counter(df_col) == collections.Counter(my_col):
        print ("The Column_names are correct")
    else :
        10/0
except:
    error = f'Column_names are incorrect in transition_master. Column names should be {my_col}. NOTE : No leading or trailing spaces in the column names.'
    error_log = np.append(error_log,error)         

# COMMAND ----------

# DBTITLE 1,npf_master column name and status check
try:
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/npf_master.csv",inferSchema=True,header=True)
    df_col =df.columns
    my_col=['org_unit_id', 'channel_id', 'product_id', 'status']
    if len(df_col) == len(my_col) and collections.Counter(df_col) == collections.Counter(my_col):
        print ("The Column_names are correct")
        try:
            df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/npf_master.csv",inferSchema=True,header=True)
            pandasDF2 = df.toPandas()
            unique_keys =pandasDF2.groupby(['org_unit_id', 'channel_id', 'product_id'])['status'].unique()
            df1 = pd.DataFrame(unique_keys)
            df1['statusLength'] = df1.status.map(len)
            unexpected_data = df1[df1['statusLength'] > 1]
            if unexpected_data.shape[0]==0:
                print("Data is Proper")
            else:
                column = unexpected_data.loc[:, 0]
                10/0
        except: 
            error = f'npf_master has multiple status {column}'
            error_log = np.append(error_log,error)        
    else :
        10/0
except:
    error = f'Column_names are incorrect in npf_master. Column names should be {my_col}. NOTE : No leading or trailing spaces in the column names.'
    error_log = np.append(error_log,error)        

# COMMAND ----------

# DBTITLE 1,future_discontinue_master column name check
try:
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/future_discontinue_master.csv",inferSchema=True,header=True)
    df_col =df.columns
    my_col=['org_unit_id', 'channel_id', 'product_id', 'period']
    if len(df_col) == len(my_col) and collections.Counter(df_col) == collections.Counter(my_col):
        print ("The Column_names are correct")
    else :
        10/0
except:
    error = f'Column_names are incorrect in future_discontinue_master. Column names should be {my_col}. NOTE : No leading or trailing spaces in the column names.'
    error_log = np.append(error_log,error)        

# COMMAND ----------

# DBTITLE 1,Future_Forecast_Master [Period]
try:
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/future_forecast_master.csv",inferSchema=True,header=True)
    new_list = [col for col in df.columns if col.startswith('20')]
    a=[]
    for elements in new_list:
        a=len(elements)
        if a==6:
            print('Format is correct')
        else :
            10/0
except:
    error = 'period format is incorrect in future_forecast_master file'
    error_log = np.append(error_log,error)       

# COMMAND ----------

# DBTITLE 1,Period format in sales_master
try:
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)
    new_list = [col for col in df.columns if col.startswith('20')]
    a=[]
    for elements in new_list:
        a=len(elements)
        if a==6:
            print('Format is correct')
        else :
            10/0
except:
    error = 'period format is incorrect in sales_master file'
    error_log = np.append(error_log,error)  

# COMMAND ----------

# DBTITLE 1,Period format in past_forecast_master
try:
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/past_forecast_master.csv",inferSchema=True,header=True)
    new_period=df.select("period")
    a=[]
    for elements in new_list:
        a=len(elements)
        if a==6:
            print('Format is correct')
        else :
            10/0
except:
    error = 'period format is incorrect in past_forecast_master'
    error_log = np.append(error_log,error)   

# COMMAND ----------

# DBTITLE 1,Checking Non Revenue and Revenue in masters
# try:    
#     masters_names = ('sales_master','past_forecast_master','nts_master','npf_master','future_forecast_master','future_discontinue_master','data_correction','transition_master')
#     for names in masters_names:    
#         df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/{names}.csv",inferSchema=True,header=True)
#         if 'Demand Stream' in df.columns:
#             df_new=df.select("Demand Stream")
#             df_new = df_new.select("Demand Stream").rdd.flatMap(lambda x: x).collect()
#         elif 'channel_id' in df.columns:
#             df_new=df.select("channel_id")
#             df_new = df_new.select("channel_id").rdd.flatMap(lambda x: x).collect()
#         else:
#             print('Format is correct')
#         for i in df_new:
#             if domain_id.startswith('NA'):               
#                 if ((i=='Non Revenue') or (i=='Revenue') or (i == 'Grant')):
#                     print('Format is correct')
#                 else:
#                     10/0
#             elif not domain_id.startswith('NA'):
#                 if ((i=='Non Revenue') or (i=='Revenue')):
#                     print('Format is correct')
#                 else:
#                     10/0
# except:
#     error = f'Non Revenue and Revenue is incorrect in {names}'
#     error_log = np.append(error_log,error)  

# COMMAND ----------

# DBTITLE 1,Find Discrepancies in sales_master
try:    
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)
    pandasDF2 = df.toPandas()
    sales_master_cols = ('Planning Hierarchy 2', 'Planning Hierarchy 3', 'Planning Hierarchy 4',
    'Planning Hierarchy 5', 'Planning Hierarchy 6', 'Planning Hierarchy 7')
    for i, colname in enumerate(sales_master_cols):
        if i == len(sales_master_cols) - 1:
            break
        else:
            unique_keys =pandasDF2.groupby([colname])[sales_master_cols[i+1]].unique()
            df1 = pd.DataFrame(unique_keys)
            df1['statusLength'] = df1[sales_master_cols[i+1]].map(len)
            unexpected_data = df1[df1['statusLength'] > 1]
            if unexpected_data.shape[0]==0:
                print("Data is correct")
            else:
                unexpected_data_dict = unexpected_data.to_dict()
                unexpected_data_dict = [*unexpected_data_dict.values()]
                unexpected_data_dict = pd.DataFrame(unexpected_data_dict)
                unexpected_data_dict = dict(unexpected_data_dict)
                keys = [*unexpected_data_dict.keys()]
                10/0
except:    
    error = f'The following values at {colname} has multiple parents : {keys}'
    error_log = np.append(error_log,error)

# COMMAND ----------

# DBTITLE 1,sales_master description check
try:    
    df = spark.read.csv(f"/mnt/customerdata/{env}/Full_Load/{domain_id}/sales_master.csv",inferSchema=True,header=True)
    pandasDF2 = df.toPandas()
    sales_master_cols = [('Planning Hierarchy 2','Planning Hierarchy 2 Description'),('Planning Hierarchy 3','Planning Hierarchy 3 Description'),('Planning Hierarchy 4','Planning Hierarchy 4 Description'),('Planning Hierarchy 5','Planning Hierarchy 5 Description'),('Planning Hierarchy 6','Planning Hierarchy 6 Description'), ('Planning Hierarchy 7','Planning Hierarchy 7 Description'),('Region','Region Description'),('Cluster','Cluster Description'),('Country','Country Description')]

    for i in range(len(sales_master_cols)):

        unique_keys =pandasDF2.groupby([sales_master_cols[i][0]])[sales_master_cols[i][1]].unique()
        df1 = pd.DataFrame(unique_keys)
        df1['statusLength'] = df1[sales_master_cols[i][1]].map(len)
        unexpected_data = df1[df1['statusLength'] > 1]
        if unexpected_data.shape[0]==0:
            print("Data is correct")
        else:
            column = unexpected_data.columns.values[0]
            unexpected_data_dict = unexpected_data.to_dict()
            unexpected_data_dict = [*unexpected_data_dict.values()]
            unexpected_data_dict = pd.DataFrame(unexpected_data_dict)
            unexpected_data_dict = dict(unexpected_data_dict)
            keys = [*unexpected_data_dict.keys()]
            10/0
except:
    error = f'sales_master has multiple descriptions for {column} in {keys}'
    error_log = np.append(error_log,error)

# COMMAND ----------

if len(error_log) > 0:    
    error_log = " , ".join(error_log)
    return_json = json.dumps(error_log)    
    dbutils.notebook.exit(return_json)
else:
    dbutils.notebook.exit("N")
