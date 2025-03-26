# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
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

def run_parameter_data(run_id,url,properties):
    
    parameter = spark.read.jdbc(url=url,table=f"(select parameter_id,run_value from run_parameter where run_id = '{run_id}') as run_parameter" ,properties=properties)
    parameter = parameter.toPandas()
   
    parameter = dict(zip(parameter['parameter_id'],parameter['run_value']))
    
    return parameter

# COMMAND ----------

def fetch_f3_data(run_id: str,domain_id):
    
    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F3_run_data.parquet"
    f3_data = spark.read.parquet(working_path, inferSchema = True, header= True)
    return f3_data

# COMMAND ----------

def df_from_database(url,properties):
    
    df = spark.read.jdbc(url=url, table= f"(SELECT domain_id,hierarchy_type,hierarchy_value,parent_value,level_id,description FROM hierarchy where domain_id = '{domain_id}') as df" , properties = properties)
    df = df.toPandas()
    return df

# COMMAND ----------

# def generate_hierarchy(data, suffix='product'):
#     data = data[['hierarchy_value', 'parent_value', 'level_id']].copy()
#     split_df = {}
#     for i in np.unique(data.level_id):
#         split_df[i] = data[data['level_id'] == i].copy()
    
#     df = pd.DataFrame()
#     df = split_df[list(split_df.keys())[-1]][['hierarchy_value']].copy()
#     df.rename(columns={'hierarchy_value': f'{suffix}_{list(split_df.keys())[-1]}'}, inplace=True)
    
#     for i in list(split_df.keys())[-2::-1]:
#         data = split_df[i][['hierarchy_value', 'parent_value']].copy()
#         data.rename(columns={'hierarchy_value': f'{suffix}_{i}'}, inplace=True)
#         df = pd.merge(df, data, left_on=f'{suffix}_{int(i)+1}', right_on='parent_value', how='inner')
#         df.drop(columns={'parent_value'}, inplace=True)
#     df.drop_duplicates(inplace=True)
#     return df

def generate_hierarchy(data, parameters_values,suffix='product'):
    forecast_level_product = int(parameters_values[18])
    data = data[['hierarchy_value', 'parent_value', 'level_id', 'description']].copy()
    split_df = {}
    for i in np.unique(data.level_id):
        split_df[i] = data[data['level_id'] == i].copy()
    
    df = pd.DataFrame()
    df = split_df[list(split_df.keys())[-1]][['hierarchy_value','description']].copy()
    df.rename(columns={'hierarchy_value': f'{suffix}_{list(split_df.keys())[-1]}', 'description': f'{suffix}_{list(split_df.keys())[-1]}_desc'}, inplace=True)

    for i in list(split_df.keys())[-2::-1]:
        data = split_df[i][['hierarchy_value', 'parent_value', 'description']].copy()
        data.rename(columns={'hierarchy_value': f'{suffix}_{i}', 'description': f'{suffix}_{i}_desc'}, inplace=True)
        df = pd.merge(df, data, left_on=f'{suffix}_{int(i)+1}', right_on='parent_value', how='inner')
        if i < forecast_level_product:
            df[f'{suffix}_{i}'] = ''
            df[f'{suffix}_{i}_desc'] = ''
        df.drop(columns={'parent_value'}, inplace=True)
    return df


# COMMAND ----------

def fetch_ref_master(run_id: str,url,properties):
    
    df = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}') as df", properties = properties)
   
    return df

# COMMAND ----------

def get_run_user(run_id:str,url,properties):
    
    run_id = str(run_id)
    run = spark.read.jdbc(url=url, table= f"(SELECT * FROM run WHERE run_id = '{run_id}') as run", properties = properties)
    run.createOrReplaceTempView(f'run_{run_id}')
    user = spark.read.jdbc(url=url, table= f"(SELECT * FROM [user]) as u", properties = properties)
    user.createOrReplaceTempView(f'user_{run_id}')
    user_name = spark.sql(f'select user_name from user_{run_id} u join run_{run_id} r on r.user_id=u.id')
    user_name = user_name.select('user_name').distinct().rdd.map(lambda x : x[0]).collect()[0]
    return user_name

# COMMAND ----------


def brp_product_master(run_id: str, parameters_values,url,properties):
    
    forecast_level_product = int(parameters_values[18])

    #get F3 data 
    f3_df = fetch_f3_data(run_id,domain_id)
    master1 = f3_df.select('run_id','domain_id', 'org_unit_id', 'channel_id', 'product_id')
    master1 = master1.drop_duplicates()
    master1 = master1.withColumn('key',concat_ws('@',master1.org_unit_id,master1.channel_id,master1.product_id))
    master1 = master1.withColumnRenamed('org_unit_id','affiliate')

    #get hierarchy file
    h_file = df_from_database(url,properties)
    h_file_desc = h_file.copy()

    h_file = h_file[h_file['hierarchy_type'] == 'product']

    master2 = generate_hierarchy(h_file,parameters_values, suffix='product')
    print(master2.columns)
#     master3 = spark.createDataFrame(master2)
    h_file = spark.createDataFrame(h_file)
    
   
    ### get descriptions from hierarchy file
#     for i in master2.columns:
        
#         if int(i.split('_')[1]) < forecast_level_product:
            
#             master2 = master2.withColumn(i,lit(''))
#         h_file = h_file.select('hierarchy_value', 'description')
#         h_file.createOrReplaceTempView(f'h_file_{run_id}')
#         master2.createOrReplaceTempView(f'master2_{run_id}')
#         result_data = spark.sql(f'select * from h_file_{run_id} h right join master2_{run_id} m on h.hierarchy_value=m.{i}')
#         result_data = result_data.withColumnRenamed('description',f'{i}_desc')
#         result_data = result_data.drop('hierarchy_value')
#         master2 = result_data.select('*')
    result_data = spark.createDataFrame(master2)

    result_data = result_data.withColumnRenamed(f'product_{forecast_level_product}','product_id')
    master1.createOrReplaceTempView(f'master1_{run_id}')
    result_data = result_data.select('product_id','product_0_desc','product_1_desc', 'product_2_desc', 'product_3_desc', 'product_4_desc',
        'product_5_desc')
    result_data = result_data.withColumnRenamed('product_id','product_id_result')
    result_data.createOrReplaceTempView(f'result_data_{run_id}')
    final_masterdata = spark.sql(f'select * from master1_{run_id} m1 left join result_data_{run_id} rd on m1.product_id=rd.product_id_result')
   
    final_masterdata = final_masterdata.drop_duplicates()
    final_masterdata = final_masterdata.drop('product_id_result')
    

    #For channel desc
    h_file_desc = spark.createDataFrame(h_file_desc)
    h_file_desc_1 = h_file_desc.select('hierarchy_value', 'description')
    h_file_desc_1.createOrReplaceTempView(f'h_file_desc_{run_id}')
    final_masterdata.createOrReplaceTempView(f'final_masterdata_{run_id}')
    final_masterdata = spark.sql(f'select * from h_file_desc_{run_id} hf right join final_masterdata_{run_id} fm on hf.hierarchy_value=fm.channel_id')
    final_masterdata = final_masterdata.withColumnRenamed('description','channel')
    final_masterdata = final_masterdata.drop('hierarchy_value','channel_id')


    #For affiliate desc
    forecast_level_org_unit = int(parameters_values[16])
    h_file_desc_aff = h_file_desc.filter(h_file_desc['level_id']==forecast_level_org_unit)
    h_file_desc_aff = h_file_desc_aff.select('hierarchy_value','description')
    final_masterdata = final_masterdata.withColumn('affiliate',col('affiliate').cast(StringType()))
    h_file_desc_aff.createOrReplaceTempView(f'h_file_desc_aff_{run_id}')
    final_masterdata.createOrReplaceTempView(f'final_masterdata_{run_id}')
    final_masterdata = spark.sql(f'select * from h_file_desc_aff_{run_id} hfd right join final_masterdata_{run_id} fmd on hfd.hierarchy_value = fmd.affiliate')
    final_masterdata = final_masterdata.drop('hierarchy_value','affiliate')
    final_masterdata = final_masterdata.withColumnRenamed('description','Affiliate')
    #final_masterdata = final_masterdata.drop('key')
    final_masterdata = final_masterdata.withColumnRenamed('product_5_desc','p7').withColumnRenamed('product_4_desc','p6') \
                        .withColumnRenamed('product_3_desc','p5').withColumnRenamed('product_2_desc','p4') \
                        .withColumnRenamed('product_1_desc','p3').withColumnRenamed('product_0_desc','p2')
    final_masterdata = final_masterdata.withColumn('key1',concat_ws('@',final_masterdata.Affiliate,final_masterdata.channel,final_masterdata.product_id))
    
    final_masterdata = final_masterdata.drop_duplicates()
    try:
        ref_df = fetch_ref_master(run_id,url,properties)
    except:
        ref_df = pd.DataFrame()
    
    ref_df.createOrReplaceTempView(f'ref_df_{run_id}')
    ref_count = spark.sql(f'select count(*) from ref_df_{run_id}').rdd.map(lambda r:r[0]).collect()[0]
    if ref_count==0:
        user_name = get_run_user(run_id,url,properties)
        
        final_masterdata = final_masterdata.withColumn('planner',lit(user_name))
        final_masterdata = final_masterdata.drop('key1')
    else:
        
        ref_df = ref_df.withColumn('key2',concat_ws('@',ref_df.org_unit_id,ref_df.channel_id,ref_df.product_id))
        final_masterdata.createOrReplaceTempView(f'final_masterdata_{run_id}')
        ref_df = ref_df.select('key2','planner')
        ref_df.createOrReplaceTempView(f'ref_df_{run_id}')
        
        final_masterdata = spark.sql(f'select * from final_masterdata_{run_id} fi left join ref_df_{run_id} rfd on fi.key = rfd.key2')
        final_masterdata = final_masterdata.drop('key1','key2')
        
        final_masterdata = final_masterdata.select('run_id','domain_id','key', 'planner', 'Affiliate','p7','p6',
                                        'p5','p4','p3','p2', 'product_id','channel')
        final_masterdata = final_masterdata.drop_duplicates()
   

    return final_masterdata

# COMMAND ----------

#run_id = 416
#domain_id = 'APAC_TECA'

# COMMAND ----------

#parameter_values = run_parameter_data(run_id,url,properties)

#brp_product_master = brp_product_master(run_id, parameter_values,url,properties)

# COMMAND ----------

# brp_product_master.display()
