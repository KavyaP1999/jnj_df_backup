# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, lower, col, split
from pyspark.sql import functions as f
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import traceback

# COMMAND ----------

ingestion_flag = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ####Get run_id and domain_id 

# COMMAND ----------

dbutils.widgets.text("run_id", "","")
run_id = dbutils.widgets.get("run_id")
print ("run_id:",run_id)

dbutils.widgets.text("domain_id", "","")
domain_id = dbutils.widgets.get("domain_id")
print ("domain_id:",domain_id)

ref_master_flag = dbutils.widgets.get("ref_master_flag")
ref_master_flag = int(ref_master_flag)
print ("ref_master_flag:",ref_master_flag)

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

# MAGIC %run /Shared/Prod/configuration_prodV2

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

# new code for NPF master
# transitioned_nts_master = spark.read.csv(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_nts_master_{domain_id}.csv",inferSchema=True,header=True)
# #npf_master = spark.read.jdbc(url=url,table= f"(SELECT * FROM npf_master WHERE domain_id = '{domain_id}') as npf_master", properties=properties)
# npf_master = spark.read.csv(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_npf_master_{domain_id}.csv",inferSchema=True,header=True)
# #Read the heirarchy master for reverse hierarchy
# d_file = spark.read.jdbc(url=url, table= f"(SELECT * FROM hierarchy WHERE domain_id = '{domain_id}') as d_file", properties = properties)
# d_file = d_file.toPandas()

# COMMAND ----------

# Old code reverted for NPF
transitioned_nts_master = spark.read.csv(f"/mnt/disk/staging_data/{env}/{domain_id}/transitioned_raw_nts_master_{domain_id}.csv",inferSchema=True,header=True)
npf_master = spark.read.jdbc(url=url,table= f"(SELECT * FROM npf_master WHERE domain_id = '{domain_id}') as npf_master", properties=properties)
#Read the heirarchy master for reverse hierarchy
d_file = spark.read.jdbc(url=url, table= f"(SELECT * FROM hierarchy WHERE domain_id = '{domain_id}') as d_file", properties = properties)
d_file = d_file.toPandas()

# COMMAND ----------

npf_master = npf_master.withColumn('keys', concat_ws('@', npf_master.domain_id, npf_master.org_unit_id, npf_master.channel_id, npf_master.product_id))
npf_master = npf_master.select([lower('status').alias('status'), 'keys'])
npf_master.createOrReplaceTempView(f'npf_master_run{run_id}')
transitioned_nts_master.createOrReplaceTempView(f'transitioned_nts_master_{run_id}')
final_data = spark.sql(f'select * from npf_master_run{run_id} npf inner join transitioned_nts_master_{run_id} t on npf.keys=t.key')
final_data = final_data.filter((final_data.status=='active'))
final_data = final_data.drop('keys', 'status')
final_data = final_data.withColumn("org_unit_id", split(col("key"), "@").getItem(1)).withColumn("channel_id", split(col("key"), "@").getItem(2)).withColumn("product_id", split(col("key"), "@").getItem(3))
final_data = final_data.drop("key")

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

try:
    if ref_master_flag == 1 :
        #Reading reference master 
        reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{domain_id}') as reference_master",properties = properties)
        #print("The count of reference_master",reference_master.count())
        reference_master.createOrReplaceTempView(f'reference_master_{run_id}')
        ref_count = spark.sql(f'select count(*) from reference_master_{run_id}').rdd.map(lambda r:r[0]).collect()[0]
        print("The count of reference_master",ref_count)
        if ref_count == 0:
            raise Exception

        p_file = d_file[d_file['hierarchy_type'] == 'product'].copy()
        c_file = d_file[d_file['hierarchy_type'] == 'channel'].copy()
        o_file = d_file[d_file['hierarchy_type'] == 'org_unit'].copy()

        p_file = generate_hierarchy(p_file, 'product')
        c_file = generate_hierarchy(c_file, 'channel')
        o_file = generate_hierarchy(o_file, 'org_unit')

        p_file = spark.createDataFrame(p_file)
        c_file = spark.createDataFrame(c_file)
        o_file = spark.createDataFrame(o_file)
#filtered_nts_master---->final_data
        #Reference master filtering
#         hierarchy_master = p_file.filter(p_file[f'product_{product_forecast_level}'].isin(reference_master.select('product_id').distinct().rdd.map(lambda r:r[0]).collect()))
#         final_data = final_data.filter(final_data['product_id'].isin(hierarchy_master.select(f'product_{product_input_level}').distinct().rdd.map(lambda r:r[0]).collect()))
#         hierarchy_master = c_file.filter(c_file[f'channel_{channel_forecast_level}'].isin(reference_master.select('channel_id').distinct().rdd.map(lambda r:r[0]).collect()))
#         final_data = final_data.filter(final_data['channel_id'].isin(hierarchy_master.select(f'channel_{channel_input_level}').distinct().rdd.map(lambda r:r[0]).collect()))
#         hierarchy_master = o_file.filter(o_file[f'org_unit_{org_unit_forecast_level}'].isin(reference_master.select('org_unit_id').distinct().rdd.map(lambda r:r[0]).collect()))
#         final_data = final_data.filter(final_data['org_unit_id'].isin(hierarchy_master.select(f'org_unit_{org_unit_input_level}').distinct().rdd.map(lambda r:r[0]).collect()))
        
        p_file.createOrReplaceTempView(f'p_file_{run_id}')
        reference_master.createOrReplaceTempView(f'reference_master_{run_id}')
        hierarchy_master = spark.sql(f'select * from p_file_{run_id} p inner join reference_master_{run_id} rf on p.product_{product_forecast_level}=rf.product_id')
        hierarchy_master = hierarchy_master.drop('org_unit_id','channel_id','product_id','planner','domain_id','run_id')
        hierarchy_master = hierarchy_master.drop_duplicates()
        final_data.createOrReplaceTempView(f'final_data_{run_id}')
        hierarchy_master.createOrReplaceTempView(f'hierarchy_master_{run_id}')
        final_data = spark.sql(f'select * from final_data_{run_id} f inner join hierarchy_master_{run_id} h on f.product_id=h.product_{product_input_level}')
        final_data = final_data.select('domain_id','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','org_unit_id','channel_id','product_id')
        final_data = final_data.drop_duplicates()
        c_file.createOrReplaceTempView(f'c_file_{run_id}')
        hierarchy_master = spark.sql(f'select * from c_file_{run_id} c inner join reference_master_{run_id} rf on c.channel_{channel_forecast_level}=rf.channel_id')
        hierarchy_master = hierarchy_master.drop('org_unit_id','channel_id','product_id','planner','domain_id','run_id')
        hierarchy_master = hierarchy_master.drop_duplicates()
        hierarchy_master.createOrReplaceTempView(f'hierarchy_master_{run_id}')
        final_data.createOrReplaceTempView(f'final_data_{run_id}')
        final_data = spark.sql(f'select * from final_data_{run_id} f inner join hierarchy_master_{run_id} h on f.channel_id=h.channel_{channel_input_level}')
        final_data = final_data.select('domain_id','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','org_unit_id','channel_id','product_id')
        final_data = final_data.drop_duplicates()
        o_file.createOrReplaceTempView(f'o_file_{run_id}')
        hierarchy_master = spark.sql(f'select * from o_file_{run_id} o inner join reference_master_{run_id} rf on o.org_unit_{org_unit_forecast_level}=rf.org_unit_id')
        hierarchy_master = hierarchy_master.drop('org_unit_id','channel_id','product_id','planner','domain_id','run_id')
        hierarchy_master = hierarchy_master.drop_duplicates()
        hierarchy_master.createOrReplaceTempView(f'hierarchy_master_{run_id}')
        final_data.createOrReplaceTempView(f'final_data_{run_id}')
        final_data = spark.sql(f'select * from final_data_{run_id} f inner join hierarchy_master_{run_id} h on f.org_unit_id=h.org_unit_{org_unit_input_level}')
        final_data = final_data.select('domain_id','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','org_unit_id','channel_id','product_id')
        final_data = final_data.drop_duplicates()
except:
    
    traceback.print_exc()
    10/0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregation

# COMMAND ----------

def product_aggregator(df, Heirarchy, product_input_level, product_output_level):
    suffix = 'product'

    if 'product_id' not in df.columns:
        df = df.withColumn('org_unit_id',split(df['key'],'@').getItem(0)) \
        .withColumn('channel_id',split(df['key'],'@').getItem(1)) \
        .withColumn('product_id',split(df['key'],'@').getItem(2))
    hm = Heirarchy.select(col(f'product_{product_input_level}'),col(f'product_{product_output_level}'))

    df = df.join(hm, df.product_id == hm[f'{suffix}_{int(product_input_level)}'],how='left')
    df = df.drop('product_id',f'{suffix}_{int(product_input_level)}')
    df = df.withColumnRenamed(f'{suffix}_{int(product_output_level)}','product_id')
    df = df.withColumn('m1',col('m1').cast(FloatType())).withColumn('m2',col('m2').cast(FloatType())).withColumn('m3',col('m3').cast(FloatType())).withColumn('m4',col('m4').cast(FloatType())).withColumn('m5',col('m5').cast(FloatType())).withColumn('m6',col('m6').cast(FloatType())).withColumn('m7',col('m7').cast(FloatType())).withColumn('m8',col('m8').cast(FloatType())).withColumn('m9',col('m9').cast(FloatType())).withColumn('m10',col('m10').cast(FloatType())).withColumn('m11',col('m11').cast(FloatType())).withColumn('m12',col('m12').cast(FloatType()))
    
    df = df.groupby('domain_id', 'org_unit_id','channel_id','product_id').agg(f.sum('m1').alias('m1'),f.sum('m2').alias('m2'),f.sum('m3').alias('m3'),f.sum('m4').alias('m4')
                                                                         ,f.sum('m5').alias('m5'),f.sum('m6').alias('m6'),f.sum('m7').alias('m7'),f.sum('m8').alias('m8')                                                                           ,f.sum('m9').alias('m9'),f.sum('m10').alias('m10'),f.sum('m11').alias('m11'),f.sum('m12').alias('m12'))
    return df

# COMMAND ----------

def channel_aggregator(df, Heirarchy, channel_input_level, channel_output_level):
    suffix = 'channel'

    if 'channel_id' not in df.columns:
        df[["org_unit_id", "channel_id", "product_id"]] = df.key.str.split("@", expand=True)
          
    hm = Heirarchy.select(col(f'channel_{channel_input_level}'),col(f'channel_{channel_output_level}'))
    df = df.join(hm, df.channel_id == hm[f'{suffix}_{int(channel_input_level)}'],how='left')
    df = df.drop('channel_id',f'{suffix}_{int(channel_input_level)}')
    df = df.withColumnRenamed(f'{suffix}_{int(channel_output_level)}','channel_id')
    df = df.withColumn('m1',col('m1').cast(FloatType())).withColumn('m2',col('m2').cast(FloatType())).withColumn('m3',col('m3').cast(FloatType())).withColumn('m4',col('m4').cast(FloatType())).withColumn('m5',col('m5').cast(FloatType())).withColumn('m6',col('m6').cast(FloatType())).withColumn('m7',col('m7').cast(FloatType())).withColumn('m8',col('m8').cast(FloatType())).withColumn('m9',col('m9').cast(FloatType())).withColumn('m10',col('m10').cast(FloatType())).withColumn('m11',col('m11').cast(FloatType())).withColumn('m12',col('m12').cast(FloatType()))
    
    df = df.groupby('domain_id', 'org_unit_id','channel_id','product_id').agg(f.sum('m1').alias('m1'),f.sum('m2').alias('m2'),f.sum('m3').alias('m3'),f.sum('m4').alias('m4')
                                                                         ,f.sum('m5').alias('m5'),f.sum('m6').alias('m6'),f.sum('m7').alias('m7'),f.sum('m8').alias('m8')                                                                           ,f.sum('m9').alias('m9'),f.sum('m10').alias('m10'),f.sum('m11').alias('m11'),f.sum('m12').alias('m12'))
    return df

# COMMAND ----------

def org_unit_aggregator(df, Heirarchy, org_unit_input_level, org_unit_output_level):
    
    suffix = 'org_unit'

    if 'org_unit_id' not in df.columns:
        df[["org_unit_id", "channel_id", "product_id"]] = df.key.str.split("@", expand=True)
        
    hm = Heirarchy.select(col(f'org_unit_{org_unit_input_level}'),col(f'org_unit_{org_unit_output_level}'))
    df = df.join(hm, df.org_unit_id == hm[f'{suffix}_{int(org_unit_input_level)}'],how='left')
    df = df.drop('org_unit_id',f'{suffix}_{int(org_unit_input_level)}')
    df = df.withColumnRenamed(f'{suffix}_{int(org_unit_output_level)}','org_unit_id')
    df = df.withColumn('m1',col('m1').cast(FloatType())).withColumn('m2',col('m2').cast(FloatType())).withColumn('m3',col('m3').cast(FloatType())).withColumn('m4',col('m4').cast(FloatType())).withColumn('m5',col('m5').cast(FloatType())).withColumn('m6',col('m6').cast(FloatType())).withColumn('m7',col('m7').cast(FloatType())).withColumn('m8',col('m8').cast(FloatType())).withColumn('m9',col('m9').cast(FloatType())).withColumn('m10',col('m10').cast(FloatType())).withColumn('m11',col('m11').cast(FloatType())).withColumn('m12',col('m12').cast(FloatType()))
    
    df = df.groupby('domain_id', 'org_unit_id','channel_id','product_id').agg(f.sum('m1').alias('m1'),f.sum('m2').alias('m2'),f.sum('m3').alias('m3'),f.sum('m4').alias('m4')
                                                                         ,f.sum('m5').alias('m5'),f.sum('m6').alias('m6'),f.sum('m7').alias('m7'),f.sum('m8').alias('m8')                                                                           ,f.sum('m9').alias('m9'),f.sum('m10').alias('m10'),f.sum('m11').alias('m11'),f.sum('m12').alias('m12'))
    

    return df

# COMMAND ----------

def nts_master_agg_function(df, Heirarchy, product_output_level, channel_output_level, org_unit_output_level, product_input_level,
                         channel_input_level, org_unit_input_level):
    df = df.withColumn('product_id',col('product_id').cast(StringType()))

    if product_input_level != product_output_level:
        level_difference = product_output_level - product_input_level
        if level_difference > 0:
            p_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'product'], 'product')
            p_file = spark.createDataFrame(p_file)
            df = product_aggregator(df, p_file, product_input_level, product_output_level)
        elif level_difference < 0:
            print("Product Level aggregation is not required")

    if channel_input_level != channel_output_level:
        level_difference = channel_output_level - channel_input_level
        if level_difference > 0:
            c_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'channel'], 'channel')
            c_file = spark.createDataFrame(c_file)
            df = channel_aggregator(df, c_file, channel_input_level, channel_output_level)
        elif level_difference < 0:
            print("Channel level aggregation is not required")

    if org_unit_input_level != org_unit_output_level:
        level_difference = org_unit_output_level - org_unit_input_level
        if level_difference > 0:
            o_file = generate_hierarchy(Heirarchy[Heirarchy['hierarchy_type'] == 'org_unit'], 'org_unit')
            o_file = spark.createDataFrame(o_file)
            df = org_unit_aggregator(df, o_file, org_unit_input_level, org_unit_output_level)
        elif level_difference < 0:
            print("Org_Unit level aggregation is not required")
    df = df.withColumn('key',concat_ws('@',col('org_unit_id'),col('channel_id'),col('product_id')))
    df = df.select('key', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'domain_id')
    df = df.select('key','domain_id',f.round('m1',0).alias('m1'),f.round('m2',0).alias('m2'),f.round('m3',0).alias('m3'),f.round('m4',0).alias('m4'),
                  f.round('m5',0).alias('m5'),f.round('m6',0).alias('m6'),f.round('m7',0).alias('m7'),f.round('m8',0).alias('m8'),
                  f.round('m9',0).alias('m9'),f.round('m10',0).alias('m10'),f.round('m11',0).alias('m11'),f.round('m12',0).alias('m12'))
    df = df.drop_duplicates()
    
    return df

# COMMAND ----------

nts_master1 = nts_master_agg_function(final_data, d_file, product_output_level=product_forecast_level, channel_output_level=channel_forecast_level,
                                      org_unit_output_level=org_unit_forecast_level, product_input_level=product_input_level,
                                      channel_input_level=channel_input_level,
                                      org_unit_input_level=org_unit_input_level)

nts_master1 = nts_master1.withColumn('run_id',f.lit(run_id))

# COMMAND ----------

if ref_master_flag == 1 :
    reference_master = spark.read.jdbc(url=url, table= f"(SELECT * FROM reference_master WHERE run_id = '{run_id}' and domain_id = '{domain_id}') as   reference_master",properties = properties)
    reference_master = reference_master.withColumn('key',concat_ws('@',col('org_unit_id'),col('channel_id'),col('product_id')))
    nts_master1 = nts_master1.filter(nts_master1['key'].isin(reference_master.select('key').distinct().rdd.map(lambda r: r[0]).collect()))

# COMMAND ----------

if ingestion_flag ==1:
    db_ingestion(nts_master1, 'nts_master', 'append')
else:
    print("Ingestion not done")
