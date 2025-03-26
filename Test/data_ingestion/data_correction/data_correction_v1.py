# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import time
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import concat_ws,col

# COMMAND ----------

dbutils.widgets.text("domain_id", "","")
domain = dbutils.widgets.get("domain_id")
print ("domain_id:",domain)

# COMMAND ----------

spark = SparkSession.builder.appName("demo").getOrCreate()

# COMMAND ----------

# MAGIC %run /Shared/Test/configuration_test

# COMMAND ----------

properties = {
 "user" : userName,
 "password" : userPassword }
url = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname,jdbcPort,databaseName)

# COMMAND ----------

pd.set_option('display.max_columns', None)

# COMMAND ----------

def db_ingestion(df, table_name, mode):
    counts = df.count()
    paritions = counts // 1000000 + 1
    #default partition = 24
    df.repartition(paritions).write.jdbc(url=url, table=table_name, mode=mode, properties=properties)

# COMMAND ----------

raw_history_master = spark.read.csv(f"/mnt/disk/staging_data/test/{domain}/filtered_active_npi_raw_history_master_{domain}.csv",inferSchema=True,header=True)

# COMMAND ----------

raw_history_master_df = raw_history_master.select("*").toPandas()

# COMMAND ----------

data_correction = spark.read.csv(f"/mnt/disk/staging_data/test/{domain}/transitioned_data_correction_{domain}.csv",inferSchema=True,header=True)
data_correction_df = data_correction.toPandas()

# COMMAND ----------

def datacorrection(raw_history_master, new_data):
    raw_history_master['key'] = raw_history_master['domain_id'] + '~' + raw_history_master['org_unit_id'] + '~' + raw_history_master['channel_id'] + '~' + \
                     raw_history_master['product_id'] + '~' + raw_history_master['period']
    
    if len(new_data) > 0:
        
        new_data['period'] = new_data['period'].astype(str).apply(
            lambda x: (x[0:4] + 'M' + x[5:].zfill(3)))

        new_data['to_period'] = new_data['to_period'].apply(
            lambda x: np.nan if pd.isnull(x) else (x[0:4] + 'M' + x[5:].zfill(3)))

        new_data['action'] = new_data['action'].apply(
            lambda x: np.nan if pd.isnull(x) else (x.upper()))

        new_data['product_id'] = new_data['product_id'].astype(str)
        new_data['key'] = new_data['domain_id'] + '~' + new_data['org_unit_id'] + '~' + new_data['channel_id'] + '~' + \
                          new_data['product_id'] + '~' + new_data['period']

        new_data = new_data[new_data['key'].isin(raw_history_master['key'])].copy()
        new_data = new_data.merge(raw_history_master,on='key', suffixes=('', '_drop'))
        to_drop = [x for x in new_data if x.endswith('_drop')]
        new_data.drop(to_drop, axis=1, inplace=True)
        new_data.drop(columns=['run_state','segmentation','model_id','promotion','forecast','expost_sales'], axis=1, inplace=True)                        
        
        values_with_hist = new_data[new_data['historical_sale'] > 0]
        new_data = new_data[~new_data['key'].isin(values_with_hist.key)]
        values_with_period = new_data[new_data['to_period'].notnull()]
        new_data = new_data[~new_data['key'].isin(values_with_period.key)]
        values_with_UP = new_data[new_data['action'] == 'UP']
        values_with_Down = new_data[new_data['action'] == 'DOWN']
        
        # Processing Down data
        if (len(values_with_Down)) > 0:
            value_list = []
            for i in values_with_Down['key']:
                values_with_Down_temp = values_with_Down[values_with_Down['key'] == i]
                values_with_Down_temp['period_replace'] = values_with_Down_temp['period'].apply(lambda x: x[-2:])
                subset_down_data = raw_history_master[raw_history_master['key'].str.rsplit('~', n=1).str[0].isin(
                    values_with_Down_temp['key'].str.rsplit('~', n=1).str[0])]
#                 values_with_Down['cdh'] = raw_history_master[raw_history_master['key']==i]['cdh'].values[0]
                subset_down_data['period_replace'] = subset_down_data['period'].apply(lambda x: x[-2:])
                subset_down_data_temp = subset_down_data[(subset_down_data['period_replace'] ==
                                                          values_with_Down_temp['period_replace'].reset_index().iloc[
                                                              0, 1]) &
                                                         (subset_down_data['period'] <= i.split('~')[
                                                             4])].reset_index().sort_values(by=['key'], ascending=False)
                if subset_down_data_temp.shape[0] == 0:
                    continue
                max = subset_down_data_temp.reset_index(drop=True).loc[0, 'expost_sales']
                subset_down_data_temp = subset_down_data_temp[subset_down_data_temp['expost_sales'] < max]
                subset_down_data_temp = subset_down_data_temp[['expost_sales', 'key', 'period_replace', 'cdh']]. \
                    sort_values(by=['expost_sales'], ascending=False)
                if len(subset_down_data_temp) >= 1:
                    value = subset_down_data_temp.reset_index().iloc[0, 1]
                else:
                    value = max
                value_list.append(value)
            if len(value_list) == values_with_Down.shape[0]:
                values_with_Down["historical_sale"] = value_list
                

        # Processing UP data
        if (len(values_with_UP)) > 0:
            value_list = []
            for i in values_with_UP['key']:
                values_with_Up_temp = values_with_UP[values_with_UP['key'] == i]
                values_with_Up_temp['period_replace'] = values_with_Up_temp['period'].apply(lambda x: x[-2:])
                subset_up_data = raw_history_master[
                    raw_history_master['key'].str.rsplit('~', n=1).str[0].isin(
                        values_with_Up_temp['key'].str.rsplit('~', n=1).str[0])]
#                 values_with_UP['cdh'] = raw_history_master[raw_history_master['key']==i]['cdh'].values[0]
                subset_up_data['period_replace'] = subset_up_data['period'].apply(lambda x: x[-2:])

                subset_up_data_temp = subset_up_data[
                    (subset_up_data['period_replace'] == values_with_Up_temp['period_replace'].reset_index().iloc[
                        0, 1]) & (
                            subset_up_data['period'] <= i.split('~')[4])].reset_index().sort_values(by=['key'],
                                                                                                    ascending=False)
                if subset_up_data_temp.shape[0] == 0:
                    continue
                min = subset_up_data_temp['expost_sales'].reset_index(drop=True)[0]
                subset_up_data_temp = subset_up_data_temp[subset_up_data_temp['expost_sales'] > min]
                subset_up_data_temp = subset_up_data_temp[['expost_sales', 'key', 'period_replace']].sort_values(
                    by=['key'], ascending=False)
                if len(subset_up_data_temp) >= 1:
                    value = subset_up_data_temp.reset_index().iloc[0, 1]
                else:
                    value = min
                value_list.append(value)
            if len(value_list) == values_with_UP.shape[0]:
                values_with_UP["historical_sale"] = value_list

        # processing values_with_period
        if (len(values_with_period)) > 0:
            values_with_period_temp = values_with_period.copy()
            values_with_period_temp['new_key'] = values_with_period_temp['domain_id'] + '~' + values_with_period_temp[
                'org_unit_id'] + '~' + values_with_period['channel_id'] + '~' + \
                                                 values_with_period_temp['product_id'] + '~' + values_with_period_temp[
                                                     'to_period']
            temp_hist = []
            for i in values_with_period_temp.new_key:
                if i in raw_history_master['key'].to_list():                 
                    temp_hist.append(float(raw_history_master[raw_history_master['key'] == i]['expost_sales'].to_list()[0]))
                else:
                    temp_hist.append(np.nan)

            values_with_period['historical_sale'] = temp_hist
            values_with_period.dropna(subset=['historical_sale'], inplace=True)
        
        new_data = pd.concat([values_with_hist, values_with_Down, values_with_UP, values_with_period])
        new_data = new_data.rename(columns={'historical_sale':'expost_sales'})
       #         new_data['cdh'] = raw_history_master[raw_history_master['key'].isin(new_data['key'])]['cdh'].values

        new_data1 = new_data[['domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period',
                              'expost_sales', 'key']]

        raw_history_master1 = raw_history_master[['domain_id', 'org_unit_id', 'channel_id', 'product_id', 'period',
                            'expost_sales', 'key']]

        match = new_data1['key']
        length = len(raw_history_master[raw_history_master1['key'].isin(match)])
        if length > 0:
            df = raw_history_master[~raw_history_master1['key'].isin(match)]
            new_df = pd.concat([df, new_data])
            new_df['domain_id'] = raw_history_master['domain_id']

            new_df['run_state'] = 'I5'
            extacol = pd.DataFrame(columns=['forecast', 'segmentation', 'model_id', 'promotion'])
            final_df = pd.concat([new_df, extacol], axis=0)
            final_df = final_df.drop(['key', 'action', 'to_period'], axis=1)
            final_df['expost_sales'][final_df['expost_sales'] < 0] = 0


            final_df.fillna(value={'domain_id': domain}, inplace=True)

#             final_df1 = final_df.reset_index(drop=True)
#             final_df1['segmentation'] = ''
#             final_df1['model_id'] = ''
#             final_df1['promotion'] = ''

        else:
            raw_history_master['run_state'] = 'I5'
            final_df = raw_history_master.drop(['key'], axis=1)
            final_df['expost_sales'][final_df['expost_sales'] < 0] = 0



            final_df.fillna(value={'domain_id': domain}, inplace=True)
        return final_df



    else:

        raw_history_master['run_state'] = 'I5'
        raw_history_master = raw_history_master.drop(['key'], axis=1)
        raw_history_master['expost_sales'][raw_history_master['expost_sales'] < 0] = 0
        return raw_history_master

# COMMAND ----------

data = datacorrection(raw_history_master_df, data_correction_df)

# COMMAND ----------

raw_run_data = data.rename(columns={'expost_sales':'historical_sale'})

# COMMAND ----------

raw_run_data = raw_run_data.drop(columns=['cdh'], axis=1)

# COMMAND ----------

raw_history_master_schema = StructType([StructField('domain_id',StringType(),True),

 StructField('run_state',StringType(),True),

 StructField('org_unit_id',StringType(),True),

 StructField('channel_id',StringType(),True),

 StructField('product_id',StringType(),True),

 StructField('period',StringType(),True),

 StructField('segmentation',StringType(),True),

 StructField('model_id',StringType(),True),

 StructField('promotion',StringType(),True),

 StructField('cdh',FloatType(),True),

 StructField('expost_sales',FloatType(),True),

 StructField('forecast',StringType(),True)

 ])

# COMMAND ----------

raw_run_data_schema = StructType([StructField('domain_id',StringType(),True),

 StructField('run_state',StringType(),True),

 StructField('org_unit_id',StringType(),True),

 StructField('channel_id',StringType(),True),

 StructField('product_id',StringType(),True),

 StructField('period',StringType(),True),

 StructField('segmentation',StringType(),True),

 StructField('model_id',StringType(),True),

 StructField('promotion',StringType(),True),

 StructField('historical_sale',FloatType(),True),

 StructField('forecast',StringType(),True)

 ])

# COMMAND ----------

df1 = spark.createDataFrame(data=data, schema=raw_history_master_schema)

# COMMAND ----------

df2 = spark.createDataFrame(data=raw_run_data, schema=raw_run_data_schema)

# COMMAND ----------

db_ingestion(df1, 'raw_history_master', 'append')

# COMMAND ----------

db_ingestion(df2, 'raw_run_data', 'append')
