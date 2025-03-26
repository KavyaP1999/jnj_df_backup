# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, DecimalType, IntegerType
import pyspark.sql.functions as F
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

exp_parameters = spark.read.jdbc(url=url, table=f"(select parameter_default_value from parameter where parameter_id =159) as exp", properties=properties)

# COMMAND ----------

exp_pd = exp_parameters.toPandas()
exp_parameters = exp_pd['parameter_default_value'].values[0]
exp_parameters = eval(exp_parameters)

# COMMAND ----------

def h_sales_from_database(run_id, domain_id):
    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_DP3_run_data.parquet"
    df = spark.read.format('parquet').load(working_path,inferSchema=True)
    
    df = df.withColumn('historical_sale',F.col('historical_sale').cast(FloatType())).withColumn('forecast',F.col('forecast').cast(FloatType()))
    df = df.na.fill(value=0, subset=["forecast"])
    df = df.toPandas()
    return df

# COMMAND ----------

def nts_from_database(run_id, domain_id):
    df_nts_1 = spark.read.jdbc(url=url, table="nts_master", properties=properties)
    df_nts_1 = df_nts_1.filter((df_nts_1.run_id==run_id) & (df_nts_1.domain_id==domain_id))
    df_nts_1 = df_nts_1.withColumn('m1',F.col('m1').cast(FloatType())).withColumn('m2',F.col('m2').cast(FloatType())).withColumn('m3',F.col('m3').cast(FloatType())).withColumn('m4',F.col('m4').cast(FloatType())).withColumn('m5',F.col('m5').cast(FloatType())).withColumn('m6',F.col('m6').cast(FloatType())).withColumn('m7',F.col('m7').cast(FloatType())).withColumn('m8',F.col('m8').cast(FloatType())).withColumn('m9',F.col('m9').cast(FloatType())).withColumn('m10',F.col('m10').cast(FloatType())).withColumn('m11',F.col('m11').cast(FloatType())).withColumn('m12',F.col('m12').cast(FloatType()))
    df_nts_1 = df_nts_1.na.fill(value=0)
    df_nts_1 = df_nts_1.toPandas()
    return df_nts_1

# COMMAND ----------

def nts_master_ABC(run_id, domain_id):
    df_nts = nts_from_database(run_id, domain_id)
    df_nts['sum'] = df_nts.iloc[:, -12:].sum(axis=1)
    df_nts = df_nts[['key', 'sum']]
    df_nts.columns = ['product_id', 'sum']
    df_nts = df_nts.sort_values(by="sum", ascending=False, na_position='first')
    sum_nts = df_nts['sum'].sum()
    df_nts = df_nts.reset_index(drop=True)
    df_nts['cumsum'] = df_nts['sum'].cumsum()

    class_a_seg = exp_parameters['abc_seg']['class_a']
    class_b_seg = exp_parameters['abc_seg']['class_b']

    class_A_thresh = sum_nts * class_a_seg / 100
    class_B_thresh = sum_nts * class_b_seg / 100

    df_nts['Class'] = df_nts['cumsum'].apply(
        lambda x: 0 if x <= class_A_thresh else (1 if x <= class_B_thresh else 2))

    df_nts = df_nts[['product_id', 'sum', 'Class']]

    return df_nts

# COMMAND ----------

def hist_sales_ABC(run_id, domain_id):
    df_h = h_sales_from_database(run_id, domain_id)
    df_h['key'] = df_h['org_unit_id'] + '@' + df_h['channel_id'] + '@' + df_h['product_id'].astype(str)
    h_groups = df_h.groupby(by='key')
    df_sales = pd.DataFrame()
    for key, df in h_groups:
        df = df[-12:]
        sum_sales = df['historical_sale'].sum(axis=0) 
        df_sales = df_sales.append({"product_id": key, "sales": sum_sales}, ignore_index=True)

    df_sales.columns = ['product_id', 'sum']
    df_sales = df_sales.sort_values(by="sum", ascending=False, na_position='first')
    sum_h = df_sales['sum'].sum()
    df_sales = df_sales.reset_index(drop=True)
    df_sales['cumsum'] = df_sales['sum'].cumsum()

    class_a_seg = exp_parameters['abc_seg']['class_a']
    class_b_seg = exp_parameters['abc_seg']['class_b']

    class_A_thresh = sum_h * class_a_seg / 100
    class_B_thresh = sum_h * class_b_seg / 100

    df_sales['Class'] = df_sales['cumsum'].apply(
        lambda x: 0 if x <= class_A_thresh else (1 if x <= class_B_thresh else 2))
    df_sales = df_sales[['product_id', 'sum', 'Class']]
    return df_sales

# COMMAND ----------

def final_ABC(run_id, domain_id):
    df_sales = hist_sales_ABC(run_id, domain_id)
    df_nts = nts_master_ABC(run_id, domain_id)

    df_class = pd.merge(df_sales, df_nts, how='inner', on='product_id')
    df_class['Final_class'] = df_class[['Class_x', 'Class_y']].min(axis=1)
    df_class = df_class[['product_id', 'Final_class']]
    return df_class

# COMMAND ----------

def df_from_database(run_id, domain_id):
    working_path=f"/mnt/disk/staging_data/{env}/{domain_id}/Run_id_{run_id}/{run_id}_F3_run_data.parquet"
    df = spark.read.parquet(working_path,inferSchema=True,header=True)

    df = df.withColumn('historical_sale',F.col('historical_sale').cast(FloatType())).withColumn('forecast',F.col('forecast').cast(FloatType()))
    df = df.na.fill(value=0, subset=["forecast"])
    df = df.toPandas()
    
    df_f = spark.read.jdbc(url=url, table="future_forecast_master", properties=properties)
    df_f = df_f.filter((df_f.run_id==run_id) & (df_f.domain_id==domain_id))
    df_f = df_f.withColumn('tf',F.col('tf').cast(FloatType())).withColumn('bf',F.col('bf').cast(FloatType()))
    df_f = df_f.toPandas()
    
    df_p = spark.read.jdbc(url=url, table="past_forecast_master", properties=properties)
    df_p = df_p.filter((df_p.run_id==run_id) & (df_p.domain_id==domain_id))
    df_p = df_p.withColumn('bf_m00',F.col('bf_m00').cast(FloatType())).withColumn('bf_m01',F.col('bf_m01').cast(FloatType())).withColumn('bf_m02',F.col('bf_m02').cast(FloatType())).withColumn('bf_m03',F.col('bf_m03').cast(FloatType())).withColumn('tf_m00',F.col('tf_m00').cast(FloatType())).withColumn('tf_m01',F.col('tf_m01').cast(FloatType())).withColumn('tf_m02',F.col('tf_m02').cast(FloatType())).withColumn('tf_m03',F.col('tf_m03').cast(FloatType()))
    df_p = df_p.toPandas()
    
    df_class = final_ABC(run_id, domain_id)
    return df, df_f, df_p, df_class

# COMMAND ----------

# Adding User_file reading from blob which is uploaded by Planner.

# Reading ABC_format file from the blob storage
from pyspark.sql.functions import concat, col, lit
from pyspark.sql.types import StringType,BooleanType,DateType

# user_file = spark.read.format("csv") \
#                         .option("header", True) \
#                         .load(f"/mnt/customerdata/test/Incremental_Load/EMEA_PT/abc_format.csv")

user_file = spark.read.format("csv") \
                        .option("header", True) \
                        .load(f"/mnt/customerdata/{env}/Incremental_Load/{domain_id}/abc_format.csv")

#print(user_file)
# Creating Key column 
user_file = user_file.withColumn('key', concat(user_file['org_unit_id'], lit('@'), user_file['channel_id'], lit('@'), user_file['product_id']))
user_file = user_file.toPandas()
user_file.dtypes


# COMMAND ----------

def Exceptions_df(df, df_f, df_p, df_class, forecast_length, run_id, domain_id):
    dp3_data = h_sales_from_database(run_id, domain_id)

    df['key'] = df['org_unit_id'] + '@' + df['channel_id'] + '@' + df['product_id'].astype(str)

    df_dp3 = dp3_data.copy()
    df_dp3['key'] = df_dp3['org_unit_id'] + '@' + df_dp3['channel_id'] + '@' + df_dp3['product_id'].astype(str)

    df['forecast'] = df['forecast'].astype(float)
    df['historical_sale'] = df['historical_sale'].astype(float)
    df_dp3['forecast'] = df_dp3['forecast'].astype(float)
    df_dp3['historical_sale'] = df_dp3['historical_sale'].astype(float)

    df_p[['bf_m00','bf_m01','bf_m02','bf_m03','tf_m00','tf_m01','tf_m02','tf_m03']] = df_p[['bf_m00','bf_m01','bf_m02','bf_m03','tf_m00','tf_m01','tf_m02','tf_m03']].astype(float)
    df_f[['bf','tf']] = df_f[['bf','tf']].astype(float)
    ### exception 4 : forecast change percentage
    s4 = pd.DataFrame()
    df_values = pd.DataFrame()
    for i in np.unique(df['key']):
        j = i
        df_1 = df[df["key"] == i]
        df_1 = df_1.sort_values(by=['period'])

        df_1_forecast = df_1[-forecast_length:]
        df_1_forecast = df_1_forecast[0:12]

        df_f1 = df_f[df_f["key"] == i]
        df_f1 = df_f1.sort_values(by=['period'])
        df_f1 = df_f1[0:12] 

        df_merged = pd.merge(df_1_forecast, df_f1, how='inner', on='period')
        df_merged = df_merged[["forecast", "bf"]]

        df_merged['bf'] = df_merged['bf'].astype(float)

        df_merged['diff'] = np.abs(df_merged['forecast'] - df_merged['bf'])
        abs_sum = df_merged['diff'].sum(axis=0)
        forecast_sum = df_merged['bf'].sum(axis=0)

        if forecast_sum == 0:
            df_values = df_values.append({"product_id": j, "value": 0}, ignore_index=True)
            continue

        value = abs_sum / forecast_sum * 100

        df_values = df_values.append({"product_id": j, "value": value}, ignore_index=True)

    df_final = pd.merge(df_class, df_values, how='inner', on='product_id')

    excp_4_class_a_th = exp_parameters['exp_4']['class_a']
    excp_4_class_b_th = exp_parameters['exp_4']['class_b']
    excp_4_class_c_th = exp_parameters['exp_4']['class_c']

    df_final['thresh'] = df_final['Final_class'].apply(lambda x: excp_4_class_a_th if x == 0 else (excp_4_class_b_th if x == 1 else excp_4_class_c_th))
    df_final['Exception'] = df_final['value'] <= df_final['thresh']
    df_final['#4'] = df_final["Exception"].apply(lambda x: 1 if x == False else 0)
    s4 = df_final[['product_id', '#4']]
    s4 = s4.drop_duplicates()

    print('--\nException 4 \n')
    print(df_final.to_string())
    # print(s4)


    ### exception 5 :Comparing average sales for last six months and average forecast for next 6 months
    df_values = pd.DataFrame()
    s5 = pd.DataFrame()
    for i in np.unique(df['key']):
        j = i

        df_hist = df_dp3.loc[df_dp3['key'] == i]
        df_hist = df_hist.sort_values(by=['period'])

        df_fcst = df.loc[df['key'] == i]
        df_fcst = df_fcst.sort_values(by=['period'])
        df1_forecast = df_fcst[-forecast_length:]

        df_last_6_historical = df_hist[-6:]
        df_last_6_historical = df_last_6_historical[df_last_6_historical['historical_sale'] != 0]

        df_first_6_forecast = df1_forecast[0:6]
        df_first_6_forecast = df_first_6_forecast[df_first_6_forecast['forecast'] != 0]

        mean_historical = df_last_6_historical.historical_sale.sum(axis=0)

        if df_last_6_historical.shape[0] == 0:
            df_values = df_values.append({"product_id": j, "value": 0}, ignore_index=True)
            continue

        mean_historical = mean_historical / 6

        mean_forecast = df_first_6_forecast.forecast.sum(axis=0)

        if df_first_6_forecast.shape[0] == 0:
            df_values = df_values.append({"product_id": j, "value": 0}, ignore_index=True)
            continue
        
        mean_forecast = mean_forecast / 6

        percentage_change = np.abs(((mean_forecast - mean_historical) / mean_historical) * 100)
        df_values = df_values.append({"product_id": j, "value": percentage_change}, ignore_index=True)


    df_final = pd.merge(df_class, df_values, how='inner', on='product_id')

    excp_5_class_a_th = exp_parameters['exp_5']['class_a']
    excp_5_class_b_th = exp_parameters['exp_5']['class_b']
    excp_5_class_c_th = exp_parameters['exp_5']['class_c']

    df_final['thresh'] = df_final['Final_class'].apply(lambda x: excp_5_class_a_th if x == 0 else (excp_5_class_b_th if x == 1 else excp_5_class_c_th))
    df_final['Exception'] = df_final['value'] <= df_final['thresh']
    df_final['#5'] = df_final["Exception"].apply(lambda x: 1 if x == False else 0)
    s5 = df_final[['product_id', '#5']]
    s5 = s5.drop_duplicates()

    print('--\nException 5 \n')
    print(df_final.to_string())
    # print(s5)


    ### exception 6 : absolute difference between year to date growth and year to go growth percentage 
    df_values = pd.DataFrame()
    s6 = pd.DataFrame()
    for i in np.unique(df['key']):
        j = i

        df_hist = df_dp3.loc[df_dp3['key'] == i]
        df_hist = df_hist.sort_values(by=['period'])
        df_hist = df_hist.reset_index(drop=True)
        df1_historical = df_hist

        df1 = df.loc[df['key'] == i]
        df1 = df1.sort_values(by=['period'])
        df1 = df1.reset_index(drop=True)
        df1_forecast = df1[-forecast_length:]

        current_period = df1_forecast['period'].iloc[0]
        if current_period[-2:] in ['12', '01']:
            df_forecast = df1_forecast[0:12]
            df_forecast = df_forecast[df_forecast['forecast'] != 0]
            if df_forecast.shape[0] == 0:
                df_values = df_values.append({"product_id": j, "value": 0}, ignore_index=True)
                continue

            df_historical_1 = df1_historical[-24:-12]
            df_historical_1 = df_historical_1[df_historical_1['historical_sale'] != 0]
            if df_historical_1.shape[0] == 0:
                df_values = df_values.append({"product_id": j, "value": 0}, ignore_index=True)
                continue

            df_historical_2 = df1_historical[-12:]
            df_historical_2 = df_historical_2[df_historical_2['historical_sale'] != 0]
            if df_historical_2.shape[0] == 0:
                df_values = df_values.append({"product_id": j, "value": 0}, ignore_index=True)
                continue

            avg_1 = df_historical_1.historical_sale.sum(axis=0) / 12

            avg_2 = df_historical_2.historical_sale.sum(axis=0) / 12

            avg_forecast = df_forecast.forecast.sum(axis=0) / 12

            percent_diff_1 = ((avg_2 - avg_1) / avg_1) * 100

            percent_diff_2 = ((avg_forecast - avg_2) / avg_2) * 100

            # absolute difference
            diff = np.abs(percent_diff_1 - percent_diff_2)

        else:
            current_year = current_period[0:4]
            last_year = str(int(current_period[0:4])-1)
            df1['year'] = df1["period"].apply(lambda x: x[0:4])
            df1["month"] = df1["period"].apply(lambda x: x[-2:])

            df_hist['year'] = df_hist["period"].apply(lambda x: x[0:4])
            df_hist["month"] = df_hist["period"].apply(lambda x: x[-2:])

            ytd_months = [str(x).zfill(2) for x in range(1,int(current_period[-2:]))]
            ytg_months = [str(x).zfill(2) for x in range(int(current_period[-2:]), 13)]

            df_historical_1 = df_hist[df_hist['year']== last_year]
            df_historical_1 = df_historical_1[df_historical_1['historical_sale'] != 0]
            if df_historical_1.shape[0] == 0:
                df_values = df_values.append({"product_id": j, "value": 0}, ignore_index=True)
                continue

            df_historical_2 = df_hist[df_hist['year']== current_year]
            df_forecast_2 = df1[df1['year']== current_year]
            df_historical_2 = pd.merge(df_historical_2, df_forecast_2[['period','forecast','year','month']], on=['period','year','month'], how='outer')            
            df_historical_2['historical_sale'] = df_historical_2['historical_sale'].fillna(df_historical_2['forecast_y'])
            if df_historical_2.shape[0] == 0:
                df_values = df_values.append({"product_id": j, "value": 0}, ignore_index=True)
                continue

            ytd_cy_sum = df_historical_2[df_historical_2['month'].isin(ytd_months)]['historical_sale'].sum(axis=0).astype(float)
            ytd_ly_sum = df_historical_1[df_historical_1['month'].isin(ytd_months)]['historical_sale'].sum(axis=0).astype(float)

            ytg_cy_sum = df_historical_2[df_historical_2['month'].isin(ytg_months)]['historical_sale'].sum(axis=0).astype(float)
            ytg_ly_sum = df_historical_1[df_historical_1['month'].isin(ytg_months)]['historical_sale'].sum(axis=0).astype(float)

            if ytd_ly_sum==0 or ytg_ly_sum==0:
                df_values = df_values.append({"product_id": j, "value": 0}, ignore_index=True)
                continue

            ytd_growth = (ytd_cy_sum - ytd_ly_sum)/ytd_ly_sum*100
            ytg_growth = (ytg_cy_sum - ytg_ly_sum)/ytg_ly_sum*100

            diff = np.abs(ytd_growth - ytg_growth)
            # print(i, ' : ',diff)

        df_values = df_values.append({"product_id": j, "value": diff}, ignore_index=True)

    df_final = pd.merge(df_class, df_values, how='inner', on='product_id')

    excp_6_class_a_th = exp_parameters['exp_6']['class_a']
    excp_6_class_b_th = exp_parameters['exp_6']['class_b']
    excp_6_class_c_th = exp_parameters['exp_6']['class_c']

    df_final['thresh'] = df_final['Final_class'].apply(lambda x: excp_6_class_a_th if x == 0 else (excp_6_class_b_th if x == 1 else excp_6_class_c_th))
    df_final['Exception'] = df_final['value'] <= df_final['thresh']
    df_final['#6'] = df_final["Exception"].apply(lambda x: 1 if x == False else 0)
    s6 = df_final[['product_id', '#6']]
    s6 = s6.drop_duplicates()

    print('--\nException 6 \n')
    print(df_final.to_string())

    # print(s6)


    ### exception 7 : Comparing average sales for last year’s sales value and forecast values for 6 months.
    df_values = pd.DataFrame()
    s7 = pd.DataFrame()
    for i in np.unique(df['key']):
        j = i

        df_hist = df_dp3.loc[df_dp3['key'] == i]
        df_hist = df_hist.sort_values(by=['period'])
        df_hist = df_hist.reset_index(drop=True)
        df1_historical = df_hist

        df1 = df.loc[df['key'] == i]
        df1 = df1.sort_values(by=['period'])
        df1 = df1.reset_index(drop=True)
        df1_forecast = df1[-forecast_length:]
        
        df1_historical["month"] = df1_historical["period"].apply(lambda x: x[-4:])
        df1_historical["year"] = df1_historical["period"].apply(lambda x: x[0:4])

        df_first_6_forecast = df1_forecast[0:6]

        last_year_df = pd.DataFrame()
        for i in df_first_6_forecast.period:
            a, b = int(i[:4]) - 1, i[4:]
            x = df1_historical[(df1_historical["year"] == str(a)) & (df1_historical["month"] == b)]
            last_year_df = last_year_df.append(x, ignore_index=True)

        last_year_df = last_year_df[last_year_df['historical_sale'] != 0]

        if last_year_df.shape[0] <= 3:
            df_values = df_values.append({"product_id": j, "value": 0}, ignore_index=True)
            continue

        else:
            mean_historical_for_same_months = last_year_df['historical_sale'].sum() / 6

        sum_forecast = df_first_6_forecast.forecast.sum(axis=0)
        if df_first_6_forecast.shape[0] != 0:
            mean_forecast = sum_forecast / 6
        else:
            df_values = df_values.append({"product_id": j, "value": 0}, ignore_index=True)
            continue

        percentage_change = np.abs(
            ((mean_forecast - mean_historical_for_same_months) / mean_historical_for_same_months) * 100)
        df_values = df_values.append({"product_id": j, "value": percentage_change}, ignore_index=True)

    df_final = pd.merge(df_class, df_values, how='inner', on='product_id')

    excp_7_class_a_th = exp_parameters['exp_7']['class_a']
    excp_7_class_b_th = exp_parameters['exp_7']['class_b']
    excp_7_class_c_th = exp_parameters['exp_7']['class_c']

    df_final['thresh'] = df_final['Final_class'].apply(lambda x: excp_7_class_a_th if x == 0 else (excp_7_class_b_th if x == 1 else excp_7_class_c_th))
    df_final['Exception'] = df_final['value'] <= df_final['thresh']
    df_final['#7'] = df_final["Exception"].apply(lambda x: 1 if x == False else 0)
    s7 = df_final[['product_id', '#7']]
    s7 = s7.drop_duplicates()

    print('--\nException 7 \n')
    print(df_final.to_string())
    # print(s7)


    ### exception 8: Comparing forecast with past_forecast_masters for first 3 months of the historical_sales in terms of MAPE
    s8 = pd.DataFrame()
    for i in np.unique(df["key"]):
        j = i

        df_hist = df_dp3.loc[df_dp3['key'] == i]
        df_hist = df_hist.sort_values(by=['period'])
        df_hist = df_hist.reset_index(drop=True)

        df_1 = df_hist

        df_f1 = df_p[df_p["key"] == i]
        df_f1 = df_f1.sort_values(by="period")
        df_f1 = df_f1.reset_index(drop=True)

        df_1 = df_1[-3:]
        df_1 = df_1.reset_index(drop=True)
        df_f1 = df_f1[-3:]
        df_f1 = df_f1.reset_index(drop=True)
        df_final = pd.concat([df_1['historical_sale'], df_f1['bf_m03']], axis=1)
        df_final = df_final[df_final['historical_sale'] != 0]
        if df_final.shape[0] == 0:
            s8 = s8.append({'product_id': j, 'mean': 'not evalualted' , "#8": int(0)}, ignore_index=True)
            continue

        df_final["diff"] = np.abs(df_final["historical_sale"] - df_final['bf_m03'])

        if (df_final.bf_m03.isna().sum() != 0) or (df_final.historical_sale.isna().sum() != 0):
            s8 = s8.append({'product_id': j, 'mean': 'not evalualted' ,  "#8": int(0)}, ignore_index=True)
            continue
        
        sum_diff = df_final["diff"].sum()
        sum_sales = df_final["historical_sale"].sum()

        mean = sum_diff/sum_sales*100

        excp_8_mean_mape_th = exp_parameters['exp_8']['mean_mape']

        if mean <= excp_8_mean_mape_th:
            s8 = s8.append({'product_id': j, 'mean': mean ,  "#8": int(0)}, ignore_index=True)
        else:
            s8 = s8.append({'product_id': j, 'mean': mean ,  "#8": int(1)}, ignore_index=True)
    s8 = s8.drop_duplicates()

    print('--\nException 8 \n')
    print(s8.to_string())

    s8 = s8[['product_id', '#8']]



    ### exception 9 
    s9 = pd.DataFrame()
    for i in np.unique(df['key']):
        j = i
        df1 = df.loc[df['key'] == i]

        df_hist = df_dp3.loc[df_dp3['key'] == i]
        df_hist = df_hist.sort_values(by=['period'])
        df_hist = df_hist.reset_index(drop=True)

        df1_historical = df_hist

        excp_9_zero_check_m = exp_parameters['exp_9']['zero_check_months']

        df1_historical = df1_historical[-excp_9_zero_check_m:]
        df1_historical = df1_historical[df1_historical['historical_sale'] != 0]

        ## uncomment following if else block to enable this exception if required
        if df1_historical.shape[0] <= 3:
            s9 = s9.append({'product_id': j, "#9": 1}, ignore_index=True)
        else:
            s9 = s9.append({'product_id': j, "#9": 0}, ignore_index=True)

        ## to maintain the table schema the exception 9 flag is appended 0 for all product_id
        ## comment the following line if exception 9 is to be enabled in above if else block
        # s9 = s9.append({'product_id': j, "#9": int(0)}, ignore_index=True)
    s9 = s9.drop_duplicates()
    # print(s9)


    ### exception 10: Checking for phase of forecast.
    s10 = pd.DataFrame()
    for i in np.unique(df['key']):
        df1 = df.loc[df['key'] == i]
        j = i
        df_hist = df_dp3.loc[df_dp3['key'] == i]
        df_hist = df_hist.sort_values(by=['period'])
        df_hist = df_hist.reset_index(drop=True)
        df1_historical = df_hist

        df1 = df1.sort_values(by=['period'])
        df1 = df1.reset_index(drop=True)
        df1_forecast = df1[-forecast_length:]

        l3 = list(df1_forecast.forecast[0:12])  ### for MD it is change from 6 to 12
        df1_historical["month"] = df1_historical["period"].apply(lambda x: x[-4:])
        df1_historical["year"] = df1_historical["period"].apply(lambda x: x[0:4])

        df_first_12_forecast = df1_forecast[0:12] ### for MD it is change from 6 to 12

        df1_forecast = df1_forecast.forecast.replace(0, -1, inplace=True)

        #         if (df1_forecast.shape[0]!=6):
        #                       s10 = s10.append({'product_id': j, "#10": 1}, ignore_index=True)
        #                       continue

        l = []
        val = 0
        count = 0
        for month in df_first_12_forecast.period:
            val = 0
            count = 0
            a, b = int(month[:4]) - 1, month[4:]
            c, d = int(month[:4]) - 2, month[4:]
            e, f = int(month[:4]) - 3, month[4:]

            x = df1_historical[(df1_historical["year"] == str(a)) & (df1_historical["month"] == b)]
            x = x.reset_index()
            if x.shape[0] != 0:
                val += x["historical_sale"][0]
                count += 1

            x = df1_historical[(df1_historical["year"] == str(c)) & (df1_historical["month"] == d)]
            x = x.reset_index()
            if x.shape[0] != 0:
                val += x["historical_sale"][0]
                count += 1

            x = df1_historical[(df1_historical["year"] == str(e)) & (df1_historical["month"] == f)]
            x = x.reset_index()
            if x.shape[0] != 0:
                val += x["historical_sale"][0]
                count += 1

            if val == 0:
                break

            elif count != 0:
                l.append(val / count)
            else:
                break

        if val == 0:
            s10 = s10.append({'product_id': j, 'value': 'not evaluated', "#10": int(0)}, ignore_index=True)
            continue
        if count == 0:
            s10 = s10.append({'product_id': j, 'value': 'not evaluated',  "#10": int(0)}, ignore_index=True)
            continue

        df_final = pd.DataFrame({"forecast": l3, "l1": l})
        df_final = df_final[df_final['forecast'] != 0]

        l = list(df_final.forecast)
        l1 = list(df_final.l1)

        f_final = []
        for i, e in enumerate(l[:-1]):
            value = (l[i + 1] - l[i]) / l[i]
            f_final.append(value)

        l_final = []
        for i, e in enumerate(l1[:-1]):
            value = (l1[i + 1] - l1[i]) / l1[i]
            l_final.append(value)

        df_final = pd.DataFrame({"forecast": f_final, "l1": l_final})

        df_final['forecast'] = df_final['forecast'].astype(float)
        df_final['l1'] = df_final['l1'].astype(float)
        df_final['diff'] = np.abs((df_final['forecast'] - df_final['l1']) / (df_final['forecast'] + df_final['l1']))

        if df_final.shape[0] == 0:
            s10 = s10.append({'product_id': j, 'value': 'not evaluated',  "#10": int(0)}, ignore_index=True)
            continue

        mean = df_final[~df_final['diff'].isna()]['diff'].sum(axis=0) / df_final.shape[0]

        excp_10_phase_th = exp_parameters['exp_10']['phase_th']

        if mean <= excp_10_phase_th:
            s10 = s10.append({'product_id': j, 'value': mean,  "#10": int(0)}, ignore_index=True)
        else:
            s10 = s10.append({'product_id': j, 'value': mean,  "#10": int(1)}, ignore_index=True)
    s10 = s10.drop_duplicates()
    s10['#10'] = s10['#10'].astype(int)

    print('--\nException 10 \n')
    print(s10.to_string())
    s10 = s10[['product_id', '#10']]
    # print(s10)

    ### exception 11 : Compare TF and BF present in past_forecast_masters with historical_sales for last 3 months
    ### disabled in MD (to disable do not consider for sum in cumulative exception)
    s11 = pd.DataFrame()

    for i in np.unique(df["key"]):
        j = i
        df_1 = df[df["key"] == i]
        df_f1 = df_p[df_p["key"] == i]
        df_1 = df_1.sort_values(by="period")
        df_1 = df_1.reset_index(drop=True)
        df_f1 = df_f1.sort_values(by="period")
        df_f1 = df_f1.reset_index(drop=True)

        df_1 = df_1[0:-forecast_length]
        df_1 = df_1.reset_index(drop=True)

        # df_f1["period"]=df_f1["period"].apply(lambda x:str(x)[0:4]+"M0"+str(x)[4:])
        df_final = pd.merge(df_1, df_f1, how='inner', on='period')
        df_final = df_final[['period', 'historical_sale', 'bf_m03', 'tf_m03']]
        df_final = df_final[-3:]  # if need to check for all remove this line and it will work
        ### no data

        if (df_final.shape[0] == 0):
            s11 = s11.append({'product_id': j, "#11": int(0)}, ignore_index=True)
            continue

        df_final['diffA'] = np.abs(df_final['historical_sale'] - df_final['bf_m03'])
        df_final['diffB'] = np.abs(df_final['historical_sale'] - df_final['tf_m03'])
        df_final["ans"] = df_final["diffB"] <= df_final["diffA"]
        Exception_bool = False
        l = list(df_final.ans)
        for i in range(len(l) - 2):
            if (l[i] == False and l[i + 1] == False and l[i + 2] == False):
                Exception_bool = True
                break

        if Exception_bool:
            s11 = s11.append({'product_id': j, "#11": int(1)}, ignore_index=True)
        else:
            s11 = s11.append({'product_id': j, "#11": int(0)}, ignore_index=True)
    s11 = s11.drop_duplicates()
    # print(s11)


    ### exception 12: 
    s12 = pd.DataFrame()
    for i in np.unique(df['key']):
        j = i

        df_hist = df_dp3.loc[df_dp3['key'] == i]
        df_hist = df_hist.sort_values(by=['period'])
        df_hist = df_hist.reset_index(drop=True)
        df1_historical = df_hist[['historical_sale']]

        base = df_p.loc[df_p['key'] == i]
        base = base.sort_values(by=['period'])
        base = base.reset_index(drop=True)

        actuals = df1_historical[-3:].reset_index(drop=True)
        actuals.columns = ['actual']

        base = base[-3:][['bf_m03']].reset_index(drop=True)
        base.columns = ['base']

        naive = df1_historical[-15:-12].reset_index(drop=True)
        naive.columns = ['naive']

        final = pd.concat([actuals, base, naive], axis=1)
        final = final[final['actual'] != 0]
        final = final[final['base'] != 0]
        final = final[final['naive'] != 0]

        if final.shape[0] != 3 or final.isnull().values.any():
            s12 = s12.append({'product_id': j, 'mape_actual_base': 'not evaluated', 'mape_actual_naive': 'not evaluated', "#12": int(0)}, ignore_index=True)  ### no data no exception
            continue

        else:
            final['actual_base'] = (np.abs(final['base'] - final['actual']) / final['actual']) * 100
            final['actual_naive'] = (np.abs(final['naive'] - final['actual']) / final['actual']) * 100
            mape_actual_base = final['actual_base'].sum(axis=0) / 3
            mape_actual_naive = final['actual_naive'].sum(axis=0) / 3
            if mape_actual_base < mape_actual_naive:
                s12 = s12.append({'product_id': j,  'mape_actual_base': mape_actual_base, 'mape_actual_naive': mape_actual_naive, "#12": int(0)}, ignore_index=True)
            else:
                s12 = s12.append({'product_id': j,  'mape_actual_base': mape_actual_base, 'mape_actual_naive': mape_actual_naive,  "#12": int(1)}, ignore_index=True)
    s12 = s12.drop_duplicates()

    print('--\nException 12 \n')
    print(s12.to_string())
    
    s12 = s12[['product_id', '#12']]
    # print(s12)


    ### merge all exceptions

    Exc_final = pd.merge(s4, s5, how='inner', on='product_id')
    Exc_final = pd.merge(Exc_final, s6, how='inner', on='product_id')
    Exc_final = pd.merge(Exc_final, s7, how='inner', on='product_id')
    Exc_final = pd.merge(Exc_final, s8, how='inner', on='product_id')
    Exc_final = pd.merge(Exc_final, s9, how='inner', on='product_id')
    Exc_final = pd.merge(Exc_final, s10, how='inner', on='product_id')
    Exc_final = pd.merge(Exc_final, s11, how='inner', on='product_id')
    Exc_final = pd.merge(Exc_final, s12, how='inner', on='product_id')
    Exc_final = pd.merge(Exc_final, df_class, how='inner', on='product_id')

    ### exclude excp 9 for MD
    Exc_final["sum"] = Exc_final[['#4', '#5', '#6','#7', '#8', '#10', '#12']].sum(axis=1) 
    Exc_final["Exception"] = Exc_final["sum"].apply(lambda x: "Exception" if x > 0 else "Not an Exception")
    Exc_final.drop(['sum'], axis=1, inplace=True)

    Exc_final.columns = ['key', 'exception_4', 'exception_5', 'exception_6', 'exception_7', 'exception_8',
                         'exception_9', 'exception_10', 'exception_11', 'exception_12', 'abc_class', 'exception_status']

    Exc_final['run_id'] = int(run_id)
    Exc_final['domain_id'] = domain_id
    Exc_final = Exc_final.reindex(
        columns=['run_id', 'domain_id', 'key', 'abc_class', 'exception_4', 'exception_5', 'exception_6', 'exception_7',
                 'exception_8', 'exception_9', 'exception_10', 'exception_11', 'exception_12',
                 'exception_status'])

    Exc_final[['exception_4', 'exception_5', 'exception_6', 
                'exception_7', 'exception_8', 'exception_9',
                'exception_10', 'exception_11', 'exception_12']] = Exc_final[['exception_4', 'exception_5', 'exception_6',
                                                                                'exception_7', 'exception_8', 'exception_9',
                                                                                'exception_10', 'exception_11', 'exception_12']].astype(int)
    # Exc_final.to_csv('except.csv')
    Exc_final['abc_class'] = Exc_final['abc_class'].apply(lambda x: "A" if x == 0 else ("B" if x == 1 else "C"))
# 
# Condition for abc_class file uploaded by planner in blobe
#     # if condition 
    if user_file.shape[0] > 2:
        Exc_final = pd.merge(Exc_final, user_file, how='left', on='key')
        Exc_final = Exc_final.drop(columns=['abc_class_x','domain_id_y', 'org_unit_id', 'product_id', 'channel_id'])
        Exc_final = Exc_final.rename(columns = {'abc_class_y':'abc_class', 'domain_id_x':'domain_id'})
        my_abc = Exc_final.pop("abc_class")
        Exc_final.insert(3,"abc_class", my_abc)

    return Exc_final

# COMMAND ----------

def get_decisionTree_df(run_id, forecast_length, domain_id):
    df, df_f, df_p, df_class = df_from_database(run_id, domain_id)
    final_df = Exceptions_df(df, df_f, df_p, df_class, forecast_length, run_id, domain_id)
    final_df = final_df.drop_duplicates()
    return final_df
