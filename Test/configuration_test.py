# Databricks notebook source
name = 'bsv_demand_forecast'
type = 'web-intelligent'
name = 'app_log'
type = 'file'
level = 'DEBUG'
root_folder = 'cinnamon'
data_folder = 'data'
data_share = '/usr/app_data'
output_folder = 'output'
model_folder = 'model'
db_driver = 'mssql+pymssql'
server = 'jnj-md-sqldb.public.53da1f976c5f.database.windows.net'
schema = 'dbo'
# user = 'dbadmin'
# password = 'Summer@123456789'
# port = '3342'
#databaseName = 'jnj_db_dbx_test'

env = 'test'

jdbcHostname=dbutils.secrets.get(scope = "KeyVault_Key", key = "jdbcHostname") 
jdbcPort=dbutils.secrets.get(scope = "KeyVault_Key", key = "jdbcPort") 
userName = dbutils.secrets.get(scope = "KeyVault_Key", key = "UserName") 
userPassword = dbutils.secrets.get(scope = "KeyVault_Key", key = "UserPassword") 
databaseName = dbutils.secrets.get(scope = "KeyVault_Key", key = "testDatabase")
#databaseName = dbutils.secrets.get(scope = "KeyVault_Key", key = "devDatabase")
#jdbcHostname = "jnj-md-sqldb.public.53da1f976c5f.database.windows.net"

# COMMAND ----------


