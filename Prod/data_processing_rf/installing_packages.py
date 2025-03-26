# Databricks notebook source
dbutils.library.installPyPI("sqlalchemy")
dbutils.library.installPyPI("pymssql")
dbutils.library.installPyPI("pystan")
dbutils.library.installPyPI("prophet")
dbutils.library.installPyPI("croston")
dbutils.library.installPyPI("pmdarima")
dbutils.library.installPyPI("statsmodels")
dbutils.library.installPyPI("tbats")
dbutils.library.installPyPI("xgboost")
dbutils.library.installPyPI("sktime")
dbutils.library.installPyPI("arch")
dbutils.library.restartPython()

# COMMAND ----------


