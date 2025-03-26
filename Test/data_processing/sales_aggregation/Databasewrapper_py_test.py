# Databricks notebook source
from datetime import datetime
from sqlalchemy import create_engine, text
from pandas import DataFrame
import os
import time

# COMMAND ----------

SELECT_TYPE = 'select'
INSERT_TYPE = 'insert'
UPDATE_TYPE = 'update'
DELETE_TYPE = 'delete'


class DatabaseWrapper:
    def __init__(self, db_driver=None, database=None, server=None, user=None, password=None, port=None):

        if db_driver is None:
            self.__db_driver__ = 'mssql+pymssql'
        else:
            self.__db_driver__ = 'mssql+pymssql'

        if database is None:
            self.__database__ = 'jnj_db_dbx_test'
            #self.__database__ = 'jnj_md_dev_v2'
        else:
            self.__database__ = 'jnj_db_dbx_test'
            #self.__database__ = 'jnj_md_dev_v2'
        if server is None:
            self.__server__ = 'jnj-md-sqldb.public.53da1f976c5f.database.windows.net'
        else:
            self.__server__ = 'jnj-md-sqldb.public.53da1f976c5f.database.windows.net'

        if user is None:
            self.__user__ = 'developer'
            #self.__user__ = 'dbadmin'
        else:
            self.__user__ = 'developer'
            #self.__user__ = 'dbadmin'

        if password is None:
            self.__password__ = 'Summer#123'
            #self.__password__ = 'Summer@123456789'
        else:
            self.__password__ = 'Summer#123'
            #self.__password__ = 'Summer@123456789'

        if port is None:
            self.__port__ = '3342'
        else:
            self.__port__ = '3342'

        # database_uri = '{}://{}/{}?driver=SQL Server?Trusted_Connection=yes'.format(self.__db_driver__,
        #                                                                             self.__server__,
        #                                                                             self.__database__)

        database_uri = '{}://{}:{}@{}:{}/{}'.format(self.__db_driver__,
                                                    self.__user__,
                                                    self.__password__,
                                                    self.__server__,
                                                    self.__port__,
                                                    self.__database__)
        print(database_uri)
        self._engine = create_engine(database_uri)

    def connect(self):
        conn = self._engine.connect()
        return conn

    def connect_raw(self):
        conn = self._engine.raw_connection()
        return conn

    def close(self, conn):
        try:
            conn.close()
        except:
            pass
        return True

    def execute_query(self, conn=None, query=None, query_type='select'):
        assert conn, "Connection is required"
        assert query_type in ['select', 'insert', 'update', 'delete'], \
            "Query Type can only be select, insert, update, delete"
        query = text(query)

        if query_type == 'select':
            results = conn.execute(query)
            data = DataFrame(data=results)
            return data
        else:
            results = conn.execute(query)
            return results

    # def insert_from_df(self, conn, table_name, data):
    #     start = datetime.now()
    #     config = AppConfiguration().get_config()
    #     data.to_sql(table_name, self._engine, index=False, if_exists="append", schema=config["database"]["schema"])
    #     end = datetime.now()
    #     # print("Bulk Insert Completed in Time:{}".format(end - start))

    def insert_from_df(self, conn, table_name, data):
        start = datetime.now()
        
        data.to_sql(table_name, self._engine, index=False, if_exists="append", schema="dbo",chunksize=5000, method='multi')
        end = datetime.now()
        print("Bulk Insert Completed in Time:{}".format(end - start))


        '''
         staging_table = self._create_staging_table(conn, table_name)
        start = datetime.now()
        copy_from_sql = """COPY {} FROM STDIN WITH (FORMAT CSV, DELIMITER ',', HEADER)""".format(table_name)
        # # print("Copy Query:{}".format(copy_from_sql))
        # write to file
        file_name = "temp_df_" + str(round(time.time() * 1000)) + ".csv"
        data.to_csv(file_name, index=False)
        with open(file_name) as file:
          conn.cursor().copy_expert(copy_from_sql, file=file)
        conn.commit(
        os.remove(file_name)
        end = datetime.now()
        print("Bulk Insert Completed in Time:{}".format(end - start))
       '''
