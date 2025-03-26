# Databricks notebook source
import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.initializers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# COMMAND ----------

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

seed_value = 1234

os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
#tf.random.set_random_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)

def mlp(data, forecast_length = 12, max_iter = 500):
    try:
        non_zero_index = data['historical_sale'].ne(0).idxmax()
        data = data.loc[non_zero_index:, :].copy()
        data = data[['historical_sale']]

        step_in = 12
        step_out = 12

        if data.shape[0] < (step_out + step_in):
            zeros_added = step_out + step_in - data.shape[0]
            data = pd.concat([pd.DataFrame({'historical_sale': np.zeros(zeros_added)}), data])
            data = data[['historical_sale']]

        shift_window = step_in + step_out

        data_shift = pd.concat([data.shift(periods=i).rename(columns={'sales': f'lag_{i}'}).copy() for i in range(shift_window)], axis=1)
        data_shift.reset_index(inplace=True, drop=True)
        data_shift.dropna(inplace=True)
        data_shift = data_shift.loc[:, ::-1].copy()
        data_shift = data_shift.iloc[[i for i in range(data_shift.shape[0]-1, 0, -step_out)], :].copy()

        data_values = np.array(data_shift.values).reshape((data_shift.shape[0], data_shift.shape[1]))

        x, y = data_values[:, :step_in].copy(), data_values[:, step_in:].copy()

        scale_x = MinMaxScaler()
        scale_y = MinMaxScaler()

        x = scale_x.fit_transform(x)
        y = scale_y.fit_transform(y)

        x = x.reshape(x.shape[0], x.shape[1])
        y = y.reshape(y.shape[0], y.shape[1])

        #         model = MLPRegressor(random_state=seed_value, max_iter = max_iter) 
        #         model.fit(x, y)
        initializer = he_uniform(seed=seed_value)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(step_in, activation='relu', input_shape=(x.shape[1], ), kernel_initializer=initializer))
        model.add(tf.keras.layers.Dropout(0.2, seed=seed_value)) # parameter
        model.add(tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer))
        model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer))
        model.add(tf.keras.layers.Dropout(0.2, seed=seed_value)) # parameter
        model.add(tf.keras.layers.Dense(step_out, kernel_initializer=initializer))
        model.compile(optimizer='adam', loss='mse')
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=2, mode='auto')
        model.fit(x, y, epochs=100, verbose=0, callbacks=[early_stop]) # parameter

        nearest_forecast_length = ((forecast_length // step_out) + 1) * step_out

        predictions = []

        x_pred = x[-1].reshape((1, step_in))

        for i in range(0, nearest_forecast_length, step_out):
            y_pred = model.predict(x_pred)
            predictions.extend(np.array(scale_y.inverse_transform(y_pred)).flatten())
            x_pred = x_pred.flatten()
            x_pred = x_pred[step_out:]
            x_pred = list(x_pred)
            x_pred.extend(list(y_pred.flatten()))
            x_pred = np.array(x_pred)
            x_pred = x_pred.reshape((1, step_in))

        yhat = predictions[:forecast_length]
        yhat = np.array(yhat).reshape((-1,1)).flatten()
        yhat = np.array(list(map(lambda x: int(np.round(x, 0)), yhat)))
    except:
        yhat = np.zeros(forecast_length)
    
    return yhat

# COMMAND ----------


