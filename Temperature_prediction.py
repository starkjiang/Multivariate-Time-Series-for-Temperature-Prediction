# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:11:37 2018

@author: Zhanhong Jiang
"""
import numpy as np
from numpy import concatenate
from math import sqrt
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# load data - three different datasets: temperature_prediction, temperature_prediction1
# temperature_prediction2
# dataset = read_csv('temperature_prediction2.csv', parse_dates = True, index_col='Time')
dataset = read_csv('temperature_prediction2.csv', parse_dates = True)
dataset.drop('Occupancy', axis=1, inplace=True)
# manually specify column names
# temperature_prediction
# dataset.columns = ['Airflow', 'Discharge_air_temperature', 'CO2', 'Zone_temp']
# temperature_prediction1
# dataset.columns = ['Zone_temp', 'CO2', 'Discharge_air_temperature', 'Airflow']
# temperature_prediction2 without 'Time' column
dataset.columns = ['Airflow', 'Discharge_air_temperature', 'CO2', 'Zone_temp']
# dataset.index.name = 'Time'
# mark all NA values with 0
dataset['Zone_temp'].fillna(0, inplace=True)
# dataset = DataFrame(dataset)
dataset = dataset[['Zone_temp', 'Airflow', 'Discharge_air_temperature', 'CO2']]
# summarize first 5 rows
print (dataset.head(5))
# save to file
dataset.to_csv('temperature_data.csv')

# load dataset
dataset = read_csv('temperature_data.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
# load datast
dataset = read_csv('temperature_data.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,3] = encoder.fit_transform(values[:, 3])
# unsure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[5,6,7]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours = 2000
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# define the input dimension for neural networks
lags = 4

# design lstm network
model_lstm = Sequential()
model_lstm.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mae', optimizer='adam')
# design neural network
model_nn = Sequential()
model_nn.add(Dense(10, input_dim = lags, activation='relu'))
model_nn.add(Dense(1))
model_nn.compile(loss='mae', optimizer='adam')
# ARIMAX
#df = read_csv('temperature_data.csv')
# X = dataset[['Airflow', 'Discharge_air_temperature', 'CO2']]
X_1 = train_X.reshape((train_X.shape[0], train_X.shape[2]))
#X_1 = sm.add_constant(X_1)
X_2 = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#X_2 = sm.add_constant(X_2)
#y = train_y
#armax = sm.tsa.ARMA(y, order=(1,0), exog=X_1[:, 1:])
#results_armax = armax.fit()
#results_armax.summary()
#arimax_results = X_2[['Airflow', 'Discharge_air_temperature', 'CO2', 'Zone_temp']].dot(results_armax.params[1:,:]) + results_armax.params[0] * np.ones(test_y.shape)
# fit lstm network
history_lstm = model_lstm.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# fit nn
train_X_nn = train_X.reshape((train_X.shape[0], train_X.shape[2]))
test_X_nn = test_X.reshape((test_X.shape[0], test_X.shape[2]))
history_nn = model_nn.fit(train_X_nn, train_y, epochs=100, batch_size=72, validation_data=(test_X_nn, test_y), verbose=2)
# plot history
pyplot.plot(history_lstm.history['loss'], label='train_lstm')
pyplot.plot(history_lstm.history['val_loss'], label='test_lstm')
pyplot.plot(history_nn.history['loss'], label='train_nn')
pyplot.plot(history_nn.history['val_loss'], label='test_nn')
pyplot.legend()
pyplot.show()

# make a prediction
yhat_lstm = model_lstm.predict(test_X)
yhat_nn = model_nn.predict(test_X_nn)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat_lstm = concatenate((yhat_lstm, test_X[:, 1:]), axis=1)
inv_yhat_lstm = scaler.inverse_transform(inv_yhat_lstm)
inv_yhat_lstm = inv_yhat_lstm[:,0]
inv_yhat_nn = concatenate((yhat_nn, test_X_nn[:, 1:]), axis=1)
inv_yhat_nn = scaler.inverse_transform(inv_yhat_nn)
inv_yhat_nn = inv_yhat_nn[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse_lstm = sqrt(mean_squared_error(inv_y, inv_yhat_lstm))
rmse_nn = sqrt(mean_squared_error(inv_y, inv_yhat_nn))
#rmse_arimax = sqrt(mean_squared_error(inv_y, arimax_results))
print('Test LSTM_RMSE: %.3f' % rmse_lstm)
print('Test Neural Networks RMSE: %.3f' % rmse_nn)
#print('Test ARIMAX RMSE: %.3f' % rmse_arimax)
pyplot.plot(inv_y, label='Real')
pyplot.plot(inv_yhat_lstm, label='Predict using lstm')
pyplot.plot(inv_yhat_nn, label='Predict using nn')
#pyplot.plot(arimax_results, label='Predict using ARIMAX')
pyplot.legend()
pyplot.show()
