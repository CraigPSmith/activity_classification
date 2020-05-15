######################################################  TRAIN  ######################################################
from readData import readRawData
from shape_data_lstm import reshape_lstm
from lstm_modelling import split_data, train_model, save_model

raw_data = readRawData()

lstm_data, targets, sensor_order, fs, sample_len = reshape_lstm(raw_data, 20, 4, undersample=True)

X_train, X_test, y_train, y_test, labels = split_data(lstm_data, targets, 0.25, labels_filename='class_labels')
model = train_model(X_train, y_train, X_test, y_test, labels, batch_size=int(X_train.shape[0] / 10), epochs=500,
                    dropout=0.5, matrix_plot=True)

save_model(model, labels, sensor_order, fs, sample_len, model_name='activity')

######################################################  PREDICT  ######################################################
from lstm_modelling import import_model
from readData import readSessionSignals
from activity_classification_reps import predictActivityReps

model, labels, sensor_order, fs, sample_len = import_model('activity')

session_data = readSessionSignals(sensor_order)

results = predictActivityReps(model, labels, session_data, fs, sample_len, figs=False)




















xyz = pd.DataFrame()
for axis in list(['X_value', 'Y_value', 'Z_value']):

    x = ent_data[100:-100].loc[ent_data[axis] < ent_data[axis][100:-100].mean()-ent_data[axis][100:-100].std(), ['time']].dropna().reset_index(drop='index')

    peaks, _ = find_peaks(x.time.diff(), height=1)
    peak_times = x.time[peaks]
    periods = np.array(x.time[peaks])


    for pp in range(len(periods) + 1):

        if pp == 0 and len(periods) == 0:
            t = x.time

        elif pp == 0 and len(periods) > 0:
            t = x.loc[x['time'] < periods[pp]].time

        elif pp == len(periods):
            t = x.loc[x['time'] > periods[pp - 1]].time

        else:
            t = x.loc[
                (x['time'] > periods[pp - 1]) & (x['time'] < periods[pp])].time

        if pp == 0:
            t_diff = t.max() - t.min()
        else:
            t_diff = np.hstack((t_diff, t.max() - t.min()))

    max_diff = pd.DataFrame(np.array([[axis, t_diff.max()]]), columns=['axis', 'val'])

    xyz = xyz.append(max_diff)

axis_selection = xyz.loc[xyz['val'] == xyz['val'].min()].axis.to_numpy()


#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################


d = raw['s_press']['1']['r001']['lac']
filt_data = filterData(d, cutoff=2)  # FILTER SIGNAL
lac = downsample(filt_data, resampleFs=20)


rms_data = rms(lac)  # FIND ACCELERMOTER AXIS WITH LARGEST RMS AND USE THS AXIS TO SELECT

maxrms = rms_data.idxmax(axis=1).to_numpy()

ma_data_acc = lac.abs().rolling(100, center=True).mean()

plt.figure()
plt.plot(lac.time)
plt.plot(gyr.time)

plt.figure()
plt.plot(gyr.X_value, color='b')
plt.plot(gyr.Y_value, color='r')
plt.plot(gyr.Z_value, color='g')

plt.figure()
plt.plot(lac.X_value, color='b')
plt.plot(lac.Y_value, color='r')
plt.plot(lac.Z_value, color='k')



plt.figure()
plt.plot(ma_data_acc .X_value, color='b')
plt.plot(ma_data_acc .Y_value, color='r')

ma_data_acc.mean()

a=ma_data_acc.loc[:,['X_value', 'Y_value', 'Z_value']].mean()

b=a.idxmax(axis=0)

#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################

import pandas as pd
from downsample import downsample
from filter import filterData

from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array


d = pd.read_csv('/Users/craigsmith/phoneSensorData/quiet_standing_gyrm.csv')[1:].reset_index(drop=True)
filt_data = filterData(d, cutoff=5)
ds_data = downsample(filt_data, resampleFs=50).iloc[1000:-1000,:]
roll_mean = ds_data.rolling(25, center=True).mean().dropna().reset_index(drop=True)
series = roll_mean.loc[:, ['Y_value']]


# # date-time parsing function for loading the dataset
# def parser(x):
#     return datetime.strptime('190' + x, '%Y-%m')


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test


# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]


# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts


# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted


# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i + 1), rmse))


# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    # show the plot
    pyplot.show()


# load dataset
# series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# configure
n_lag = 50
n_seq = 20
n_test = 1000
n_epochs = 100
n_batch = 1
n_neurons = 100
# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test + 2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test + 2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
pyplot.figure()
plot_forecasts(series, forecasts, n_test + 2)

x = range(len(series)-n_test , len(series), 1)

pyplot.figure()
pyplot.plot(x,forecasts)

import numpy as np
array = np.array(forecasts[:,:])




