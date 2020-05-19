######################################################  TRAIN  ######################################################
from code.lstm_modelling import split_data, train_model, save_model
from code.readData import readRawData
from code.shape_data_lstm import reshape_lstm

raw_data = readRawData()  # Reads in training data

lstm_data, targets, sensor_order, fs, sample_len = reshape_lstm(raw_data, 20, 4,
                                                                undersample=True)  # Shapes data for LSTM model

X_train, X_test, y_train, y_test, labels = split_data(lstm_data, targets, 0.25, labels_filename='class_labels')
model = train_model(X_train, y_train, X_test, y_test, labels, batch_size=int(X_train.shape[0] / 10), epochs=500,
                    dropout=0.5,
                    matrix_plot=True)  # Builds LSTM model and outputs acurracy and confusion matrix for test data

save_model(model, labels, sensor_order, fs, sample_len, model_name='activity')  # Saves model and necessary parameters

######################################################  PREDICT  ######################################################
from code.lstm_modelling import import_model
from code.readData import readSessionSignals
from code.activity_classification_reps import predictActivityReps

model, labels, sensor_order, fs, sample_len = import_model('activity')  # Loads model and parameters

session_data = readSessionSignals(sensor_order)  # Loads signals from workout sessions

results = predictActivityReps(model, labels, session_data, fs, sample_len,
                              figs=True)  # Outputs results table (exercises completed, reps done, duration of each activity)
