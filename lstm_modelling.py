import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json


def split_data(data, targets, test_perc, labels_filename='test'):

    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=test_perc)

    y_train = pd.get_dummies(pd.DataFrame(data={'target': y_train.T})['target'])
    y_test = pd.get_dummies(pd.DataFrame(data={'target': y_test.T})['target'])
    labels = np.array(y_train.columns)



    return X_train, X_test, y_train, y_test, labels


def train_model(trainX, trainy, testX, testy, labels, batch_size=10, epochs=10, dropout=0.5, matrix_plot=False):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(dropout))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=0)

    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    print("accuracy = ", accuracy * 100)

    if matrix_plot == True:
        predictions = model.predict(testX)
        predictions = pd.DataFrame(data=predictions, columns=labels)
        preds = predictions.idxmax(axis=1)
        tas = testy.idxmax(axis=1)

        cm = confusion_matrix(tas, preds)
        norm_cm = cm / cm.sum(axis=1)[:, None] * 100

        plt.figure(figsize=(6, 6))
        plt.imshow(norm_cm, interpolation='nearest', cmap=plt.cm.rainbow)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=90)
        plt.yticks(tick_marks, labels)
        plt.ylabel('True')
        plt.xlabel('Predicted')

    return model


def save_model(model, labels, sensor_order, fs, sample_len, model_name='test'):

    np.save(model_name + '_labels', labels)
    np.save(model_name + '_sensor_order', sensor_order)
    np.save(model_name + '_fs', fs)
    np.save(model_name + '_sample_len', sample_len)

    model_json = model.to_json()
    with open(model_name + "_model.json", "w") as json_file:json_file.write(model_json)
    model.save_weights(model_name + "_model.h5")


def import_model(model_name):

    labels = np.load(model_name+'_labels.npy', allow_pickle=True)
    sensor_order = np.load(model_name + '_sensor_order.npy', allow_pickle=True)
    fs = np.load(model_name + '_fs.npy', allow_pickle=True)
    sample_len = np.load(model_name + '_sample_len.npy', allow_pickle=True)

    json_file = open(model_name+'_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name+'_model.h5')

    return loaded_model, labels, sensor_order, fs, sample_len
