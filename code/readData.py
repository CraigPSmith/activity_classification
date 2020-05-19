# loads and sorts phone sensor data
import os
import pandas as pd
import numpy as np


def readRawData(path='data/train_data/'):

    exercises = os.listdir(path)
    exercises.remove('.DS_Store')
    eData = {}

    for e in exercises:
        path1 = path + e + '/'
        subjects = os.listdir(path1)
        subjects.remove('.DS_Store')
        sData = {}

        for s in subjects:
            path2 = path1 + s + '/'
            dataFiles = os.listdir(path2)

            if '.DS_Store' in dataFiles:
                dataFiles.remove('.DS_Store')

            for d, file in enumerate(dataFiles, start=0):

                sensor = file[4:7]
                rep = file[0:3]

                if d == 0:
                    reps = rep
                    sensors = sensor
                else:
                    reps = np.hstack([reps, rep])
                    sensors = np.hstack([sensors, sensor])

            reps = np.unique(reps)
            sensors = np.unique(sensors)
            rData = {}
            for repx in reps:

                senData={}
                for senx in sensors:

                    dat = pd.read_csv(path2 + repx + '_' + senx + '.csv')[1:].reset_index(drop=True)
                    senData[senx] = dat

                rData['r' + repx] = senData
            sData[s] = rData
        eData[e] = sData
    return eData


def readSessionSignals(sensor_order, path='data/sessions'):
    signals = os.listdir(path)
    signals.remove('.DS_Store')

    for d, file in enumerate(signals, start=0):

        session = file[0:3]

        if d == 0:
            sessions = session

        else:
            sessions = np.hstack([sessions, session])

    sessions = np.unique(session)

    session_data = {}
    for s in sessions:

        senData = {}
        for sen in sensor_order:
            dat = pd.read_csv(path + '/' + s + '_' + sen + '.csv')[1:].reset_index(drop=True)
            senData[sen] = dat

        session_data['session' + s] = senData

    return session_data
