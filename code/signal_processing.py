import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt


def butterLowFilt(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def filterData(data, cutoff=5, order=4):
    data = data.astype(float)

    t = data['time']
    xyz = data.iloc[:, 1:]
    dur = t[len(t) - 1] - t[0]
    fs = len(t) / dur

    xyzfilt = pd.DataFrame()

    for axis in xyz:
        dat = xyz[axis]
        xyzfilt[axis] = pd.Series(butterLowFilt(dat, cutoff, fs, order))
    output = pd.concat([data['time'], xyzfilt], axis=1)

    return output


def butterBand(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    y = filtfilt(b, a, data)

    return y


def downsample(data, resampleFs=20):
    names = data.columns
    t = data.time
    end = float(t[len(t) - 1])
    start = float(t[0])
    dur = end - start
    fs = len(t) / dur

    d = signal.resample(data, int(len(data) / (fs / resampleFs)))
    d = d[:, 1:]

    t1 = np.linspace(0, dur, int(dur * resampleFs))[np.newaxis]
    t1 = t1.reshape((-1, 1))

    output = pd.DataFrame(np.concatenate([t1, d], axis=1), columns=names)

    return output


def movingAverage_removed(data, windowSize=1):
    d = data.astype(float)

    t = d['time']
    xyz = d.iloc[:, 1:]
    dur = t[len(t) - 1] - t[0]
    fs = len(t) / dur

    window = int(round(fs * windowSize))

    xyz_meanRemoved = pd.DataFrame()

    for axis in xyz:
        dat = xyz[axis]
        xyz_meanRemoved[axis] = dat - dat.rolling(window, center=True).mean()
    output = pd.concat([data['time'], xyz_meanRemoved], axis=1)

    return output


def normalise(data):
    data = data.astype(float)

    xyz = data[['X_value', 'Y_value', 'Z_value']]

    xyznorm = (xyz - min(xyz.min())) / (max(xyz.max()) - min(xyz.min()))

    output = pd.concat([data['time'], xyznorm], axis=1)

    return output


def rms(data):

    data = data.astype(float)
    xyz = data.iloc[:, 1:]
    xyzrms = pd.DataFrame()

    for axis in xyz:
        dat = xyz[axis]
        xyzrms[axis] = pd.Series(np.sqrt(np.mean(np.square(dat))))
    output = xyzrms

    return output