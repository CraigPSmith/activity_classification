import pandas as pd
from butterLowpassFilt import butterLowFilt
from scipy.signal import butter, filtfilt

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