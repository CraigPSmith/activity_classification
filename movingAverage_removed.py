import pandas as pd

def movingAverage_removed (data, windowSize = 1):


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



