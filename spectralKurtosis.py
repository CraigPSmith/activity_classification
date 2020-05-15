import pandas as pd
from scipy import signal
from scipy import stats
import numpy as np

def specKurt(data, windowSize = 1):

    d = data.astype(float)

    t = d['time']
    xyz = d.iloc[:, 1:]
    dur = t[len(t) - 1] - t[0]
    fs = len(t) / dur

    window = int(round(fs * windowSize))

    xyz_entropy = pd.DataFrame()

    for axis in xyz:
        dat = xyz[axis].fillna(0)

        sd = pd.Series()
        pad1 = pd.DataFrame(np.zeros((int(window / 2 - 1), 1)))
        pad2 = pd.DataFrame(np.zeros((int(window / 2), 1)))
        for ii in range(0, len(dat) - (window - 1)):
            f, Pxx_den = signal.periodogram(dat[ii:ii + window], window, return_onesided=True, scaling='spectrum')
            Pxx_denN = Pxx_den / sum(Pxx_den)  # normalise spectral density
            ent = pd.Series(stats.kurtosis(Pxx_denN)) # entropy of normalised spectral density
            sd = sd.append(ent, ignore_index=True)

        sd_padded=pd.concat([pad1, sd, pad2], ignore_index=True)

        xyz_entropy[axis] =sd_padded.iloc[:, 0]

    output = pd.concat([data['time'], xyz_entropy], axis=1)

    return output
