import numpy as np
import pandas as pd
from scipy import signal


def downsample(data, resampleFs=20):
    output = {}

    for key, items in data.items():
        print(key)

        rData = {}
        for key1, items in data[key].items():

            datars = {}
            for key2, items in data[key][key1].items():

                d = data[key][key1][key2]

                names = d.columns
                t = d.time
                end = float(t[len(t) - 1])
                start = float(t[0])
                dur = end - start
                fs = len(t) / dur

                d1 = signal.resample(d, int(len(d) / (fs / resampleFs)))
                d1 = d1[:, 1:]

                t1 = np.linspace(0, dur, int(dur * resampleFs))[np.newaxis]
                t1 = t1.reshape((-1, 1))

                dat = pd.DataFrame(np.concatenate([t1, d1], axis=1), columns=names)

                datars[key2] = dat

            rData[key1] = datars

        output[key] = rData

    return output
