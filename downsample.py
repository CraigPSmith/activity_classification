import numpy as np
import pandas as pd
from scipy import signal

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
