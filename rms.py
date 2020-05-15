import pandas as pd
import numpy as np

def rms(data):

    data = data.astype(float)
    xyz = data.iloc[:, 1:]
    xyzrms = pd.DataFrame()

    for axis in xyz:
        dat = xyz[axis]
        xyzrms[axis] = pd.Series(np.sqrt(np.mean(np.square(dat))))
    output = xyzrms

    return output
