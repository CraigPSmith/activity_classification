import numpy as np
import pandas as pd
from butterLowpassFilt import butterLowFilt
from scipy import signal
from spectralEntropy import specEntropy


class signalProcess:
    def __init__(self, sensor_data):
        self.sensor_data = sensor_data

    def filterLp(self, cutoff=5, order=4):
        d = self.sensor_data
        d = d.astype(float)
        t = d['time']
        xyz = d.iloc[:, 1:]
        dur = t[len(t) - 1] - t[0]
        fs = len(t) / dur
        xyzfilt = pd.DataFrame()

        for axis in xyz:
            dat = xyz[axis]
            xyzfilt[axis] = pd.Series(butterLowFilt(dat, cutoff, fs, order))
            filteredData = pd.concat([d['time'], xyzfilt], axis=1)

        return filteredData


    def downsample(self, resampleFs=20):

        d = self.sensor_data

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

        downSampledData = pd.DataFrame(np.concatenate([t1, d1], axis=1), columns=names)

        return downSampledData

    def movingWindow(self, period=10, operation='average'):

        d = self.sensor_data
        time = d['time']
        pad1 = pd.DataFrame(np.zeros((int(period/2-1), 1)))
        pad2 = pd.DataFrame(np.zeros((int(period/2), 1)))

        data = pd.DataFrame()
        for names in d.columns[1:]:
            movingData = pd.DataFrame()
            for ii in range(0, len(d) - (period - 1)):

                if operation == 'average':

                    wn = pd.Series(np.mean(d[names].iloc[ii:ii+period]))

                #elif operation == 'removeAverage':

                    #wn = pd.Series(d[names].iloc[ii:ii+period] - np.mean(d[names].iloc[ii:ii+period]))

                elif operation == 'specEntropy':

                    wn = pd.Series(specEntropy(d[names].iloc[ii:ii+period]))

                df = pd.DataFrame(wn)
                movingData = movingData.append(df, ignore_index=True)

            paddedData = pd.concat([pad1, movingData, pad2], ignore_index=True)

            data[names] = paddedData.iloc[:, 0]

        output = pd.concat([time, data], axis=1)

        return output






