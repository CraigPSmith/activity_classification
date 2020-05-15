
import pandas as pd

def moving_range (data, windowSize = 5):


    d = data.astype(float)

    t = d['time']
    xyz = d.iloc[:, 1:]
    dur = t[len(t) - 1] - t[0]
    fs = len(t) / dur

    window = int(round(fs * windowSize))

    xyz_range = pd.DataFrame()

    for axis in xyz:
        dat = xyz[axis]
        xyz_range[axis] = (dat.rolling(window, center=True).max()-dat.rolling(window, center=True).min()).diff().abs()
    output = pd.concat([data['time'], xyz_range], axis=1)

    return output

data = processed_data['s1']['pushups']['r001']['acc']

a = moving_range(data)
plt.figure()
plt.plot(a['time'],a["X_value"])
plt.plot(data['time'],data["X_value"])


plt.figure()
plt.plot(a['time'],a["X_value"])
plt.plot(a['time'],a["Y_value"])
plt.plot(a['time'],a["Z_value"])