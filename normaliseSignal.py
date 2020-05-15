import pandas as pd

def normalise(data):
    data = data.astype(float)

    xyz = data[['X_value', 'Y_value', 'Z_value']]

    xyznorm = (xyz - min(xyz.min())) / (max(xyz.max()) - min(xyz.min()))

    output = pd.concat([data['time'], xyznorm], axis=1)

    return output

