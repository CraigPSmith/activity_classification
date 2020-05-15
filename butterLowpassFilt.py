from scipy.signal import butter, filtfilt

def butterLowFilt(data, cutoff, fs, order=4):

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


