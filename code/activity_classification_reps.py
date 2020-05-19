
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.signal import find_peaks

from code.signal_processing import downsample, filterData, rms


def predictActivityReps(model, labels, session_data, fs, sample_len, figs=False):
    if figs == True:
        fig = plt.figure()
    window = fs * sample_len
    results = {}
    for s, session in enumerate(session_data.keys()):

        for sen, sensor in enumerate(session_data[session].keys()):

            signal_processed = downsample(filterData(session_data[session][sensor], cutoff=2), resampleFs=fs)

            for i in range(0, len(signal_processed) - window):
                segment = signal_processed[i:i + window]

                xyz_seg = np.dstack([np.transpose(segment.X_value.values), np.transpose(segment.Y_value.values),
                                     np.transpose(segment.Z_value.values)])
                xyz_seg_norm = (xyz_seg - xyz_seg.min()) / (xyz_seg.max() - xyz_seg.min())

                if i == 0:
                    windows = xyz_seg_norm
                else:
                    windows = np.vstack((windows, xyz_seg_norm))

            if sen == 0:
                sen_windows = windows
            else:
                sen_windows = np.dstack([sen_windows, windows])

        predictions = model.predict(sen_windows)
        preds = predictions.argmax(axis=1)

        signal_preds = pd.DataFrame(data={'time': signal_processed.time[0:-window], 'class': preds})
        peaks, _ = find_peaks(signal_preds['class'].diff().abs(), height=0.5)

        activity_times = signal_preds['time'][peaks].reset_index(drop=True)
        activity_times_diff = activity_times.loc[activity_times.diff() > 5]

        activity_type = ['interval']
        reps = [0]
        duration = []
        starts = []
        ends = []
        activity_count = 0

        for i, ii in enumerate(activity_times_diff):

            idx = activity_times.loc[activity_times == ii].index.values
            start = (activity_times.loc[idx - 1]) + 1
            end = (activity_times.loc[idx]) - 1
            dur = round(end.iloc[0] - start.iloc[0])

            class_idx = \
                signal_preds.loc[(signal_preds['time'] > start.iloc[0]) & (signal_preds['time'] <= end.iloc[0])][
                    'class'].unique()
            activity = labels[class_idx]

            if activity != 'interval':

                activity_count = activity_count + 1

                if activity == 'squats':
                    signal_sensor = downsample(filterData(session_data[session]['lac'], cutoff=2), resampleFs=fs)
                else:
                    signal_sensor = downsample(filterData(session_data[session]['gyr'], cutoff=2), resampleFs=fs)

                signal_select = signal_sensor.loc[(signal_sensor['time'] > start.iloc[0]) & (
                        signal_sensor['time'] <= end.iloc[0] + sample_len)].reset_index(drop=True)

                rmsxyz = rms(signal_select)
                max_axis = rmsxyz.idxmax(axis=1)
                x = np.array(signal_select[max_axis.iloc[0]])
                X = fftpack.fft(x)
                freqs = fftpack.fftfreq(len(x)) * 20
                peak_freq = abs(freqs[np.argmax(X)])
                x = x - np.mean(x)
                t = np.array(signal_select['time'])

                if np.mean(x[x > 0]) < abs(np.mean(x[x < 0])):
                    x = x * -1

                mean_signal = np.mean(x[x > 0])

                peaks, _ = find_peaks(x, height=mean_signal, distance=(20 / peak_freq) - (20 / peak_freq) * 0.5)

                activity_type.append(activity[0])
                activity_type.append('interval')

                reps.append(len(peaks))
                reps.append(0)

                starts.append(start.iloc[0])
                ends.append(end.iloc[0])

                if activity_count == 1:

                    interval_dur = round(start.iloc[0])

                elif activity_count > 1:

                    interval_dur = round(start.iloc[0] - ends[activity_count - 2])

                duration.append(interval_dur)
                duration.append(dur)

                if figs == True:
                    ax = fig.add_subplot(2, 3, activity_count)
                    ax.set_title(activity[0])
                    ax.plot(t, x, color='k')
                    ax.scatter(t[peaks], x[peaks], color='r')
                    plt.xlabel('time (s)')
                    plt.ylabel('rad/s')

        activity_type = activity_type[:-1]
        reps = reps[:-1]

        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        results[session] = pd.DataFrame(data={'activity': activity_type, 'reps': reps, 'duration': duration})

    return results
