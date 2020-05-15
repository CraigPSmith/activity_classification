import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from downsample import downsample
from filter import filterData
from movingAverage_removed import movingAverage_removed


def segmentSignal (data, fs, window, dominant_sensor):

    segment_data = data[dominant_sensor]

    filt_data = filterData(segment_data, cutoff=2)  # FILTER SIGNAL
    ds_data = downsample(filt_data, resampleFs=fs)  # DOWN SAMPLE SIGNAL TO REDUCE DATA SIZE
    maRemoved_data = movingAverage_removed(ds_data, windowSize=1).dropna()  # REMOVE MOVING AVERAGE
    ma_data = maRemoved_data.abs().rolling(80, center=True).mean()  # rolling absolute average to find periods of exercise

    for axis in ma_data.iloc[:, 1:]:

        ma_data[axis].loc[ma_data[axis] > ma_data.loc[:, axis].mean() + (
                ma_data.loc[:, axis].std() * 2)] = ma_data.loc[:, axis].mean() + (
                ma_data.loc[:, axis].std() * 2)  # cap values to mean + 2SD

    mean_axis = ma_data.loc[:, ['X_value', 'Y_value', 'Z_value']].mean()  # get average amplitude for each axis
    max_axis = mean_axis.idxmax(axis=0)  # find axis with max movement

    for type in range(2):

        if type == 0:
            signal_select = ma_data.loc[ma_data[max_axis] < ma_data[max_axis].mean(), ['time']].dropna().reset_index(drop=True)

        elif type == 1:
            signal_select = ma_data.loc[ma_data[max_axis] > ma_data[max_axis].mean(), ['time']].dropna().reset_index(
                drop=True)  # get time where movement is greater than average i.e. exercise

        signal_peaks, _ = find_peaks(signal_select.time.diff(), height=5)  # select periods where time between movement is >5s
        signal_periods = pd.DataFrame(signal_select.time[signal_peaks].reset_index(drop=True))
        signal_periods['flag'] = type  # create flag to show period of time IS exercise

        p = len(signal_periods)

        for pp in range(p + 1):

            if pp == 0:
                t = signal_select.loc[signal_select['time'] < signal_periods.loc[pp, 'time']].time

            elif pp == len(signal_periods):
                t = signal_select.loc[signal_select['time'] > signal_periods.loc[pp - 1, 'time']].time

            else:
                t = signal_select.loc[
                    (signal_select['time'] > signal_periods.loc[pp - 1, 'time']) & (signal_select['time'] < signal_periods.loc[pp, 'time'])].time

            if pp == 0:
                t_mins = t.min()
                t_maxs = t.max()
            else:
                t_mins = np.hstack((t_mins, t.min()))
                t_maxs = np.hstack((t_maxs, t.max()))

        ts_min = pd.DataFrame(data=t_mins, columns=['time'])
        ts_max = pd.DataFrame(data=t_maxs, columns=['time'])

        segment_n_all = []
        segment_n_round_all = []
        sensor_order = []
        for sen, sensor in enumerate(data.keys(), start=0):
            sensor_order.append(sensor)

            d = data[sensor]
            filt_data = filterData(d, cutoff=2)  # FILTER SIGNAL
            ds_data = downsample(filt_data, resampleFs=fs)  # DOWN SAMPLE SIGNAL TO REDUCE DATA SIZE

            samples = np.empty([1, window, 3])
            seg_count = 0

            for tt, min_t in enumerate(ts_min['time'], start=0):

                # flag = tss_min['flag'].iloc[tt]

                select = ds_data.loc[
                    (ds_data['time'] > min_t) & (ds_data['time'] < ts_max['time'].iloc[tt])].reset_index(
                    drop=True).drop(['time'], axis=1)

                if sen == 0:
                    segment_n = len(select) / window
                    segment_n_round = round(segment_n - 0.8 + 0.5)

                    segment_n_all.append(segment_n)
                    segment_n_round_all.append(segment_n_round)
                else:
                    segment_n = segment_n_all[tt]
                    segment_n_round = segment_n_round_all[tt]

                if segment_n_round >= 1:
                    seg_count = seg_count + 1

                    for seg in range(segment_n_round):

                        if seg == segment_n_round - 1 and segment_n_round > segment_n:
                            segment = select[seg * window:].reset_index(drop='index')
                        else:
                            segment = select[seg * window:(seg + 1) * window].reset_index(drop=True)

                        xyz_seg = np.dstack(
                            [np.transpose(segment.X_value.values), np.transpose(segment.Y_value.values),
                             np.transpose(segment.Z_value.values)])
                        xyz_seg_norm = (xyz_seg - xyz_seg.min()) / (xyz_seg.max() - xyz_seg.min())

                        if xyz_seg_norm.shape[1] < window:
                            pad = np.zeros((1, window - xyz_seg_norm.shape[1], 3))
                            xyz_seg_norm = np.hstack((xyz_seg_norm, pad))

                        if seg == 0:
                            xyz = xyz_seg_norm
                        else:
                            xyz = np.vstack((xyz, xyz_seg_norm))

                    if seg_count == 1:
                        samples = xyz

                    elif seg_count > 1:
                        samples = np.vstack((samples, xyz))

            if sen == 0:
                sen_samples = samples
            else:
                sen_samples = np.dstack([sen_samples, samples])


        flags = np.empty((sen_samples.shape[0], 1))
        flags.fill(type)

        if type == 0:
            samples1 = sen_samples
            flags1 = flags

        else:
            samples1 = np.vstack((samples1, sen_samples))
            flags1 = np.vstack((flags1, flags))

    flag_key = {'flag': [0,1], 'key':['interval', 'activity']}
    flag_key1 = pd.DataFrame(data=flag_key)

    return samples1, flags1, sensor_order, flag_key1
