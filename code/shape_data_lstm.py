
import numpy as np
import pandas as pd
from code.segment_signal import segmentSignal
import random


def reshape_lstm(raw_data, fs, sample_len, undersample=False):

    for e, exercise in enumerate(raw_data.keys(), start=0):

        for s, subject in enumerate(raw_data[exercise].keys(), start=0):

            for r, rep in enumerate(raw_data[exercise][subject].keys(), start=0):

                if exercise == 'squats':
                    dominant_sensor = "lac"
                else:
                    dominant_sensor = "gyr"

                sen_samples, flags1, sensor_order, flag_key = segmentSignal(raw_data[exercise][subject][rep], fs,
                                                                            fs * sample_len, dominant_sensor)

                if r == 0:
                    r_samples = sen_samples
                    r_flags = flags1
                else:
                    r_samples = np.vstack((r_samples, sen_samples))
                    r_flags = np.vstack((r_flags, flags1))

            if s == 0:
                s_samples = r_samples
                s_flags = r_flags
            else:
                s_samples = np.vstack((s_samples, r_samples))
                s_flags = np.vstack((s_flags, r_flags))

            label = np.full((np.size(s_samples, 0), 1), exercise)  ##############

            if e == 0:
                lstm_data = s_samples
                labels = label
                noise_flags = s_flags
            else:
                lstm_data = np.vstack((lstm_data, s_samples))
                labels = np.vstack((labels, label))
                noise_flags = np.vstack((noise_flags, s_flags))


    idx = noise_flags == 0
    labels_new = pd.DataFrame(np.hstack((labels, idx)), columns=['label', 'flag'])
    labels_new['label'].loc[labels_new['flag'] == 'True'] = 'interval' #e + 1
    targets = np.array(labels_new['label'])

    if undersample == True:

        activity_idx = np.where(targets != 'interval')[0]
        interval_idx = np.where(targets == 'interval')[0]
        _, counts = np.unique(targets[activity_idx], return_counts=1)

        rand_idx = interval_idx[random.sample(range(0, interval_idx.size), np.max(counts))]
        all_idx = np.append(activity_idx, rand_idx)

        targets_output = targets[all_idx]
        lstm_data_output = lstm_data[all_idx, :, :]

    else:

        targets_output = targets
        lstm_data_output = lstm_data


    print('lstm_data shape...', lstm_data_output.shape)
    print('target counts...', np.unique(targets_output, return_counts=1))

    return lstm_data_output, targets_output, sensor_order, fs, sample_len
