import argparse
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
import pandas as pd
import os
import csv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter",
                        help='use bandpass filter,"lowpass" or "bandpass" or "none", implement before R detection',
                        default='lowpass')
    parser.add_argument("--lp_ct", help='lowpass cutoff frequency', type=float, default=35)
    parser.add_argument("--hp_ct", help='highpass cutoff frequency', type=float, default=0.25)
    parser.add_argument("--order", help='Butterworth filter order', type=int, default=8)
    parser.add_argument("--fl_sampleLG", help='How may sample for fix length', type=int, default=430)
    parser.add_argument("--sample_rate", help='What is the data sampleing rate', type=int, default=55)
    parser.add_argument("--output_tt_graph", help='Output test graph', type=bool, default=False)
    parser.add_argument("--database", help='ECG-ID / PTB / MIT-BIH', default="ECG-ID")
    parser.add_argument("--ptb_option", help='health subjects or all subjects ', default="health")
    parser.add_argument("--test_ratio", help='You can enter test ratio between 0~0.5', type=float, default=0.1)
    args = parser.parse_args()

    return args


def butter_lowpass(cutoff, sample_rate, order=8):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_bandpass(lowcut, highcut, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filter_signal(data, cutoff, sample_rate, order=2, filtertype='lowpass'):
    if filtertype.lower() == 'lowpass':
        b, a = butter_lowpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'bandpass':
        b, a = butter_bandpass(cutoff[0], cutoff[1], sample_rate, order=order)
    elif filtertype.lower() == 'notch':
        b, a = iirnotch(cutoff, Q=0.005, fs=sample_rate)
    else:
        raise ValueError(
            'filtertype: %s is unknown, available are: \lowpass, highpass, bandpass, and notch' % filtertype)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def remove_baseline_wander(data, sample_rate, cutoff=0.05):
    return filter_signal(data=data, cutoff=cutoff, sample_rate=sample_rate,
                         filtertype='notch')


def output_test_graph(filename, first_signal, first_signal_label, second_signal=None,
                      second_signal_label=None, wave_amount=4):
    print("Creating figure...")
    sample_rate = 55
    samples_amount = sample_rate * wave_amount
    npa = np.asarray(first_signal[:samples_amount], dtype=np.float32)
    plt.plot(npa, color='r', linewidth=1, label=first_signal_label)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude(mv)')
    plt.title(f"Database {filename}")
    if second_signal_label is not None:
        plt.plot(second_signal[:samples_amount], color='b', linewidth=1, label=second_signal_label)
    plt.legend()
    plt.savefig(f'./figure/filterCompare/{filename}.png')
    plt.clf()
    plt.close()
    print("Finish creating figure.")


if __name__ == '__main__':
    args = parse_args()
    database_name = args.database.strip()
    print(args.database)
    with open('filtTry0126.csv', 'w', newline='') as Wr:
        writer = csv.writer(Wr)
        with open('try0126.csv', newline='') as csvFile:
            rows = csv.reader(csvFile)
            count = 0
            for row in rows:
                npa = np.asarray(row, dtype=np.float32)
                output_test_graph(f'sample_{count}', row, 'raw', remove_baseline_wander(filter_signal(npa, 20, 55), 55),'after')
                count += 1

                writer.writerow(remove_baseline_wander(filter_signal(npa, 15, 80), 80))
