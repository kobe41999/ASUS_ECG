import argparse
import csv
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_mode", help='Peak to Peak: pp / Fix length:　fl　', default='fl')
    parser.add_argument("--split_waveam", help='how many wave in one data', type=int, default=1)

    parser.add_argument("--fl_sampleLG", help='How may sample for fix length', type=int, default=430)
    parser.add_argument("--sample_rate", help='What is the data sampleing rate', type=int, default=55)
    parser.add_argument("--output_tt_graph", help='Output test graph', type=bool, default=False)
    parser.add_argument("--database", help='ECG-ID / PTB / MIT-BIH', default="ECG-ID")
    args = parser.parse_args()

    return args


def output_peak_graph(filename, signal, peaks, threshold_value, wave_amount=3):
    print("Creating peak figure...")
    samples_amount = args.sample_rate * wave_amount
    plt.plot(signal[:samples_amount], color='r', linewidth=1, label="signal")
    signalList = [signal[peaks[0]], signal[peaks[1]]]
    plt.plot(peaks[:2], signalList, "x")
    plt.plot([threshold_value for i in range(samples_amount)], color='b', linewidth=1, label="threshold_value")
    plt.xlabel('Samples')
    plt.ylabel('Amplitude(mv)')
    plt.title(f"Database {args.database}")
    plt.legend()
    plt.savefig(f'./figure/peakCompare/{filename}.png')
    plt.clf()
    plt.close()
    print("Finish creating peak figure.")


def findPeaks(filename, signal):
    # 人平均心律60~100下/分 以100下/分，每一個sample rate*(3/5)應該至少就要有一個p波
    # 更正方法: 取最大跟平均距離，最大值*0.8距離作為peak基準
    max_value = max(signal)
    mean_value = sum(signal) / len(signal)
    dis = abs(max_value - mean_value)
    threshold_value = mean_value + dis * 0.5
    print(threshold_value)
    peaks, _ = find_peaks(signal, height=threshold_value)
    print(filename)
    output_peak_graph(filename, signal, peaks, threshold_value)
    return peaks


def split_data_resample(filename, signal_data, peak):
    # 平均心律70下
    # 55hz 每秒55個點
    # 每60秒70下
    # 一下60/70秒
    # 60/70 * 55 = 47.1個點左右
    # sample_rate * 60/70
    each_data_sample = int(args.sample_rate * (6 / 7))
    prev_sample = int(each_data_sample * (2 / 5))
    next_sample = int(each_data_sample * (3 / 5))
    one_session_list = []
    if (args.split_mode == "pp"):
        for i in range(len(peak) - 1):
            temp_signal = signal_data[peak[i]:peak[i + 1]]
            signal.resample(temp_signal, each_data_sample)
            one_session_list.append(temp_signal)
    elif (args.split_mode == "fl"):
        for i in peak:
            if i - prev_sample >= 0 and i + next_sample < len(signal_data):
                one_session_list.append(signal_data[i - prev_sample:i + next_sample])
    return one_session_list


def ListStrToFloat(test_list):
    for i in range(0, len(test_list)):
        test_list[i] = float(test_list[i])

    return test_list


def ListStrToInt(test_list):
    print(test_list)
    for i in range(0, len(test_list)):
        test_list[i] = int(test_list[i])

    return test_list


if __name__ == '__main__':
    args = parse_args()
    with open('try0118.csv', 'r') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter=',')
        with open('splitTry.csv', 'w', newline='') as File:
            writer = csv.writer(File)
            count = 0
            for lines in csv_reader:
                ListStrToFloat(lines)
                # print(lines)
                peak_list = findPeaks(f'sample_{count}', lines)
                reSampleList = split_data_resample(f'sample_{count}', lines, peak_list)
                count += 1
                for sample in reSampleList:
                    writer.writerow(sample)

