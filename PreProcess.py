import wfdb, math
import csv
import pandas as pd
import os
import numpy as np
import argparse
import pickle
import random
from scipy import signal
from scipy.signal import iirfilter, butter, filtfilt, iirnotch, find_peaks
from biosppy.signals import ecg
from itertools import islice
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from ecgdetectors import Detectors
import pdb

matplotlib.use('Agg')

random.seed(1000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat_csv", help='transfer ecg-id dat file to csv file', type=bool, default=False)
    # 原先版本有unfilt 和filt 但這只有在ECG-ID有效 這邊把它拿掉了
    parser.add_argument("--filter",
                        help='use bandpass filter,"lowpass" or "bandpass" or "none", implement before R detection',
                        default='lowpass')
    parser.add_argument("--lp_ct", help='lowpass cutoff frequency', type=float, default=35)
    parser.add_argument("--hp_ct", help='highpass cutoff frequency', type=float, default=0.25)
    parser.add_argument("--order", help='Butterworth filter order', type=int, default=8)
    parser.add_argument("--split_mode", help='Peak to Peak: pp / Fix length:　fl　', default='fl')
    parser.add_argument("--split_waveam", help='how many wave in one data', type=int, default=1)

    parser.add_argument("--fl_sampleLG", help='How may sample for fix length', type=int, default=430)
    parser.add_argument("--sample_rate", help='What is the data sampleing rate', type=int, default=500)
    parser.add_argument("--output_tt_graph", help='Output test graph', type=bool, default=False)
    parser.add_argument("--database", help='ECG-ID / PTB / MIT-BIH', default="ECG-ID")
    parser.add_argument("--ptb_option", help='health subjects or all subjects ', default="health")
    parser.add_argument("--test_ratio", help='You can enter test ratio between 0~0.5', type=float, default=0.1)
    args = parser.parse_args()

    return args


# generates features and labels
class ProcessData:
    def __init__(self, args, result_data={}):
        self.args = args
        self.database_dir = args.database
        self.sample_rate = args.sample_rate
        self.dir = '../' + self.database_dir + '/data/raw'
        self.persons_labels = []  # who the person is
        self.health_labels = []  # is the person health or not for PTB
        # self.age_labels = []     #age of thatperson
        # self.gender_labels = []  #is that person male or female
        # self.date_labels = []    #month.day.year of ecg record
        self.ecg_filsignal = pd.DataFrame()  # filtered ecg dataset for ECG-ID
        self.ecg_ufilsignal = pd.DataFrame()  # unfiltered ecg dataset for ECG-ID
        self.result_data = result_data
        self.total_people_amount = len(result_data)
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []

    # extracts labels and features from rec_1.hea of each person
    def extract_labels(self, filepath):
        p_index = -1
        # if self.args.database == "PTB" or self.args.database == "ECG-ID":
        for folders in os.listdir(filepath):
            if (folders.startswith('Person_') or folders.startswith('patient')):
                p_index += 1
                self.persons_labels.append(folders)

                if self.args.database == "PTB":
                    for onepersonsdir in os.listdir(os.path.join(filepath, folders)):
                        if onepersonsdir.endswith('hea'):
                            ecg_record = wfdb.rdheader(os.path.join(filepath, folders, onepersonsdir.split(".", 1)[0]))
                            patient_status = ecg_record.comments[4].split(":")[1].strip()
                            self.health_labels.append(patient_status)
                            break  # only read hea file one time per persone to know the health status
                # if (onepersonsdir.startswith('rec_1.') and onepersonsdir.endswith('hea')):
                #     with open(os.path.join(filepath, folders, onepersonsdir),"r") as f:
                #         array2d = [[str(token) for token in line.split()] for line in f]
                #         self.age_labels.append(array2d[4][2])
                #         self.gender_labels.append(array2d[5][2])
                #         self.date_labels.append(array2d[6][3])
                #     f.close()
            if (folders.endswith('csv')):  # This is for MIT-BIH database
                p_index += 1
                basename = folders.split(".", 1)[0]  # rec_1 rec_2....
                self.persons_labels.append(basename)  # no classify to person1 person2 dir

    def butter_lowpass(self, cutoff, sample_rate, order=8):
        nyq = 0.5 * sample_rate
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_bandpass(self, lowcut, highcut, sample_rate, order=2):
        nyq = 0.5 * sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def filter_signal(self, data, cutoff, sample_rate, order=2, filtertype='lowpass'):
        if filtertype.lower() == 'lowpass':
            b, a = self.butter_lowpass(cutoff, sample_rate, order=order)
        elif filtertype.lower() == 'bandpass':
            b, a = self.butter_bandpass(cutoff[0], cutoff[1], sample_rate, order=order)
        elif filtertype.lower() == 'notch':
            b, a = iirnotch(cutoff, Q=0.005, fs=sample_rate)
        else:
            raise ValueError(
                'filtertype: %s is unknown, available are: \lowpass, highpass, bandpass, and notch' % filtertype)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def remove_baseline_wander(self, data, sample_rate, cutoff=0.05):
        return self.filter_signal(data=data, cutoff=cutoff, sample_rate=sample_rate,
                                  filtertype='notch')

    def output_test_graph(self, filename, first_signal, first_signal_label, second_signal=None,
                          second_signal_label=None, wave_amount=3):
        print("Creating figure...")
        samples_amount = self.sample_rate * wave_amount
        plt.plot(first_signal[:samples_amount], color='r', linewidth=1, label=first_signal_label)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude(mv)')
        plt.title(f"Database {self.args.database}")
        if (second_signal != None and second_signal_label != None):
            plt.plot(second_signal[:samples_amount], color='b', linewidth=1, label=second_signal_label)
        plt.legend()
        plt.savefig('../' + self.args.database + '/data/filter_compare_img/example_' + filename + '.png')
        plt.clf()
        plt.close()
        print("Finish creating figure.")

    def output_peak_graph(self, filename, signal, peaks, threshold_value, wave_amount=3):
        print("Creating peak figure...")
        samples_amount = self.sample_rate * wave_amount
        plt.plot(signal[:samples_amount], color='r', linewidth=1, label="signal")
        plt.plot(peaks[:4], signal[peaks[:4]], "x")
        plt.plot([threshold_value for i in range(samples_amount)], color='b', linewidth=1, label="threshold_value")
        plt.xlabel('Samples')
        plt.ylabel('Amplitude(mv)')
        plt.title(f"Database {self.args.database}")
        plt.legend()
        plt.savefig('../' + self.args.database + '/data/peak_compare_img/example_' + filename + '.png')
        plt.clf()
        plt.close()
        print("Finish creating peak figure.")

    def find_peaks(self, filename, signal):
        # 人平均心律60~100下/分 以100下/分，每一個sample rate*(3/5)應該至少就要有一個p波
        # 更正方法: 取最大跟平均距離，最大值*0.8距離作為peak基準
        max_value = max(signal)
        mean_value = sum(signal) / len(signal)
        dis = abs(max_value - mean_value)
        threshold_value = mean_value + dis * 0.5
        peaks, _ = find_peaks(signal, height=threshold_value)
        self.output_peak_graph(filename, signal, peaks, threshold_value)
        return peaks

    def split_data_resample(self, filename, signal_data, peak):
        # 平均心律70下
        # 500hz 每秒500個點
        # 每60秒70下
        # 一下60/70秒
        # 60/70 * 500 = 428.57 ~ 430個點
        # sample_rate * 60/70
        each_data_sample = int(self.args.sample_rate * (6 / 7))
        prev_sample = int(each_data_sample * (2 / 5))
        next_sample = int(each_data_sample * (3 / 5))
        one_session_list = []
        if (self.args.split_mode == "pp"):
            for i in range(len(peak) - 1):
                temp_signal = signal_data[peak[i]:peak[i + 1]]
                signal.resample(temp_signal, each_data_sample)
                one_session_list.append(temp_signal)
        elif (self.args.split_mode == "fl"):
            for i in peak:
                if i - prev_sample >= 0 and i + next_sample < len(signal_data):
                    one_session_list.append(signal_data[i - prev_sample:i + next_sample])

        self.result_data[filename] += one_session_list

    # extract features from rec_1.csv of each person
    def extract_feats(self, filepath):
        p = -1
        global f_num
        f_num = 0  # file counter

        temp_folder_name = ""  # 因為我只檢查是否有csv，而在PTB裡面只取健康的人，因此DATA ID必需根據有CSV的情況去+1(138行)
        for folders in os.listdir(filepath):
            if (folders.startswith('Person_') or folders.startswith('patient')):
                print(f"Extracting features....{folders} \r", end='')
                for files in os.listdir(os.path.join(filepath, folders)):
                    if (files.endswith('csv')):
                        if temp_folder_name != folders:
                            temp_folder_name = folders
                            p += 1
                            self.total_people_amount += 1
                            self.result_data[str(p)] = []
                        with open(os.path.join(filepath, folders, files), "r") as x:
                            f_num = f_num + 1
                            features = pd.read_csv(x, header=[0])  # ecg-id : unfilter / PTB: lead I /
                            # pdfeats = pd.DataFrame(features)
                            pdfeats = features.apply(pd.to_numeric)  # string to float type
                            temp = [p]  # first value is the person id and the id is not the same as database
                            temp_signal = []
                            for rows in range(len(pdfeats)):
                                temp_signal.append(pdfeats.iloc[rows][0])  # ecg-id : unfilter / PTB: lead I / data

                            temp_signalnp = np.asarray(temp_signal, dtype=float)
                            temp_signalnpTlist = temp_signalnp.T.tolist()

                            if (self.args.filter == "none"):
                                temp_signalfilt = temp_signalnpTlist
                            else:
                                if (self.args.filter == "lowpass"):
                                    temp_signalfilt = self.filter_signal(temp_signalnpTlist, self.args.lp_ct,
                                                                         self.args.sample_rate, self.args.order,
                                                                         self.args.filter)
                                else:
                                    temp_signalfilt = self.filter_signal(temp_signalnpTlist,
                                                                         [self.args.lp_ct, self.args.hp_ct],
                                                                         self.args.sample_rate, self.args.order,
                                                                         self.args.filter)
                            preprocessData_list = self.remove_baseline_wander(temp_signalfilt, self.args.sample_rate)
                            self.output_test_graph(str(p), preprocessData_list, 'filter_signal', temp_signalnpTlist,
                                                   'original_signal')
                            peak_list = self.find_peaks(str(p), preprocessData_list)
                            self.split_data_resample(str(p), preprocessData_list, peak_list)
                        x.close()
            if (folders.endswith('csv')):  # This is for MIT-BIH database
                print(f"Extracting features....{folders} \r", end='')
                p += 1
                self.total_people_amount += 1
                self.result_data[str(p)] = []
                with open(os.path.join(filepath, folders), "r") as x:
                    f_num = f_num + 1
                    features = pd.read_csv(x, header=[0])  # ecg-id : unfilter / PTB: lead I /
                    # pdfeats = pd.DataFrame(features)
                    pdfeats = features.apply(pd.to_numeric)  # string to float type
                    temp = [p]  # first value is the person id and the id is not the same as database
                    temp_signal = []
                    for rows in range(len(pdfeats)):
                        temp_signal.append(pdfeats.iloc[rows][0])  # ecg-id : unfilter / PTB: lead I / data

                    temp_signalnp = np.asarray(temp_signal, dtype=float)
                    temp_signalnpTlist = temp_signalnp.T.tolist()
                    if (self.args.filter == "none"):
                        temp_signalfilt = temp_signalnpTlist
                    else:
                        if (self.args.filter == "lowpass"):
                            temp_signalfilt = self.filter_signal(temp_signalnpTlist, self.args.lp_ct,
                                                                 self.args.sample_rate, self.args.order,
                                                                 self.args.filter)
                        else:
                            temp_signalfilt = self.filter_signal(temp_signalnpTlist, [self.args.lp_ct, self.args.hp_ct],
                                                                 self.args.sample_rate, self.args.order,
                                                                 self.args.filter)
                    preprocessData_list = self.remove_baseline_wander(temp_signalfilt, self.args.sample_rate)
                    self.output_test_graph(str(p), preprocessData_list, 'filter_signal', temp_signalnpTlist,
                                           'original_signal')
                    peak_list = self.find_peaks(str(p), preprocessData_list)
                    self.split_data_resample(str(p), preprocessData_list, peak_list)
                x.close()

    def train_test_split_withSaving(self):
        min_size_index = min(self.result_data, key=lambda k: len(self.result_data[k]))
        test_size = math.floor(len(self.result_data[min_size_index]) * self.args.test_ratio)
        train_size = math.floor(len(self.result_data[min_size_index]) - test_size)
        # 開啟 CSV 檔案
        with open('../' + self.args.database + '/data/filecgdata.csv', 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            for i in range(self.total_people_amount):
                for data in self.result_data[str(i)]:
                    temp_id = [i]
                    output_data = temp_id + data.tolist()
                    writer.writerow(output_data)
                X_train, X_test = train_test_split(self.result_data[str(i)], train_size=train_size, random_state=1,
                                                   shuffle=True)
                Y_train = [i for _ in range(len(X_train))]
                Y_test = [i for _ in range(len(X_test))]

                self.train_x += X_train
                self.train_y += Y_train
                self.test_x += X_test
                self.test_y += Y_test
        csvfile.close()
        print("Finish saving...")

    def create_dataset(self):
        dataset_train = MyDataset(data=self.train_x, target=self.train_y)
        dataset_test = MyDataset(data=self.test_x, target=self.test_y)

        output = open("../" + self.args.database + "/train.pkl", 'wb')
        pickle.dump(dataset_train, output)
        output = open("../" + self.args.database + "/test.pkl", 'wb')
        pickle.dump(dataset_test, output)
        output = open("../" + self.args.database + "/labels_amount.pkl", 'wb')
        pickle.dump(self.total_people_amount, output)

    def init(self):
        # print("Setting up data labels..")
        # self.extract_labels(self.dir)
        # if self.args.database == "PTB":
        #     ecglabels = [list(i) for i in zip(self.persons_labels,self.health_labels)]
        # else:
        #     ecglabels = [list(i) for i in self.persons_labels]
        # print("Exporting labels to csv..")
        # df_ecglabels = pd.DataFrame(ecglabels)
        # df_ecglabels.to_csv(os.path.join('../'+self.args.database+'/data/', 'ecgdblabels.csv'), index=False)
        # print("Export complete.")
        if (len(self.result_data) == 0):
            print("Setting up data features..")
            self.extract_feats(self.dir)
            print("finish finding features")

            print("saving all filter data dictionary...")
            output = open("../" + self.args.database + "/filter_data.pkl", 'wb')
            pickle.dump(self.result_data, output)
            print("finish saving dictionary")

        print("Exporting feature set to csv..")
        self.train_test_split_withSaving()
        self.create_dataset()
        print("Export complete.")

        if (os.path.isfile(os.path.join('../' + self.args.database + '/data/', 'unfilecgdata' + "." + 'csv'))):
            print("Data in data/ folder is now ready for training.")


class MyDataset(Dataset):
    def __init__(self, data, target):
        super(MyDataset, self).__init__()
        self.data = data
        self.target = target

    def __getitem__(self, index):
        # print(type(self.data[index]))
        x = np.array(self.data[index], dtype=float)
        y = self.target[index]

        # x=torch.FloatTensor(x)
        return x, y

    def __len__(self):
        return len(self.data)


def main():
    # get argument
    args = parse_args()
    database_name = args.database.strip()
    print(args.database)
    # save dat to csv file
    if args.dat_csv:
        if database_name != "ECG-ID" and database_name != "MIT-BIH" and database_name != "PTB":
            print("Wrong database!")
            pass
        if (os.path.isfile(
                os.path.join('../' + database_name + '/data/', 'filecgdata' + "." + 'csv')) and os.path.isfile(
            os.path.join('../' + database_name + '/data/', 'unfilecgdata' + "." + 'csv'))):
            print("already have files")
            pass
        else:
            print("Reading files...")
            # convert all .dat files to .csv
            generate = csv_generator(args)
            generate.tocsv()

    # create an dataset
    if os.path.isfile(os.path.join('../' + database_name + '/', 'filter_data.pkl')):
        print("already have files")
        input_data = open('../' + database_name + '/filter_data.pkl', 'rb')
        filter_data = pickle.load(input_data)
        processing = ProcessData(args, filter_data)
        processing.init()
    else:
        processing = ProcessData(args)
        processing.init()


if __name__ == "__main__":
    main()
