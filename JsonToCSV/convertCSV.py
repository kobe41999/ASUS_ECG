from JsonToCSV.getJson import *
import csv

with open('train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'ECG_DATA', 'TYPE'])

    ecg_data = getData(getToken())

    for i in range(len(ecg_data)):
        writer.writerow([[ecg_data[i]['time']], [ecg_data[i]['data']], 0])
