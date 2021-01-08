from JsonToCSV.getJson import *
import csv

with open('data0108.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Period', 'ECG', 'DeviceID'])

    ecg_data = getData(getToken())

    for i in range(len(ecg_data)):
        writer.writerow([[ecg_data[i]['time']], [ecg_data[i]['period']], [ecg_data[i]['data']], [ecg_data[i]['deviceId']]])