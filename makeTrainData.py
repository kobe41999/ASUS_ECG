import csv
import config as C
import pandas as pd
from sklearn import preprocessing
import numpy as np


def changeToList(data):
    dataList = []
    first = data[0].replace("['", "")
    dataList.append(first)

    for i in range(len(data) - 3):
        dataList.append(data[i + 1])

    last = data[len(data) - 1].replace("']", "")
    dataList.append(last)

    return dataList


if __name__ == '__main__':
    df = pd.read_csv('./JsonToCSV/data0112.csv')
    ecgList = []
    recordLen = 10000
    for i in range(len(df.ECG)):
        ecgList.append(changeToList(df.ECG[i].split(" ")))

    for j in range(len(ecgList)):
        if recordLen > len(ecgList[j]):
            recordLen = len(ecgList[j])
    numOfRow = []

    for k in range(recordLen - 1):
        numOfRow.append(k)

    with open('try0112.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(numOfRow)
        for j in range(len(ecgList)):

            # 標準化處理
            npa = np.asarray(ecgList[j], dtype=np.float32)
            norm = np.linalg.norm(npa)
            normal_array = npa / norm

            writer.writerow(normal_array[0:(recordLen - 1)])
