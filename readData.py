import csv
import config as C

ECG = []


def takeNum(data):

    result = ''.join(e for e in data if e.isalnum())
    return result


with open(C.trainCSV, newline='') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        dataLength = len(row['ECG_DATA'].split(" "))
        print(dataLength)
        for i in range(dataLength):
            ECG.append([row['ECG_DATA'].split(" ")[i]])
        print(ECG)
        ECG.clear()

