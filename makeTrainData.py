import csv
import config as C

ECG = []
allECG = []


def takeNum(data):
    result = ''.join(e for e in data if e.isdigit() or (e.split('-')[-1]).isdigit())
    return result


with open(C.trainCSV, newline='') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        dataLength = len(row['ECG_DATA'].split(" "))
        for i in range(dataLength):
            ecgData = takeNum([row['ECG_DATA'].split(" ")[i]])
            if ecgData != "":
                ECG.append(ecgData)

        allECG.append(ECG)
        ECG = []
        # .clear()連賦予的值也會改變，故用Initial的方式

with open('train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    print(allECG[0])
    for data in allECG:
        writer.writerow(['Label', 'Normal'])
        writer.writerow(data)
