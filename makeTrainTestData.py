import csv
import math
import pickle
from sklearn.model_selection import train_test_split


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


if __name__ == '__main__':
