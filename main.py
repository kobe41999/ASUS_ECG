import argparse
import pickle
import torch.nn as nn
from tqdm import *
import copy
import time
from Model import RNN, AutoEncoder
import EvalModel
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", help='ECG-ID / PTB / MIT-BIH', default="ECG-ID")
    parser.add_argument("--num_hiddens", help='RNN hidden layer', type=int, default=1024)
    parser.add_argument("--num_layers", help='RNN Layer', type=int, default=1)
    parser.add_argument("--bidirectional", help='bidirectional or not', type=bool, default=True)
    parser.add_argument("--epoch", help='epoch amount', type=int, default=2)
    parser.add_argument("--batch_size", help='batch size', type=int, default=1)
    parser.add_argument("--LR", help='learning rate', type=float, default=0.001)
    args = parser.parse_args()
    return args


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


def trainRNN(train_loader, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    trainArray = []
    lossTrainArray = []
    data_num = 0
    trainAcc = 0
    for epoch in tqdm(range(args.epoch)):
        model.train()
        correct = 0
        total = 0
        for i, (data, target) in enumerate(train_loader):
            data_num += 1
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            data = data.reshape(data.shape[0], -1, 1)
            # print(data.shape)

            # data = data.reshape(-1, data.shape{})
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            # print(outputs.shape)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct_pred = (predicted == target.data).sum()
            correct += correct_pred
            if (data_num + 1) % 10 == 0:
                tqdm.write('\n Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Acc: %.4f'
                           % (epoch + 1, args.epoch, data_num + 1, len(dataset_train) // args.batch_size, loss.item(),
                              (100 * correct_pred / target.size(0))))
        data_num = 0
        tqdm.write('\n train total accuracy: %.4f %%' % (100 * correct / total))
        trainArray.append(round(float(correct.item() / total), 5))
        lossTrainArray.append(round(float(loss.item()), 5))
        if 100 * correct / total > trainAcc:
            trainAcc = 100 * correct / total
    return model, criterion


def trainAutoEncoder(train_loader, model):
    criterion = nn.L1Loss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    history = []

    for epoch in tqdm(range(args.epoch)):
        model = model.train()
        train_losses = []
        num = 0
        for i, (data, target) in enumerate(train_loader):
            num += 1
            print(num)
            optimizer.zero_grad()
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            data = data.reshape(-1, 1)
            outputs = model(data)
            output_autoencoder_graph('test', data.tolist(), outputs.tolist())

            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model = model.eval()
        train_loss = np.mean(train_losses)
        history.append(train_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} ')
    model.load_state_dict(best_model_wts)
    return model.eval(), criterion, history


def output_autoencoder_graph(filename, first_signal, second_signal=None):
    print("Creating figure...")
    npa = np.asarray(first_signal, dtype=np.float32)
    plt.plot(npa, color='r', linewidth=1, label='real')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude(mv)')
    plt.title(f"Database {filename}")
    if second_signal is not None:
        plt.plot(second_signal, color='b', linewidth=1, label='decode')
    plt.legend()
    plt.savefig(f'./figure/autoencoderCompare/{filename}.png')
    plt.clf()
    plt.close()
    print("Finish creating figure.")


if __name__ == '__main__':
    print("驅動為：", device)
    print("GPU型號：", torch.cuda.get_device_name(0))
    # ===============================================================================
    args = parse_args()
    with open("./exampleData/train.pkl", 'rb') as file:
        dataset_train = pickle.load(file)
    file.close()
    with open("./exampleData/test.pkl", 'rb') as file:
        dataset_test = pickle.load(file)
    file.close()
    with open("./exampleData/labels_amount.pkl", 'rb') as file:
        labels_amount = pickle.load(file)
    file.close()
    # 讀取預先處理好之train test data以及本次分類的種類數
    # ===============================================================================
    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=6)
    # ===============================================================================
    # data, target = next(iter(train_loader))
    # print(data)
    # print(target)
    # print(labels_amount)
    # 確認data target 是否為正確
    # ===============================================================================
    print("===Prepare Model===")
    # ===============================================================================
    model = AutoEncoder.RecurrentAutoencoder(427, 1, 128)
    # model = RNN.RNN(num_hiddens=args.num_hiddens, num_layers=args.num_layers, bidirectional=args.bidirectional,
    #                 labels=labels_amount)
    # ===============================================================================
    if torch.cuda.is_available():
        model = model.to(device)

    # model, criterion = trainRNN(train_loader, model)
    model, criterion, history = trainAutoEncoder(train_loader, model)
    # ===============================================================================
    # model = RNN.RNN(1024, 1, True, 90)
    # model.load_state_dict(torch.load('./model_RNN.pth'))
    # model = model.to(device)
    # model.eval()
    # 載入權重用
    # ===============================================================================
    torch.save(model, 'model_AutoEncoder.pth')
    # 保存權重用
    # ===============================================================================
    # startTime = time.time()
    # EvalModel.test(test_loader, model, criterion, device)
    # endTime = time.time()
    # print(endTime - startTime)
