import argparse
import pickle
import torch.nn as nn
from tqdm import *
from Model import RNN
import EvalModel
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", help='ECG-ID / PTB / MIT-BIH', default="ECG-ID")
    parser.add_argument("--num_hiddens", help='RNN hidden layer', type=int, default=1024)
    parser.add_argument("--num_layers", help='RNN Layer', type=int, default=1)
    parser.add_argument("--bidirectional", help='bidirectional or not', type=bool, default=True)
    parser.add_argument("--epoch", help='epoch amount', type=int, default=2)
    parser.add_argument("--batch_size", help='batch size', type=int, default=8)
    parser.add_argument("--LR", help='learning rate', type=float, default=0.001)
    args = parser.parse_args()
    return args


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    data, labels = zip(*batch)
    sorted_length = [len(size) for size in data]
    dataTensor = [torch.tensor(detail) for detail in data]
    dataTensor = torch.nn.utils.rnn.pad_sequence(dataTensor, batch_first=True, padding_value=-5)
    label = torch.tensor(labels)
    return dataTensor, label, sorted_length


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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


if __name__ == '__main__':
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

    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=6)
    print("===Prepare Model===")
    rnn = RNN.RNN(num_hiddens=args.num_hiddens, num_layers=args.num_layers, bidirectional=args.bidirectional,
                  labels=labels_amount)
    if torch.cuda.is_available():
        rnn = rnn.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.LR)

    trainArray = []
    lossTrainArray = []

    data_num = 0
    trainAcc = 0

    for epoch in tqdm(range(args.epoch)):
        rnn.train()
        correct = 0
        total = 0
        for i, (data, target) in enumerate(train_loader):

            data_num += 1
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            data = data.reshape(data.shape[0], -1, 1)

            # data = data.reshape(-1, data.shape{})
            target = target.to(device)
            # print(data.shape)

            optimizer.zero_grad()
            outputs = rnn(data)

            loss = loss_func(outputs, target)
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

    EvalModel.test(test_loader, rnn, loss_func, torch.cuda.is_available())
    # # rnn.test()
    # for i, (data, target) in enumerate(test_loader):
    #     outputs = rnn(data)
