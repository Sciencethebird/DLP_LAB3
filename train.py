# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import EEGNet
import DeepConvNet
from dataloader import read_bci_data

import matplotlib.pyplot as plt
import numpy as np

# print(torch.cuda.is_available())

# hyper parameters
epochs = 2000
batch_size = 64
lr = 1e-2
activation_functions = {"ReLU":nn.ReLU(), "LeakyReLU":nn.LeakyReLU(), "ELU":nn.ELU()}

# history
acc_history   = {"ReLU":{}, "LeakyReLU":{}, "ELU":{}}
best_test_acc = {"ReLU":0 , "LeakyReLU":0 , "ELU":0 }

# load data
train_X, train_Y, test_X, test_Y = read_bci_data()
train_data = DataLoader(list(zip(train_X, train_Y)), batch_size=batch_size, shuffle=True, num_workers=0)
test_data  = DataLoader(list(zip(test_X, test_Y))  , batch_size=batch_size, shuffle=True, num_workers=0)

def evaluate(net, data):
    acc_count = 0
    data_count = 0
    for idx, data_batch in enumerate(data):
        inputs = data_batch[0].float().cuda(0)
        labels = data_batch[1].long()
        outputs = net(inputs)
        predicts = np.argmax(outputs.cpu().detach().numpy(), axis = 1)
        for idx in range(len(predicts)):
            if predicts[idx] == labels[idx]:
                acc_count +=1
            data_count+=1
    acc = acc_count/data_count*100
    print(f"total correct: {acc_count}/{data_count}, {acc}%")
    return acc


for activation_function in activation_functions:
    # network
    net = EEGNet.Net(activation_function=activation_functions[activation_function]).cuda(0)
    #net = DeepConvNet.Net(activation_function=activation_functions[activation_function]).cuda(0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    print(net)

    # train history
    train_acc_history = []
    test_acc_history  = []

    for epoch in range(epochs): 
        print (f"\nEpoch {activation_function}", epoch)

        # network training
        running_loss = 0.0
        for idx, data_batch in enumerate(train_data):
            # load data
            inputs = data_batch[0].float().cuda(0)
            labels = data_batch[1].long().cuda(0)

            # zero the parameter gradients (turn off to avoid pytorch grad accumulate)
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)

            # calculate loss
            loss = criterion(outputs, labels)
            loss.backward()

            # update seights
            optimizer.step()
            running_loss += loss.item() # extracts loss's value as python float

        print("train loss: ", running_loss)

        # train dataset evaluation (validation)
        train_acc = evaluate(net = net, data = train_data)
        train_acc_history.append(train_acc)

        # test dataset evaluation (testing)
        test_acc  = evaluate(net = net, data = test_data)
        test_acc_history.append(test_acc)
        if test_acc > best_test_acc[activation_function]:
            best_test_acc[activation_function] = test_acc

    acc_history[activation_function] = {"train":train_acc_history, "test":test_acc_history}


# summary
print(best_test_acc)

plt_lines = []
for key in acc_history:
    test_line,  = plt.plot(acc_history[key]["train"], label=key+'_train')
    train_line, = plt.plot(acc_history[key]["test"] , label=key+'_test')
    plt_lines.append(test_line)
    plt_lines.append(train_line)

plt.legend(handles=plt_lines)
plt.hlines(max(best_test_acc.values()), 0, epoch, label="best test accuracy = {0:.2f}%".format(max(best_test_acc.values())))
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy(%)")
plt.ylim(50, 100)
plt.show()



#deepconv: {'ReLU': 81.66666666666667, 'LeakyReLU': 81.38888888888889, 'ELU': 81.94444444444444}
# {'ReLU': 87.96296296296296, 'LeakyReLU': 86.57407407407408, 'ELU': 83.24074074074073}