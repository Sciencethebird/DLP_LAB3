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
epochs = 1000
batch_size = 64
lr = 1e-2


# network
net = DeepConvNet.Net(activation_function=nn.ReLU()).cuda(0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
print(net)


# load data
train_X, train_Y, test_X, test_Y = read_bci_data()
train_data = DataLoader(list(zip(train_X, train_Y)), batch_size=batch_size, shuffle=True, num_workers=0)
test_data  = DataLoader(  list(zip(test_X, test_Y)), batch_size=batch_size, shuffle=True, num_workers=0)

# train history
train_acc = []
test_acc  = []


for epoch in range(epochs): 
    print ("\nEpoch ", epoch)
    
    # network training
    running_loss = 0.0
    for idx, data_batch in enumerate(train_data):
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
        #print()

    print("train loss: ", running_loss)

    # train dataset evaluation (validation)
    acc_count = 0
    data_count = 0
    for idx, data_batch in enumerate(train_data):
        inputs = data_batch[0].float().cuda(0)
        labels = data_batch[1].long()

        outputs = net(inputs)
        predicts = np.argmax(outputs.cpu().detach().numpy(), axis = 1)
        #print(predicts)
        #print(labels.numpy())
        for idx in range(len(predicts)):
            if predicts[idx] == labels[idx]:
                acc_count +=1
            data_count+=1
            
    print(f"total correct: {acc_count}/{data_count}, {acc_count/data_count}%")
    train_acc.append(acc_count/data_count)

    # test dataset evaluation (validation)
    acc_count = 0
    data_count = 0
    for idx, data_batch in enumerate(test_data):
        inputs = data_batch[0].float().cuda(0)
        labels = data_batch[1].long()

        outputs = net(inputs)
        predicts = np.argmax(outputs.cpu().detach().numpy(), axis = 1)
        #print(predicts)
        #print(labels.numpy())
        for idx in range(len(predicts)):
            if predicts[idx] == labels[idx]:
                acc_count +=1
            data_count+=1
            
    print(f"total correct: {acc_count}/{data_count}, {acc_count/data_count}%")
    test_acc.append(acc_count/data_count)

plt.plot(train_acc)
plt.plot(test_acc)
plt.show()




 