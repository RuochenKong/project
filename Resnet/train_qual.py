import os
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from cusDataset import PPG
from resnet1d import Resnet34
import numpy as np


def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)


def build_optimizer(params, lr=0.001, weight_decay=0.0001):
    paramiters = filter(lambda p: p.requires_grad, params)
    optimizer = optim.Adam(paramiters, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    return optimizer, criterion


def saveModel(bestmodel):
    path = "../data/trainedModel.pth"
    torch.save(bestmodel.state_dict(), path)


def Accuracy(data_loader, model, isvalidation=True):
    if not isvalidation:
        path = "../data/trainedModel.pth"
        model = Resnet34().to('cuda')
        model.load_state_dict(torch.load(path))

    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in data_loader:
            signals, labels = data
            signals = signals.to('cuda')
            labels = labels.to('cuda')
            
            outputs = model(signals)
            target = torch.argmax(labels, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            accuracy += len(predicted[predicted == target])

    accuracy = (100 * accuracy / total)
    return accuracy


def train(train_loader, test_loader, num_epochs, lr=0.001, weight_decay=0.0001):
    best_accuracy = 0.0

    optimizer, loss_fn = build_optimizer(model.parameters(), lr, weight_decay)

    for epoch in range(num_epochs): 
        running_loss = 0.0

        for i, (signals, labels) in enumerate(train_loader, 0):
            signals = signals.to('cuda')
            labels = labels.float().to('cuda')
            optimizer.zero_grad()
            outputs = model(signals).requires_grad_()

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


        accuracy = Accuracy(test_loader, model)
        print('For epoch', epoch + 1, 'the test accuracy is %d %%' % (accuracy))

        #save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel(model)
            print('model saved')
            best_accuracy = accuracy


train_data = PPG('ppgqual/train.txt')
test_data = PPG('ppgqual/test.txt')


train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)


model = Resnet34().to('cuda')
model.apply(initialize_weights)
train(train_loader, test_loader, 5)
