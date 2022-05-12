import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from resnet1d import Resnet34
import numpy as np
import torch.optim as optim

def build_optimizer(params, lr = 0.001,  weight_decay = 0.0001):
    #params = filter(lambda p : p.requires_grad, params)
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    return optimizer, criterion


class PPG(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def __getitem__(self, index):
        return [self.data[index],self.labels[index]]

    def __len__(self):
        return len(self.data)


def saveModel(model):
    path = "../data/trainedModel.pth"
    torch.save(model.state_dict(), path)


def Accuracy(data_loader,model,isvalidation = True):
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
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    accuracy = (100 * accuracy / total)
    return accuracy


def train(train_loader, valid_loader, num_epochs, lr = 0.0001,  weight_decay = 0.00001):
    
    best_accuracy = 0.0

    model = Resnet34().to('cuda')
    optimizer,loss_fn = build_optimizer(model.parameters(),lr,weight_decay)
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        
        for i, (signals, labels) in enumerate(train_loader, 0):
            signals = signals.to('cuda')
            labels = labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(signals)
            outlabel = torch.argmax(outputs, dim=1).float().requires_grad_()
            loss = loss_fn(outlabel,labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss  
            if i % 1000 == 999:    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        accuracy = Accuracy(valid_loader,model)
        print('For epoch', epoch+1,'the test accuracy over the valid set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel(model)
            best_accuracy = accuracy


X = np.load('../data/ppg_data.npy')
y = np.load('../data/labels.npy')
X = torch.from_numpy(X).float().requires_grad_()
y = torch.from_numpy(y).float().requires_grad_()
data = PPG(X,y)
N = len(data)
trainN = int(N*0.7)
validN = int(trainN*0.15)
train_dataset, valid_dataset, test_dataset = random_split(data,(trainN-validN,validN,N-trainN))
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)



train(train_loader,valid_loader,10)
print('Finished Training')


acc = Accuracy(test_loader,None,False)
print('The test accuracy over the test set is %d %%' % (acc))

torch.cuda.empty_cache()


