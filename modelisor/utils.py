#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:51:53 2019

@author: Abdel
"""

#utils for traininng the neural net

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn import datasets
from CNN import CNN
import torch.optim as optim
from torch.nn import BCELoss
from sklearn.metrics import accuracy_score



def narray_to_tensor_dataset(X_train, X_test, y_train, y_test):
        
    inputs_train, targets_train = torch.tensor(X_train, dtype = torch.long), torch.tensor(np.array(y_train), dtype = torch.float32 ) 
    inputs_test, targets_test = torch.tensor(X_test, dtype = torch.long), torch.tensor(np.array(y_test), dtype = torch.float32 ) 
    
    return TensorDataset(inputs_train, targets_train), TensorDataset(inputs_test, targets_test)
    

def train_model(net, X_train, y_train, X_valid, y_valid, criterion, optimizer, metric_calculator, batch_size=4):
    
    #Tensorifying
    train, valid =  narray_to_tensor_dataset(X_train, X_valid, y_train, y_valid)
    trainloader = DataLoader(train, batch_size=batch_size)
    
    net.train()
    epoch_loss = 0
    
    for i, data in enumerate(trainloader, 0):
        #Computing predictions
        inputs, targets = data
        outputs = net(inputs)
        
        #Updating parameters
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
#        #Getting loss for validation set
#        inputs = torch.tensor([ list(instance[0]) for instance in valid ])
#        targets = torch.tensor([ instance[1] for instance in valid ])
#        outputs = net(inputs)
#        with torch.no_grad():
#            t = criterion(outputs, targets)
#            epoch_loss += t
#            print(t)
#    
    
        

# import some data to play with, testing
iris = datasets.load_iris()
X_train,X_test = iris.data[:100], iris.data[100:]
y_train, y_test = iris.target[:100], iris.target[100:]
def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

y_train, y_test = one_hot(y_train, 10), one_hot(y_test, 10)


cnn = CNN(100,4)
criterion = BCELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
            
train_model(cnn, X_train,y_train, X_test, y_test, criterion, optimizer, accuracy_score)


