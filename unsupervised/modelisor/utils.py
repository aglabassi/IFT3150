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

def narray_to_tensor_dataset(X_train, X_test, y_train, y_test, device):
        
    inputs_train = torch.tensor(X_train, dtype = torch.long).to(device) 
    targets_train = torch.tensor(np.array(y_train), dtype = torch.float32 ).to(device)
    
    inputs_test = torch.tensor(X_test, dtype = torch.long).to(device) 
    targets_test = torch.tensor(np.array(y_test), dtype = torch.float32 ).to(device)
    
    return TensorDataset(inputs_train, targets_train), TensorDataset(inputs_test, targets_test)
    

def train_model(net, X_train, y_train, X_valid, y_valid, criterion, optimizer, device, batch_size=16):
    
    #Tensorifying
    train, valid =  narray_to_tensor_dataset(X_train, X_valid, y_train, y_valid, device)
    trainloader = DataLoader(train, batch_size=batch_size)
    testloader = DataLoader(valid, batch_size=X_valid.shape[0]) #whole batch
    
    net.train()
    
    epoch_losses = []
    for i, data in enumerate(trainloader, 0):
        #Computing predictions
        inputs, targets = data
        outputs = net(inputs)
        
        #Updating parameters
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        #updating statistics
        with torch.no_grad():
            epoch_losses.append(loss.item())
    
    net.eval()
    
    epoch_valid_losses = []
    with torch.no_grad():
        for data in testloader:
            inputs, targets = data
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            epoch_valid_losses.append(loss.item())
    
                                
    return np.mean(epoch_losses), np.mean(epoch_valid_losses)

