#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:40:03 2022

@author: ckervazo
"""
import torch
import torch.nn as nn
import numpy as np
from LISTA import LISTA
from LPALM import LPALM
import matplotlib.pyplot as plt
from scipy.io import savemat
import os

#%%
def train_non_blind(train_loader, val_loader, num_epochs=100, lr=0.0001, T=10, alpha = 10, theta_shared=True,We_shared=True,G_shared=True,optSave=True):
    # Check notebook for the variables documentation
    
    criterion = nn.MSELoss()
    A = next(iter(train_loader))[1]
    
    folderInitSave = './results'
    if optSave == True: # You do not need to change anything here
        folderInitSave = folderInitSave + 'LISTA num_epochs ' + str(num_epochs) + ', T ' + str(T) + ', minibatch ' + str(A.size()[0]) + ', lr ' + str(lr)
        if not os.path.isdir(folderInitSave):
            os.makedirs(folderInitSave)
        else:
            raise NameError('Already existing folder')
            
            
    model = LISTA(T, alpha= alpha, A=A,theta_shared=theta_shared,We_shared=We_shared,G_shared= G_shared) # Declare the model that you will train, namely here LISTA
        
    
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr, betas=(0.9, 0.999)) # We here declare that we will use the adam optimizer.
    
    train_total_loss = [] # This list will contain the values of the traning loss at the different epochs
    val_total_loss = [] # This list will contain the values of the testing loss at the different epochs
    
    
    # First part : train the model for an epoch
    for epoch in range(num_epochs): # Run the code for the fixed number of epochs
        train_total = 0 # For display
        model.train()

        for i, (X, _, S) in enumerate(train_loader): # Enumerate all the mini-batches
            # TO DO : code an optimization step of the neural network for the current mini-batch. The principle is that you predict the sources with the current model, compute the loss train_loss (the NMSE, not directly criterion) and perform a gradient step. You might require the following functions : torch.numel(S), torch.norm(), optimizer.step(), optimizer.zero_grad(), train_loss.backward().
            
            optimizer.zero_grad()
            S_pred = model(X)
            train_loss = torch.mean(((S_pred - S)**2).sum(axis=(-1,-2)) / ((S**2).sum(axis=(-1,-2))))
            train_loss.backward()
            optimizer.step()
            
            train_total += train_loss.item() # Just for display
        train_total /= i+1
        train_total_loss.append(train_total)

                
                
        # Second part : evaluate the model at the current epoch
        model.eval(),print('Evaluation')
        with torch.no_grad():
            val_total = 0
            for i, (X, _, S) in enumerate(val_loader): # Here, you must enumerate the different mini-batches, apply the estimated model, and evaluate the loss for display
                
                S_pred = model(X)
                val_loss = torch.mean(((S_pred - S)**2).sum(axis=(-1,-2)) / ((S**2).sum(axis=(-1,-2))))
                val_total += val_loss.item()
                
            val_total /= i+1
            val_total_loss.append(val_total)
            
        if epoch % 5 == 0:
            print()
        
        print("epoch:{} | training loss:{:.5f} | validation loss:{:.5f} ".format(epoch, train_total,val_total))
        
        
        if optSave == True: # You don't need to change anything here
            torch.save(model,folderInitSave + '/LISTA_epoch_' + str(epoch)+'.pth')
            torch.save(val_total_loss,folderInitSave + '/val_total_loss_epoch_' + str(epoch)+'.pth')
            torch.save(train_total_loss,folderInitSave + '/train_total_loss_epoch_' + str(epoch)+'.pth')
            
            
    return train_total_loss, val_total_loss,model






#%%
def train_blind(train_loader, val_loader, num_epochs=10, T=10,theta_shared=False, LISTA_CP_S = True, W_CP_S_shared= False, learn_L_A = True , L_A_shared= False, lr = 0.0001,optSave=True,folderInitSave = ''):
    
    
    
    
    
    
    torch.autograd.set_detect_anomaly(True)# You do not need to change anything here
    
    criterion = nn.MSELoss()# You do not need to change anything here
    
    A = next(iter(train_loader))[1]
    S = next(iter(train_loader))[2]
    
    if optSave == True:# You do not need to change anything here
        folderInitSave = folderInitSave + 'LPALM num_epochs ' + str(num_epochs) + ', T ' + str(T) + ', minibatch ' + str(A.size()[0]) + ', lr ' + str(lr)
        if not os.path.isdir(folderInitSave):
            os.makedirs(folderInitSave)
        else:
            raise NameError('Already existing file')
    
    model = LPALM(T, S=S , A=A,theta_shared=theta_shared, W_CP_S_shared= W_CP_S_shared, learn_L_A = learn_L_A , L_A_shared= L_A_shared)# Declare the model that you will train, namely here LPALM
    
    
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr, betas=(0.9, 0.999)) # Declare the optimizer

    train_total_loss = []
    val_total_loss = []

    for epoch in range(num_epochs):# Run the code for the fixed number of epochs
        train_total = 0
        model.train()
        print("Epoch %s"%epoch)
        for i, (X, A, S) in enumerate(train_loader): # For each mini-batch
            # TO DO : code an optimization step of the neural network for the current mini-batch. The principle is that you predict the sources with the current model, compute the loss train_loss (the NMSE over A + NMSE over S, not directly criterion) and perform a gradient step. You might require the following functions : torch.numel(S), torch.norm(), optimizer.step(), optimizer.zero_grad(), train_loss.backward().
            
            optimizer.zero_grad()
            S_pred, A_pred = model(X)
            train_loss = torch.mean(((S_pred - S)**2).sum(axis=(-1,-2)) / ((S**2).sum(axis=(-1,-2))))
            train_loss += torch.mean(((A_pred - A)**2).sum(axis=(-1,-2)) / ((A**2).sum(axis=(-1,-2))))
            train_loss.backward()
            optimizer.step()
            
            train_total += train_loss.item()# Just for display
        train_total /= i+1
        train_total_loss.append(train_total)# # Just for display

        # Second part : evaluate the model at the current epoch
        model.eval()
        with torch.no_grad():
            val_total = 0
            for i, (X, A, S) in enumerate(val_loader):# Here, you must enumerate the different mini-batches, apply the estimated model, and evaluate the loss for display
                S_pred, A_pred = model(X)
                val_loss = torch.mean(((S_pred - S)**2).sum(axis=(-1,-2)) / ((S**2).sum(axis=(-1,-2))))
                val_loss += torch.mean(((A_pred - A)**2).sum(axis=(-1,-2)) / ((A**2).sum(axis=(-1,-2))))
                val_total += val_loss.item()
                
            val_total /= i+1
            val_total_loss.append(val_total)

		
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        if epoch % 5 == 0:
            print()

        print("epoch:{} | training loss:{:.5f} | validation loss:{:.5f} ".format(epoch, train_total,val_total))
        
        if optSave == True:# Nothing to change here
            torch.save(model,folderInitSave + '/LPALM_epoch_' + str(epoch)+'.pth')
            torch.save(val_total_loss,folderInitSave + '/val_total_loss_epoch_' + str(epoch)+'.pth')
            torch.save(train_total_loss,folderInitSave + '/train_total_loss_epoch_' + str(epoch)+'.pth')
		
	
    return train_total_loss, val_total_loss,model
