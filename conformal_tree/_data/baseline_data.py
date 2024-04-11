from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
from sklearn import datasets

import torch
from torchvision import datasets as Dsets
from torchvision import transforms
import numpy as np
import os.path as osp
import json
import time 
import pathlib
from argparse import ArgumentParser



def get_mnist_np(path= "/home/kim2712/Desktop/data/MNIST/JK_np", train=True, n0 = 50000, device = None):
    type_ = "train" if train else "test"
    X, Y = np.load(path + F"/x_{type_}.npy"), np.load(path + F"/y_{type_}.npy")
    # I vectorize data to make the data compatible for other visualizers.
    sz = X.shape[0]
    X = X.reshape(sz, -1)

    if train: 
        X_train,Y_train = X[:n0], Y[:n0]         
        X_calib,Y_calib =   X[n0:], Y[n0:]
        if device is not None:
            X_train,Y_train = torch.tensor(X_train).to(device), torch.tensor(Y_train).to(device)
            X_calib,Y_calib = torch.tensor(X_calib).to(device), torch.tensor(Y_calib).to(device)
        return (X_train,Y_train), (X_calib,Y_calib)

    else:
        if device is not None:
            X,Y = torch.tensor(X).to(device), torch.tensor(Y).to(device)
        return X,Y


def get_data(args, logger=None, train=True, shuffle = True):
    '''
        In the case of dataset in [mnist, cifiar10,...] the dim argument is ignored. dim is for toy set up-dim.
        On the other hand, train, batch_size, argument if only effective for those [mnist, cifiar10,...] 
    '''
    
            
    dataset, dim = args.dataset,  args.dim
    
    if logger is None: 
        def logger(message):
            print(message)
    
    if dataset == "mnist": #dim is ignored
        
        data_dr, batch_size = args.data_dir, args.batch_size
        tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        ds = Dsets.MNIST(root=data_dr, train=train, download=True, transform=tf)
        
        if args.n_class <10: 
            if train: idx = ds.train_labels < args.n_class
            else: idx = ds.test_labels < args.n_class
            idx = np.arange(0, len(ds))[idx]
            ds = torch.utils.data.Subset(ds,idx)
            logger(F"Only {args.n_class} classes will be used for the dataset {dataset}")
        
        data_loader = torch.utils.data.DataLoader(ds,batch_size=batch_size, drop_last=True, shuffle=shuffle)
        
    elif dataset == "cifar": 
        data_dr, batch_size = args.data_dir, args.batch_size
        tf = transforms.Compose([transforms.ToTensor()])
        ds = Dsets.CIFAR10(root=data_dr, train=train, download=True, transform=tf)
        data_loader = torch.utils.data.DataLoader(ds,batch_size=batch_size, drop_last=True, shuffle=shuffle)
    
    return data_loader


