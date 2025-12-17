#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random
import os
import torch
from collections import Counter

transform_cifar = transforms.Compose([transforms.Resize(32),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def noniid2(name, dataset, num_users):
    path = f'../data/{name}-One/TwoClass/'
    trainpath = os.path.join(path, 'train.pt')
    if os.path.isfile(trainpath):
        trainsets = torch.load(trainpath)
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = trainsets[i]
    return dict_users

def noniid(name, dataset, num_users):
    path = f'../data/{name}-dirichlet/alpha0.1_seed42/'
    trainpath = os.path.join(path, f'train_{num_users}.pt')
    #valpath = os.path.join(path, 'val.pt')
    trainsets = torch.load(trainpath)
    return trainsets

def iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def sampling_num_data_iid(dataset, num, num_users):
    num_items = int(num/len(set(dataset.labels)))
    dict_users, all_idxs = {i:set([]) for i in range(num_users)}, [np.where(np.array(dataset.labels) == j)[0] for j in range(len(set(dataset.labels)))]
    for i in range(num_users):
        for j in range(len(set(dataset.labels))):
            dict_users[i] = set.union(dict_users[i],set(np.random.choice(all_idxs[j], num_items, replace=True)))
            all_idxs[j] = list(set(all_idxs[j]) - dict_users[i])
    return dict_users

def sampling_num_data_iid(dataset, num, num_users, labels):
    num_items = int(num/len(set(dataset.labels)))
    dict_users, all_idxs = {i:set([]) for i in range(num_users)}, [np.where(np.array(dataset.labels) == j)[0] for j in range(len(set(dataset.labels)))]
    for i in range(num_users):
        for j in range(len(set(dataset.labels))):
            if j in labels:
                dict_users[i] = set.union(dict_users[i],set(np.random.choice(all_idxs[j], num_items, replace=True)))
                all_idxs[j] = list(set(all_idxs[j]) - dict_users[i])
    return dict_users

def sampling_data_class(dataset, num, labels):
    dict_user = {}
    all_idxs = [np.where(np.array(dataset.targets) == j)[0] for j in range(len(set(dataset.targets)))]
    for j in range(len(set(dataset.targets))):
        if j in labels:
            dict_user = set.union(set(dict_user), set(all_idxs[j]))
    return dict_user