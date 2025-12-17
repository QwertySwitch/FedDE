import torch
import torch.nn as nn
import copy
import random
import os
import sys
import time
import argparse
import numpy as np
from utils import *
import logging
from algorithm.fedde import fedde
from algorithm.fedavg import fedavg
from algorithm.fedsr import fedsr
from algorithm.feddc import feddc
os.environ["RAY_DEDUP_LOGS"] = "0"
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='FedDE: Federated Discriminative Ensemble for Multi-Source Domain Generalization')

# Federated Learning Hyperparameters
parser.add_argument('--lr', default=1e-3, type=float, help='learning_rate')
parser.add_argument('--model', default='resnet18', type=str, help='model')
parser.add_argument('--segmentation', type=bool, default=False)
parser.add_argument('--dataset', default='pacs', type=str, help='dataset = [pacs, officehome, domainnet, fundus, prostate]')
parser.add_argument('--target_domain', type=str, default='photo', help='Unseen Domain')
parser.add_argument('--logdir', default='./logs', type=str, help='log file path')
parser.add_argument('--iid', default=1, type=int, help='dataset to iid or non-iid')
parser.add_argument('--num_users', default=3, type=int, help='The number of local clients')
parser.add_argument('--active_users', default=3, type=int, help='The number of local clients')
parser.add_argument('--epoch', default=5, type=int, help='Number of epochs of each local models')
parser.add_argument('--round', default=100, type=int, help='Number of training rounds')
parser.add_argument('--out_dim', default=256, type=int, help='Feature output dimension')
parser.add_argument('--device', type=int, default=0, help='The device to run the program')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--frac', type=float, default=1.0, help='fraction of clients')
parser.add_argument('--alg', type=str, default='fedde', help='fraction of clients')
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--disc', type=float, default=1.0)

# FedSR Hyperparameters
parser.add_argument('--L2R_coeff', type=float, default=1e-2)
parser.add_argument('--CMI_coeff', type=float, default=5e-4)
parser.add_argument('--D_beta', type=float, default=0.1)
parser.add_argument('--z_dim', type=int, default=1024)
parser.add_argument('--mix', type=bool, default=False)

args = parser.parse_args()


random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def main():
    print('\n[Phase 1] : Data Preparation')
    train_loaders, global_test_loader, domains = get_dataset(args)

    print('\n[Phase 2] : Model setup')
    print(f'| Building net type [{args.model}]...')
    
    local_models, global_model, non_global_model = getNetwork(args)
    
    if args.segmentation:
        task_type = 'Segmentation'
    else:
        task_type = 'Classification'
        
    print('| Training Rounds = ' + str(args.round))
    print('| Training Epochs = ' + str(args.epoch))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + str(args.optim))
    print('| Task Type = ' + str(task_type))
    print(f'\n[Phase 3] : Training Clients by {args.dataset}')
    
    if args.alg == 'fedde':
        fedde(args, local_models, global_model, non_global_model, train_loaders, global_test_loader, domains)
    elif args.alg == 'feddc':
        feddc(args, local_models, global_model, non_global_model, train_loaders, global_test_loader, domains)
    elif args.alg == 'fedavg':
        fedavg(args, global_model, train_loaders, global_test_loader)
    elif args.alg == 'fedsr':
        fedsr(args, global_model, train_loaders, global_test_loader)

if __name__ == '__main__':
    main()