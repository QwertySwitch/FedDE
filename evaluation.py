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
import ray
os.environ["RAY_DEDUP_LOGS"] = "0"


parser = argparse.ArgumentParser(description='FedDE: Federated Discriminative Ensemble for Multi-Source Domain Generalization')
parser.add_argument('--model', default='resnet18', type=str, help='model')
parser.add_argument('--dataset', default='pacs', type=str, help='dataset = [pacs, officehome, domainnet]')
parser.add_argument('--iid', default=1, type=int, help='dataset to iid or non-iid')
parser.add_argument('--num_users', default=3, type=int, help='The number of local clients')
parser.add_argument('--active_users', default=3, type=int, help='The number of local clients')
parser.add_argument('--out_dim', default=256, type=int, help='Feature output dimension')
parser.add_argument('--device', type=int, default=0, help='The device to run the program')
parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
parser.add_argument('--alg', type=str, default='pfedde', help='fraction of clients')
parser.add_argument('--target_domain', type=str, default='photo', help='Unseen Domain')
parser.add_argument('--edge', type=int, default=0, help='Unseen Domain')
parser.add_argument('--fourier', type=int, default=1, help='Unseen Domain')
parser.add_argument('--L2R_coeff', type=float, default=1e-2)
parser.add_argument('--CMI_coeff', type=float, default=5e-4)
args = parser.parse_args()


random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
ray.init(local_mode=True)

def main():
    print('\n[Phase 1] : Data Preparation')
    _, global_test_loader = get_dataset(args)

    print('\n[Phase 2] : Model setup')
    print(f'| Building net type [{args.model}]...')
    global_model = getNetwork(args)
    
    print(f'\n[Phase 3] : Evaluation Clients by {args.dataset}')
    
    evaluation(args, global_model, global_test_loader)

def evaluation(args, model, global_test_loader):
    domains = ['photo', 'art_painting', 'sketch', 'cartoon']
    clients = []
    for i in range(args.num_users):
        clients.append(copy.deepcopy(model))
    count = -1
    for i in range(len(domains)):
        if args.target_domain == domains[i]:
            continue
        
        count += 1
        clients[count].load_state_dict(torch.load('./save_model/a_{}_d_{}_s_{}_t_{}_m_{}.pth'.format(args.alg, args.dataset, domains[i], args.target_domain, args.model)))
        clients[count] = (domains[i], clients[count])
    count = -1
    for idx, test_loader in enumerate(global_test_loader):
        if domains[idx] == args.target_domain:
            continue
        count += 1
        acc1 = test_model(clients[count][1], domains[idx], args.target_domain, test_loader)
        acc2 = test_model(clients[count][1], domains[idx], args.target_domain, global_test_loader[domains.index(args.target_domain)])
        print('Source Domain: {} | Target Domain: {} | Source Domain Acc@1: {:.3f}%'.format(clients[count][0], args.target_domain, acc1))
        print('Source Domain: {} | Target Domain: {} | Target Domain Acc@1: {:.3f}%'.format(clients[count][0], args.target_domain, acc2))
        
def test_model(model, s, t, test_loader):
    model.eval()
    model.training = False
    test_loss = 0
    correct = 0
    total = 0
    model.to('cuda:2')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to('cuda:2'), targets.to('cuda:2')
            
            logits = model(inputs)
            
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        acc = 100.*correct/total
    model.cpu()
    return acc

if __name__ == '__main__':
    main()