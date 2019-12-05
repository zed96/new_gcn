import sys
sys.path.append('../')
from gcn.util.parameters import *
from gcn.util.utils import load_data, accuracy
from gcn.model.models import GCN, VGCN, VGCN_2
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import  numpy as np
import random

if __name__ =='__main__':
    #load_data
    adj, features, labels, idx_train,idx_val,idx_test = load_data('citeseer')
    print(labels)
    adj, features, labels = torch.autograd.Variable(adj), torch.autograd.Variable(features), torch.autograd.Variable(labels)
    # adj_2 = adj.mm(adj).mm(adj)
    print(adj)
    print(features)