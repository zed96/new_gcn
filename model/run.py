import sys
sys.path.append('../')
from gcn.util.parameters import *
from gcn.util.utils import load_data, accuracy
from gcn.model.models import GCN, VGCN, VGCN_2,S_GCN, SV_GCN
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import  numpy as np
import random

# torch.manual_seed(1)#reproducible

cuda =False
kl_lr = 0.1
#标准化 ||x||**2 = 1
def norm(features):
    features_2 = features.mul(features)
    sum_row = torch.sum(features_2, dim=1)
    sum_row_1 = torch.sqrt(sum_row)
    return torch.div(features.t(), sum_row_1).t()



best_performance=[0.0, 0.0, 0.0]

def train(epoch, best_performance, use_vgcn=False):
    t = time.time()
    model.train()
    output = model(features, adj)
    # print('output的类型',type(output),output.shape)
    # print('labels的类型',type(labels),labels.shape)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    optimizer.zero_grad()
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj_2)
    #print('labels',labels)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    if use_vgcn==2:
        print('use_kl')
        kl_divergence = 0.5/output.size(0) * (1+ 2*model.logstd - model.mean**2 - torch.exp(model.logstd)).sum(1).mean()
        loss_val -= kl_lr*kl_divergence
    #print(output[idx_val])
    #print(output.shape)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    #print(output)
    #print(output.max(1)[1])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    test_acc = test()
    if test_acc > best_performance[2]:
        best_performance = [acc_train.item(), acc_val.item(), test_acc]
    return loss_val.item(), best_performance

def test():
    model.eval()
    output = model(features,adj_2)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output, labels)
    print("Test set results", "loss ={:.4f}".format(loss_test.item()),"accuracy={:.4f}".format(acc_test.item()))
    return acc_test.item()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(25)

if __name__ =='__main__':
    #load_data
    adj, features, labels, idx_train,idx_val,idx_test = load_data('cora')
    print(adj)
    print(adj.shape)
    print(features)

    print(features.shape)
    print(labels)
    print(labels.shape)
    adj, features, labels = torch.autograd.Variable(adj), torch.autograd.Variable(features), torch.autograd.Variable(labels)
    # adj_2 = adj.mm(adj).mm(adj)
    # features = norm(features)
    adj_2 =adj
    # Model and optimizer

    if use_vgcn ==3 :
        model = SV_GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=int(labels.max()) + 1,
                ncode= encode,
                n_dim = adj.shape[0],
                dropout=dropout)
    if use_vgcn==2:
        model = VGCN_2(nfeat=features.shape[1],
                nhid=hidden,
                nclass=int(labels.max()) + 1,
                ncode= encode,
                dropout=dropout)
    elif use_vgcn ==1:
        model = S_GCN(nfeat=features.shape[1],
                    nhid=hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=dropout,
                    n_dim=adj.shape[0])
    else:
        model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=int(labels.max()) + 1,
                dropout=dropout)
    print(model)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    if cuda:
        model.cuda()
        adj = adj.cuda()
        labels= labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()
    l1_train = []
    for epoch in range(200):
        l1_train.append(train(epoch, best_performance, use_vgcn))
        best_performance = l1_train[-1][-1]
    print("loss_list",l1_train)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print(l1_train[-1])


