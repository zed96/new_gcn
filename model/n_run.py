from gcn.util.parameters import *
from gcn.util.utils import load_data, accuracy
from gcn.model.models import GCN
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)#reproducible


#标准化 ||x||**2 = 1
def norm(features):
    features_2 = features.mul(features)
    sum_row = torch.sum(features_2, dim=1)
    sum_row_1 = torch.sqrt(sum_row)
    return torch.div(features.t(), sum_row_1).t()


def train(epoch):
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
        output = model(features, adj)
    #print('labels',labels)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
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
    return loss_val



if __name__ =='__main__':
    #load_data
    adj, features, labels, idx_train,idx_val,idx_test = load_data('cora')
    print(labels)
    # features = norm(features)
    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=int(labels.max()) + 1,
                dropout=dropout)
    print(model)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    # Train model
    t_total = time.time()
    l1_train = []
    for epoch in range(100):
        l1_train.append(train(epoch))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))