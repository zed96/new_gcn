import torch.nn as nn
import torch.nn.functional as F
import  sys
sys.path.append('./')
from gcn.model.layers import GraphConvolution, Senet
import torch
import math

#for link prediction-> Reconstruct the adjacency matrix
#ADD senet | vae for link_prediction
class GCN_Link(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout):
        super(GCN_Link, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def dot_product_decode(self, Z):
        #Inner product
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    def forward(self, x, adj):
        #add senet
        x =F.relu(self.gc1(x, adj))
        x1= F.dropout(x, training=self.training)
        x= self.gc2(x1, adj)
        A_pred = self.dot_product_decode(x)
        return A_pred

class VGCN_Link(nn.Module):
    def __init__(self, nfeat,nhid,nclass,dropout):
        super(VGCN_Link, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.fc11 = GraphConvolution(nhid, nclass)
        self.fc12 = GraphConvolution(nhid, nclass)
        self.nclass = nclass
        self.dropout = dropout

    def dot_product_decode(self, Z):
        #Inner product
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    def encode(self, x, adj):
        hidden = F.relu(self.gc1(x, adj))
        # hidden = F.dropout(hidden, training= self.training)
        self.mean = self.fc11(hidden, adj)
        self.logstd = self.fc12(hidden, adj)
        eps = torch.randn(x.size(0), self.nclass)
        return eps * torch.exp(self.logstd) + self.mean

    def forward(self, x, adj):
        z = self.encode(x, adj)
        A_pred = self.dot_product_decode(z)
        return A_pred


class SV_GCN(nn.Module):
    def __init__(self, nfeat, nhid, ncode, nclass, n_dim, dropout):
        super(SV_GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.fc11 = GraphConvolution(nhid, nclass)
        self.fc12 = GraphConvolution(nhid, nclass)
        self.n_dim = n_dim
        self.senet_hid = senet_hid = int(self.n_dim // 3)
        self.senet = Senet(n_dim, senet_hid)
        self.nclass = nclass
        self.dropout = dropout

    def encode(self, x, adj):
        hidden = F.relu(self.gc1(x, adj))
        #use senet
        senet_input = torch.sum(x, dim=1, keepdim=True).t()
        print(senet_input.shape)
        senet_score = self.senet(senet_input).t()
        print(senet_score.shape)
        x = hidden * senet_score
        self.mean = self.fc11(x, adj)
        self.logstd = self.fc12(x, adj)
        eps = torch.randn(x.size(0), self.nclass)
        return eps * torch.exp(self.logstd) + self.mean

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return F.log_softmax(z, dim=1)

class VGCN_2(nn.Module):
    def __init__(self, nfeat, nhid, ncode, nclass, dropout):
        super(VGCN_2, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.fc11 = GraphConvolution(nhid, nclass)
        self.fc12 = GraphConvolution(nhid, nclass)
        self.nclass = nclass
        self.dropout = dropout

    def encode(self, x, adj):
        hidden = F.relu(self.gc1(x, adj))
        #hidden = F.dropout(hidden, training= self.training)
        self.mean = self.fc11(hidden, adj)
        self.logstd = self.fc12(hidden, adj)
        eps = torch.randn(x.size(0), self.nclass)
        return eps * torch.exp(self.logstd) + self.mean

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return F.log_softmax(z, dim=1)

class VGCN(nn.Module):
    def __init__(self, nfeat, nhid, ncode, nclass, dropout):
        super(VGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.fc11 = nn.Linear(nhid, ncode)
        self.fc12 = nn.Linear(nhid, ncode)
        self.fc2 = nn.Linear(ncode, nhid)
        self.gc2 = GraphConvolution(2*nhid, nclass)
        self.dropout = dropout
        self.ncode = ncode

    def encode(self, h1):
        return self.fc11(h1), self.fc12(h1)

    def reparametrize(self, mu, log_var):
        #std = torch.exp(log_var/2)
        #eps = torch.randn_like(std)
        std = torch.exp(log_var)
        eps = torch.randn(self.size, self.ncode)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc2(z))
        return h3

    def forward(self, x, adj):
        self.size = x.size(0)
        h1 = F.relu(self.gc1(x, adj))
        h1 = F.dropout(h1, training= self.training)
        #add vae
        mu, logvar = self.encode(h1)
        z = self.reparametrize(mu, logvar)
        x1 = self.decode(z)
        x1 = torch.cat([x1, h1], dim=1)
        x = self.gc2(x1, adj)
        return F.log_softmax(x, dim=1)

class GCN(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        #add senet
        x =F.relu(self.gc1(x, adj))
        x1= F.dropout(x, training=self.training)
        x= self.gc2(x1, adj)
        return F.log_softmax(x, dim=1)

class S_GCN(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout, n_dim):
        super(S_GCN,self).__init__()
        self.n_dim = n_dim
        self.senet_hid = senet_hid = int(self.n_dim//3)
        self.senet = Senet(n_dim, senet_hid)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        #add senet

        x =F.relu(self.gc1(x, adj))
        x1= F.dropout(x, training=self.training)
        senet_input = torch.sum(x, dim=1, keepdim=True).t()
        print(senet_input.shape)
        senet_score = self.senet(senet_input).t()
        print(senet_score.shape)
        x = x * senet_score
        x= self.gc2(x1, adj)
        return F.log_softmax(x, dim=1)

class H_GCN(nn.Module):
    def __init__(self,nfeat, nhid, nclass ,dropout, n_step):
        self.gc1 = [GraphConvolution(nfeat, nhid) for _ in range(n_step)]
        self.gc2 =  GraphConvolution(nhid*3, nclass)
        self.dropout = dropout
        self.step = n_step

    def forward(self, x, adjs):
        x = torch.cat([F.relu(gc_(x, adjs[i])) for i, gc_ in enumerate(self.gc1)], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adjs[0])
        return F.log_softmax(x, dim=1)
