import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import VGAE
from tqdm import tqdm
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score


class GraphClusNet(torch.nn.Module):

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            # print(m)
            # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    def __init__(self,
                 in_dim,
                 gcn_layers=3,
                 hidden_layers=1,
                 hidden_dim=64,
                 n_classes=2,
                 lr=5e-4,
                 weight_decay=0):
        super(GraphClusNet, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, cached=True)
        gcnlayers = []
        for i in range(gcn_layers - 1):
            gcnlayers.append(GCNConv(hidden_dim, hidden_dim, cached=True))
        self.gcnlayers = nn.ModuleList(gcnlayers)

        self.hidden_layers = hidden_layers
        layers = []
        for i in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, n_classes))

        self.mlp = nn.ModuleList(layers)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, nesterov=True)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10)

    def forward(self, x, edge_index, dropout=True):

        x = F.relu(self.conv1(x, edge_index))
        for gcn in self.gcnlayers:
            if (dropout):
                x = F.dropout(x, p=0.7)
            x = F.elu(gcn(x, edge_index))
        s = x
        for i in range(self.hidden_layers):
            s = F.elu(self.mlp[i](s))

        return x, s

    def fit(self, x, edges, A, epochs, dropout=False, print_loss=False):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            x_new, s = self.forward(x, edges, dropout)

            ncut_loss = normalized_cut_loss_sparse(s, A, EPS=1e-20)
            # ncut_loss = normalized_cut_loss(s, A, EPS=1e-20)
            loss = ncut_loss
            loss.backward()
            self.optimizer.step()
            # if(epoch%200==0):
            #     self.scheduler.step(loss)
            #     print(self.optimizer.param_groups[0]["lr"])
            if (print_loss and epoch % 300 == 0):
                print('loss:{}'.format(loss))
        return x_new, s, loss

    def _batch_train(self, trainset, power, factor, epochs):
        self.train()
        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            losses = torch.zeros(len(trainset)).cuda()
            if (epoch == 99):
                nmis = torch.zeros(len(trainset)).cuda()
            for idx, graph in enumerate(trainset):
                G = graph[0]
                x = graph[1]
                edges = graph[2]
                A = graph[3]
                _, s = self(x, edges)
                wd_loss, d_ortho_loss, _ = density_loss(s,
                                                        A,
                                                        theta=1,
                                                        EPS=1e-10,
                                                        power=power,
                                                        factor=factor)
                loss = wd_loss + d_ortho_loss
                losses[idx] += loss

                if (epoch == 99):
                    node_labels = [
                        G.nodes[n]['label'] for n in np.sort(G.nodes)
                    ]
                    pred_clusters = torch.softmax(s, dim=-1)
                    pred_clusters = np.array(
                        torch.argmax(pred_clusters, 1).cpu())
                    nmis[idx] = normalized_mutual_info_score(
                        pred_clusters, node_labels)

            loss = torch.mean(losses)
            loss.backward()
            self.optimizer.step()
            if (epoch % 1 == 0):
                print('loss:', loss)
            if (epoch == 99):
                print("train accuracy:", torch.mean(nmis))


def normalized_cut_loss(s, adj, EPS=1e-10, debug=False, do_softmax=True):

    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s
    if (do_softmax):
        s = torch.softmax(s, dim=-1)
        # s = F.gumbel_softmax(s, dim=-1, hard=True)

    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    # MinCUT regularization.
    mincut_num = out_adj
    # d_flat = torch.einsum('ijk->ij', adj)
    # eye = torch.eye(d_flat.size(1)).type_as(d_flat)
    # d = eye * d_flat.unsqueeze(2).expand(*d_flat.size(), d_flat.size(1))

    # mincut_den = torch.einsum(
    #     'bii->bi', torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    # mincut_den = mincut_num / (mincut_den + EPS)
    ncut_loss = (torch.einsum(
        'bii->bi', mincut_num)) / (torch.sum(mincut_num, dim=-1) + EPS)

    # best
    # ncut_loss = 1 / torch.sum(ncut_loss)
    #
    # ncut_loss = torch.mean(1 / (ncut_loss + EPS))

    # paper loss
    ncut_loss = 1 - torch.mean(ncut_loss)

    if (debug):
        print('mincut_num:', mincut_num)
        print('mincut_num_sum:', torch.sum(mincut_num, dim=-1))
        print(
            'ncut_loss:',
            torch.einsum('bii->bi', mincut_num) /
            (torch.sum(mincut_num, dim=-1)))
        print('mincut_loss:', ncut_loss)
    return ncut_loss


bceloss = nn.BCELoss()


def normalized_cut_loss_sparse(s, adj, EPS=1, debug=False, do_softmax=True):

    if (do_softmax):
        s = torch.softmax(s, dim=-1)

    s_t = s.t()
    out_adj = s_t.matmul(adj.matmul(s))

    # MinCUT regularization.
    mincut_num = out_adj
    mincut_num = mincut_num.unsqueeze(
        0) if mincut_num.dim() == 2 else mincut_num

    ncut_loss = (torch.einsum(
        'bii->bi', mincut_num)) / (torch.sum(mincut_num, dim=-1) + EPS)
    # print(ncut_loss)
    # best
    # ncut_loss = 1 / torch.sum(ncut_loss)
    #
    # ncut_loss = torch.mean(1 / (ncut_loss + EPS))

    ncut_loss = 1 - torch.mean(ncut_loss)
    # ncut_loss = bceloss(ncut_loss, torch.ones_like(ncut_loss))
    # print(ncut_loss)
    if (debug):
        print('mincut_num:', mincut_num)
        print('mincut_num_sum:', torch.sum(mincut_num, dim=-1))
        print(
            'ncut_loss:',
            torch.einsum('bii->bi', mincut_num) /
            (torch.sum(mincut_num, dim=-1)))
        print('mincut_loss:', ncut_loss)
    return ncut_loss


def ratiocut_loss(s, adj, EPS=1e-10, debug=False, do_softmax=True):

    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s
    if (do_softmax):
        s = torch.softmax(s, dim=-1)
        # s = F.gumbel_softmax(s, dim=-1, hard=True)

    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    ss = torch.matmul(s.transpose(1, 2), s)
    c_size = torch.sum(ss, dim=-1)
    # MinCUT regularization.
    mincut_num = out_adj
    # d_flat = torch.einsum('ijk->ij', adj)
    # eye = torch.eye(d_flat.size(1)).type_as(d_flat)
    # d = eye * d_flat.unsqueeze(2).expand(*d_flat.size(), d_flat.size(1))

    # mincut_den = torch.einsum(
    #     'bii->bi', torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    # mincut_den = mincut_num / (mincut_den + EPS)
    ratiocut_loss = (torch.einsum('bii->bi', mincut_num)) / (c_size + EPS)

    mincut_loss = torch.mean(1 / (ratiocut_loss + EPS))

    return mincut_loss


def density_loss(s,
                 adj,
                 EPS=1e-10,
                 power=2,
                 factor=2,
                 debug=False,
                 do_softmax=True):

    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s
    if (do_softmax):
        s = torch.softmax(s, dim=-1)

    ss = torch.matmul(s.transpose(1, 2), s)
    c_size = torch.sum(ss, dim=-1)

    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    links = torch.einsum('bii->bi', out_adj)
    links_all = torch.einsum('bij->', out_adj)

    density = links / ((c_size + EPS)**power)

    with torch.no_grad():
        base = links_all / (torch.sum(c_size)**power)

    density_all = out_adj / ((c_size + EPS)**power)
    d_loss = torch.einsum('bii->bi',
                          density_all) / (torch.sum(density_all, dim=-1) + EPS)
    # d_loss = -torch.mean(torch.log(d_loss))
    d_loss = -torch.mean(d_loss)
    l_size = c_size**factor
    weights = l_size / (torch.sum(l_size))

    weighted_density = weights * density
    weighted_density = torch.sum(weighted_density)
    weighted_d_loss = -weighted_density / (base + EPS)
    if (debug):
        print('c_size', c_size)
        print('l_size', l_size)
        print('weights:', weights)
        print('density:', density)
        print('weighted_density:', weighted_density)
        print('density_all:', density_all)
        print(
            'd_loss:',
            torch.einsum('bii->bi', density_all) /
            (torch.sum(density_all, dim=-1) + EPS))
    return weighted_d_loss, d_loss, weighted_density
