import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import VGAE
from tqdm import tqdm
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

bceloss = nn.BCELoss()


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

    ncut_loss = 1 - torch.mean(ncut_loss)
    # ncut_loss = bceloss(ncut_loss, torch.ones_like(ncut_loss))
    return ncut_loss