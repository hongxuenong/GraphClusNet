import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import networkx as nx
from utils import read_graphfile, prepare_data
from model import GraphClusNet
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import init_node_features, check_loss, corrdinate_to_index, compute_ncut
import yaml

torch.manual_seed(0)

with open("configs/config.yaml", "r") as f:
    config = yaml.load(f)

print("Reading graph files")
graphs, adj = read_graphfile(config['filepath'],
                             config['modulename'],
                             label_edge=False)
print("{} graphs loaded.".format(len(graphs)))
no_init = config['no_init']
print_loss = config['print_loss']
hierarchies = config['hierarchies']
num_run = config['num_run']

nmis = []
ncuts = []
nmis_lr = []
ncuts_lr = []

for run in range(num_run):
    for idx, G in enumerate(graphs):
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        G = S[0]
        # relabeling
        mapping = {}
        it = 0
        for n in G.nodes:
            mapping[n] = it
            it += 1
        # indexed from 0
        G = nx.relabel_nodes(G, mapping)
        x, edges, A, n_clusters, node_labels = prepare_data(G)

        x_norm = corrdinate_to_index(x)
        if (no_init):
            x_norm = init_node_features(G, method='random')

        node_labels = np.array([int(n) for n in node_labels])

        loss = 1
        best_nmi = 0
        iter = 0
        x_train = x_norm
        pred_labels = np.zeros((len(G.nodes), 3))
        for n_classes in hierarchies:
            print("n_classes:", n_classes)

            model = GraphClusNet(x_train.shape[1],
                                 gcn_layers=config['gcn_layers'],
                                 hidden_layers=config['hidden_layers'],
                                 hidden_dim=config['hidden_dim'],
                                 n_classes=n_classes,
                                 lr=config['lr'],
                                 weight_decay=0).cuda()

            loss = model.fit(x_train,
                             edges,
                             A,
                             epochs=config['epochs'],
                             dropout=config['dropout'],
                             print_loss=print_loss)
            x_new, s = model(x_train, edges)
            x_train = torch.cat(
                (x_norm, x_new.detach()), dim=1
            )  # to avoid the location information loss during iterative passing

            pred = torch.softmax(s, dim=-1)
            pred = np.array(torch.argmax(pred, 1).cpu())

            nmi = normalized_mutual_info_score(pred, node_labels)

            print("nmi:", nmi)
            node_labels = np.array(node_labels)
            del model

        ncut = compute_ncut(A, pred)
        print("ncut value:", ncut)

    nmis.append([nmi, ncut])

    ## Label refine
    x_train = init_node_features(G,
                                 method='inherit-correlation',
                                 correlation=1,
                                 pred=pred)
    model = GraphClusNet(x_train.shape[1],
                         gcn_layers=config['gcn_layers'],
                         hidden_layers=config['hidden_layers'],
                         hidden_dim=config['hidden_dim'],
                         n_classes=2,
                         lr=config['lr'],
                         weight_decay=1e-8).cuda()
    x_new, s, loss = model.fit(x_train,
                               edges,
                               A,
                               epochs=config['epochs'],
                               dropout=config['dropout'])
    pred = torch.softmax(s, dim=-1)
    pred = np.array(torch.argmax(pred, 1).cpu())
    nmi = normalized_mutual_info_score(pred, node_labels)

    print("refined nmi:", nmi)