import torch
import numpy as np
import os
import networkx as nx
from model import normalized_cut_loss_sparse
from torch_sparse import SparseTensor


def evaluate(G, pred, node_labels):

    class_size = {}
    for i in np.unique(node_labels):
        nodes = np.where(node_labels == i)[0]
        class_size[i] = len(nodes)
    print("class_size", class_size)

    # calculate prediction sizes
    pred_size = {}
    for i in np.unique(pred):
        nodes = np.where(pred == i)[0]
        pred_size[i] = len(nodes)
    print(pred_size)

    nodes = np.array(G.nodes)
    # print(np.where(pred==0))
    results = compute_metrics(nodes, node_labels, class_size, pred)

    return results


def compute_metrics(nodes, node_labels, class_size, pred):

    true_classes = np.unique(node_labels)
    pred_classes = np.unique(pred)
    # print(pred_classes,true_classes)

    class_nodes = {}
    for label in true_classes:
        class_nodes[label] = nodes[np.where(node_labels == label)]
    pred_nodes = {}
    for pred_label in pred_classes:
        pred_nodes[pred_label] = nodes[np.where(pred == pred_label)]

    pred_size = {a_pred: len(pred_nodes[a_pred]) for a_pred in pred_nodes}
    overlap = {
        a_class: {
            a_pred: len(
                set(class_nodes[a_class]).intersection(set(
                    pred_nodes[a_pred])))
            for a_pred in pred_nodes
        }
        for a_class in class_nodes
    }

    results = {}
    for a_class in true_classes:
        class_dict = overlap[a_class]
        max_pred = max(class_dict, key=class_dict.get)
        max_value = max(class_dict.values())
        #     print('Class', a_class, 'Max_pred', max_pred, 'Max_value', max_value)
        print('Class {}:'.format(a_class),
              '({} nodes)'.format(class_size[a_class]), 'is in Cluster',
              max_pred, '({} nodes)'.format(pred_size[max_pred]))
        precision = max_value / pred_size[max_pred]
        recall = max_value / class_size[a_class]
        f1 = 2 / (1 / recall + 1 / precision)
        print('Overlap:', max_value, 'Precision: {:.3f}'.format(precision),
              'Recall: {:.3f}'.format(recall), 'F1: {:.3f}\n'.format(f1))
        results[a_class] = f1

    return results


def read_graphfile(datadir, dataname, max_nodes=None, label_edge=False):
    """ Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    """
    prefix = os.path.join(datadir, dataname)
    filename_graph_indic = prefix + "_graph_indicator.txt"
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 0
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)

            i += 1

    filename_nodes = prefix + "_node_labels.txt"
    node_labels = []
    min_label_val = None
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                _l = int(line)
                node_labels += [_l]
                if min_label_val is None or min_label_val > _l:
                    min_label_val = _l
        # assume that node labels are consecutive
        # num_unique_node_labels = max(node_labels) - min_label_val + 1
        # node_labels = [l - min_label_val for l in node_labels]
    except IOError:
        print("No node labels")

    filename_node_attrs = prefix + "_node_attributes.txt"
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [
                    float(attr) for attr in line.split(',') if not attr == ""
                ]
                node_attrs.append(np.array(attrs))
    except IOError:
        print("No node attributes")

    filename_graphs = prefix + "_graph_labels.txt"
    graph_labels = []

    label_vals = []
    try:
        with open(filename_graphs) as f:
            for line in f:
                line = line.strip("\n")
                val = int(line)
                if val not in label_vals:
                    label_vals.append(val)
                graph_labels.append(val)
    except IOError:
        print("No  graph labels")
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[g] for g in graph_labels])

    if label_edge:
        # For Tox21_AHR we want to know edge labels
        filename_edges = prefix + "_edge_labels.txt"
        edge_labels = []

        edge_label_vals = []
        with open(filename_edges) as f:
            for line in f:
                line = line.strip("\n")
                val = int(line)
                if val not in edge_label_vals:
                    edge_label_vals.append(val)
                edge_labels.append(val)

        # edge_label_map_to_int = {
        #     val: i
        #     for i, val in enumerate(edge_label_vals)
        # }

    filename_adj = prefix + "_A.txt"
    adj_list = {i: [] for i in range(0, len(graph_labels))}

    # edge_label_list={i:[] for i in range(1,len(graph_labels)+1)}
    # index_graph = {i: [] for i in range(0, len(graph_labels))}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            # logfile.write("Reading edge: ({},{})\n".format(e0,e1))
            if (label_edge):
                edge_label = edge_labels[num_edges]
                adj_list[graph_indic[e0]].append((e0, e1, edge_label))
            else:
                adj_list[graph_indic[e0]].append((e0, e1))
            # index_graph[graph_indic[e0]] += [e0, e1]
            # edge_label_list[graph_indic[e0]].append(edge_labels[num_edges])
            num_edges += 1
    # for k in index_graph.keys():
    #     index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(0, len(adj_list)):
        # indexed from 0 here
        if (label_edge):
            G = nx.Graph()
            G.add_weighted_edges_from(adj_list[i])
        else:
            G = nx.from_edgelist(adj_list[i])

        # add features and labels
        G.graph["label"] = graph_labels[i]
        # Special label for aromaticity experiment
        # aromatic_edge = 2
        # G.graph['aromatic'] = aromatic_edge in edge_label_list[i]
        for u in G.nodes():
            if len(node_labels) > 0:
                # node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u]
                # node_label_one_hot[node_label] = 1
                G.nodes[u]["label"] = node_label
            if len(node_attrs) > 0:
                G.nodes[u]["feat"] = node_attrs[u]

        # relabeling
        mapping = {}
        it = 0
        for n in G.nodes:
            mapping[n] = it
            it += 1
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
        # dgraphs.append(nx.relabel_nodes(DG, mapping))
        # graphs.append(G)

    return graphs, adj_list


def init_node_features(G, method='random', correlation=None, pred=None):

    if (method == 'one-hot'):
        # one-hot-encoding ###
        x = torch.zeros((len(G.nodes), len(G.nodes))).cuda()
        for idx, n in enumerate(G.nodes):
            x[idx, idx] = 1
    elif (method == 'random'):
        x = torch.zeros((len(G.nodes), 100)).cuda().float()
        for idx, n in enumerate(G.nodes):
            # x[idx, np.random.choice(100, 50)] = 1
            x[idx, :] = torch.from_numpy(np.random.normal(size=100)).float()
    elif (method == 'node-degree'):
        x = torch.zeros((len(G.nodes), 1)).cuda()
        for idx, n in enumerate(G.nodes):
            d = G.degree[n]
            x[idx] = d
    elif (method == 'all-one'):
        x = torch.ones((len(G.nodes), len(G.nodes))).cuda()
    elif (method == 'node-degree-with-one-hot'):
        x = torch.zeros((len(G.nodes), 2500)).cuda()
        for idx, n in enumerate(G.nodes):
            d = G.degree[n]
            x[idx, d] = 1
    elif (method == 'cluster-correlation'):
        x = torch.zeros((len(G.nodes), 100)).cuda()
        node_labels = [G.nodes[n]['label'] for n in np.sort(G.nodes)]
        for idx, n in enumerate(G.nodes):
            feat = np.random.normal(0, 1, 100)

            if (correlation is None):
                correlation = 0
            p = np.random.uniform()
            if (p < correlation):
                feat[:25] = node_labels[n]

            x[idx, :] = torch.FloatTensor(feat)
    elif (method == 'inherit-correlation'):
        x = torch.zeros((len(G.nodes), 100)).cuda()
        node_labels = pred
        for idx, n in enumerate(G.nodes):
            feat = np.random.normal(0, 1, 100)

            if (correlation is None):
                correlation = 0
            p = np.random.uniform()
            if (p < correlation):
                feat[node_labels[n]:(node_labels[n] + 1) * 10] = 1

            x[idx, :] = torch.FloatTensor(feat)
    elif (method == 'label'):
        x = torch.zeros((len(G.nodes), 5)).cuda()

        for idx, n in enumerate(G.nodes):
            x[idx, pred[idx]] = 1
    elif (method == 'simulated-location'):
        # resolution=[1, 2, 3, 4]
        resolution = [30, 80, 150, 300]

        dim = len(resolution)
        x = torch.zeros(len(G.nodes), dim).cuda()
        for idx, n in enumerate(G.nodes):
            p = np.random.uniform()
            if (p < 0.7):
                for d, r in enumerate(resolution):
                    x[n, d] = idx // r
            else:
                x[n, :] = torch.from_numpy(np.random.normal(size=4)).float()
    return x


def prepare_data(G, label='label', feature='feat'):
    node_labels = [int(G.nodes[n][label]) for n in G.nodes]
    feat = nx.get_node_attributes(G, feature)
    if (feat and feature == 'feat'):
        # print("Assigning node features...")
        dim = feat[0].shape[0]
        x = torch.zeros((len(feat), dim))
        for n in feat:
            x[n] = torch.Tensor(feat[n])
        x = x.cuda()
    elif (feat and feature == 'Location'):
        # print("Assigning node features...")

        dim = len(feat[0].split(','))
        x = torch.zeros((len(feat), dim))

        for n in feat:
            locs = feat[n].split(',')
            loc_x = int(locs[0])
            loc_y = int(locs[1])
            x[n, 0] = loc_x
            x[n, 1] = loc_y
    else:
        print(" No node features. Initializing node features...")
        x = init_node_features(G)

    edges = np.array([[int(e[0]), int(e[1])] for e in G.edges])

    edges_v = np.array([[int(e[1]), int(e[0])] for e in G.edges])

    edges_all = np.vstack((edges, edges_v))
    edges = torch.LongTensor(edges_all.transpose()).cuda()

    adj_t = SparseTensor(row=edges[0],
                         col=edges[1],
                         sparse_sizes=(len(G.nodes), len(G.nodes)))

    n_clusters = len(np.unique(node_labels))
    return x, edges, adj_t, n_clusters, node_labels


def check_loss(adj, node_label):
    n_clusters = len(np.unique(np.array(node_label)))
    s = torch.zeros((len(node_label), n_clusters)).cuda()
    for n in range(s.shape[0]):
        s[n, node_label[n]] = 10

    loss = normalized_cut_loss_sparse(s,
                                      adj,
                                      EPS=1e-10,
                                      debug=False,
                                      do_softmax=True)
    return loss


def transform_distance(feat):
    print("largest coordinate:", torch.max(feat))
    dim = 66
    new_feat = torch.zeros((feat.shape[0], dim * 2)).cuda()
    for n in range(feat.shape[0]):
        loc_x = int(feat[n, 0])
        loc_y = int(feat[n, 1])

        if (loc_x + loc_y == 0):
            new_feat[n, :] = torch.FloatTensor(np.random.normal(0, 1, dim * 2))
        elif (loc_x == 512 or loc_y == 512):
            new_feat[n, :] = torch.FloatTensor(np.random.normal(0, 1, dim * 2))
        else:
            new_feat[n, :loc_x] = 1
            new_feat[n, dim:(dim + loc_y)] = 1

    return new_feat


def corrdinate_to_index(feat):
    # print("largest coordinate:", torch.max(feat))
    # remove memory cell:

    all_x = feat[:, 0]
    all_y = feat[:, 1]

    max_x = torch.max(all_x[torch.where(all_x < 512)])
    print("max x:", max_x)

    resolution = [3, 8, 12, 24]
    shift_steps = [0]
    dim = len(resolution) * len(shift_steps)
    new_feat = torch.zeros((feat.shape[0], dim)).cuda()
    for n in range(feat.shape[0]):
        loc_x = int(feat[n, 0])
        loc_y = int(feat[n, 1])
        for shift_step in shift_steps:
            if (loc_x + loc_y == 0):
                # new_feat[n, :] = torch.FloatTensor(np.random.normal(0, 1, dim))
                new_feat[n, :] = torch.FloatTensor(np.zeros(dim))
            elif (loc_x == 512 or loc_y == 512):
                # new_feat[n, :] = torch.FloatTensor(np.random.normal(0, 1, dim))
                new_feat[n, :] = torch.FloatTensor(np.zeros(dim))
            else:
                loc_x_new = loc_x + shift_step
                for idx, res in enumerate(resolution):
                    cell_index = loc_x_new // res + max_x // res * (loc_y //
                                                                    res)
                    new_feat[n,
                             idx * len(shift_steps) + shift_step] = cell_index

    return new_feat


def compute_ncut(adj, node_label, eps=1e-5):
    n_clusters = len(np.unique(np.array(node_label)))
    if (n_clusters == 1):
        ncut = 0
        cut = 0
        return ncut, cut
    s = torch.zeros((len(node_label), n_clusters)).cuda()

    for n in range(s.shape[0]):
        s[n, int(node_label[n])] = 1

    s_t = s.t()

    out_adj = s_t.matmul(adj.matmul(s))

    mincut_num = out_adj
    mincut_num = mincut_num.unsqueeze(
        0) if mincut_num.dim() == 2 else mincut_num

    mincut_num = mincut_num.cpu().numpy()

    ncut = 0
    for c in range(mincut_num.shape[1]):
        size = mincut_num[0, c, c]
        # print(size)
        if size < 5:
            continue
        cut = np.sum(
            np.array([
                mincut_num[0, c, i] for i in range(mincut_num.shape[1])
                if i != c
            ]))
        # print(cut)
        ncut += cut / (size + eps)

    return ncut
