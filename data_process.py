import os
import scipy
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

import dgl
import torch
import torch_geometric as pyg

from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

current_path = os.getcwd()

def _generate_feats(adjs, time_steps):
    feats = [scipy.sparse.identity(adjs[time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[time_steps - 1].shape[0]]
    new_features = []
    for feat in feats:
        rowsum = np.array(feat.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.  # inf -> 0
        r_mat_inv = sp.diags(r_inv)
        new_features.append(r_mat_inv.dot(feat).todense())
    assert time_steps <= len(adjs), "Time steps is illegal"
    return new_features

def _normalize_graph_gcn(adj):
    r"""GCN-based normalization of adjacency matrix 
    (scipy sparse format). Output is in tuple format
    """
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
    rowsum = np.array(adj_.sum(1), dtype=np.float32)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def _build_pyg_graphs(features, adjs):
    pyg_graphs = []
    for feat, adj in zip(features, adjs):
        x = torch.Tensor(feat)
        edge_index, edge_weight = pyg.utils.from_scipy_sparse_matrix(adj)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        pyg_graphs.append(data)
    return pyg_graphs

def _build_dgl_graphs(graphs, features):
    dgl_graphs = []  # graph list
    for graph, feat in zip(graphs, features):
        x = torch.Tensor(feat)
        dgl_graph = dgl.from_networkx(graph)
        dgl_graph = dgl.add_self_loop(dgl_graph)
        dgl_graph.ndata['feat'] = x
        dgl_graphs.append(dgl_graph)
    return dgl_graphs

def _create_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.2):
    r"""
    Generate postive and negative edges from next_graph (i.e., last graph)
    1. Postive edges: the edges in the next_graph while both the source and target nodes exist in 'graph' (i.e., previous graph)
    2. Negative edges: the edges in the next_graph but the source and target nodes do not exist in 'graph' (i.e., previous graph)
    """
    edges_next = np.array(list(nx.Graph(next_graph).edges()))
    print("total data:", len(edges_next))
    edges_positive = []   # Constraint to restrict new links to existing nodes.
    Num_of_edges = 1000
    for idx, e in enumerate(edges_next):
        if graph.has_node(e[0]) and graph.has_node(e[1]) and idx <= Num_of_edges:
        # if next_graph.has_edge(e[0], e[1]) and idx <= Num_of_edges:
            edges_positive.append(e)
    edges_positive = np.array(edges_positive) # [E, 2]
    edges_negative = _negative_sample(edges_positive, graph.number_of_nodes(), next_graph)

    # split the edges into train, val, and test sets
    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive,
            edges_negative, test_size=val_mask_fraction+test_mask_fraction)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos,
            test_neg, test_size=test_mask_fraction/(test_mask_fraction+val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg

def _negative_sample(edges_pos, nodes_num, next_graph):
    edges_neg = []
    while len(edges_neg) < len(edges_pos):
        idx_i = np.random.randint(0, nodes_num)
        idx_j = np.random.randint(0, nodes_num)
        if idx_i == idx_j:
            continue
        if next_graph.has_edge(idx_i, idx_j) or next_graph.has_edge(idx_j, idx_i):
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg

def load_graphs(args):
    r"""
    Load graphs with given the dataset name
    param:
        dataset_name: dataset's name
        platform: converse graph to which platform. dgl or pyg
        time_steps: the num of graphs for experiments
        features (bool): whether generate features with one-hot encoding
        graph_id: which graphs should be loaded
    """
    dataset_name = args['dataset']
    time_steps = args['time_steps']
    features = args['featureless']

    new_graphs = []
    # load networkx graphs data
    graph_path = current_path + '/Data/{}/data/{}'.format(dataset_name, 'graphs.npz')
    if dataset_name == 'Enron':
        with open(graph_path, "rb") as f:
            graphs = pkl.load(f)
    else:
        graphs = np.load(graph_path, allow_pickle=True, encoding='latin1')['graph']
    graphs = graphs[1:time_steps+1]
    adj_matrices = list(map(lambda x: nx.adjacency_matrix(x), graphs))
    # print("Loaded {} graphs ".format(len(graphs)))

    #normlized adj
    adj_matrices = [_normalize_graph_gcn(adj) for adj in adj_matrices]

    # generate features
    if features:
        feats = _generate_feats(adj_matrices, time_steps)

    return (args, graphs, adj_matrices, feats)

def get_data_example(graphs, args, local_time_steps):
    r"""
    Generate train/val/test samples to evaluate link prediction performance
    1. train_edges/train_edges_false: training samples
    2. val_edges/val_edges_false: validation samples
    3. test_edges/test_edges_false: test samples
    Note: the node embeddings for each edge is generated from the DySAT model
    """
    rank = args['rank']
    dataset = args['dataset']
    framework = args['data_str']

    eval_idx = local_time_steps - 2
    eval_graph = graphs[eval_idx]
    next_graph = graphs[eval_idx+1]

    eval_path = current_path + "/Data/{}/data/eval_{}_worker{}.npz".format(dataset, str(eval_idx), rank)
    try:
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
        print("Loaded classification evaluation data!")
    except IOError:
        print("Generating and saving eval data ....")
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            _create_data_splits(eval_graph, next_graph, val_mask_fraction=0.1,
                            test_mask_fraction=0.3)
        np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false,
                                           test_edges, test_edges_false]))
    
    print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
          len(train_edges), len(train_edges_false), len(val_edges), len(val_edges_false),
          len(test_edges), len(test_edges_false)))

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def load_dataset(train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg):
    train_pos_labels = np.array([[1,0] for i in range(len(train_edges_pos))])
    train_neg_labels = np.array([[0,1] for i in range(len(train_edges_neg))])
    val_pos_labels = np.array([[1,0] for i in range(len(val_edges_pos))])
    val_neg_labels = np.array([[0,1] for i in range(len(val_edges_neg))])
    test_pos_labels = np.array([0] * len(test_edges_pos))
    test_neg_labels = np.array([1] * len(test_edges_neg))

    dataset = {}

    # combine positive and negative data together for prediction
    dataset['train_data'] = np.vstack((train_edges_pos, train_edges_neg))  # train_pos_feats and train_neg_feats are 2-dim numpy matrix, stack them to a new numpy matrix via vstack()
    dataset['train_labels'] = np.vstack((train_pos_labels, train_neg_labels))  # train_pos_labels and train_neg_labels are 1-dim numpy array
    dataset['val_data'] = np.vstack((val_pos_labels, val_neg_labels))
    dataset['val_labels'] = np.append(val_pos_labels, val_neg_labels)
    dataset['test_data'] = np.vstack((test_edges_pos, test_edges_neg))
    dataset['test_labels'] = np.append(test_pos_labels, test_neg_labels)

    return dataset

def convert_graphs(graphs, adj, feats, framework):
    # converse nx-graphs to dgl-graph or pyg-graph
    for graph in graphs:
        if framework == 'dgl':
            new_graphs = _build_dgl_graphs(graphs, feats)
        elif framework == 'pyg':
            new_graphs = _build_pyg_graphs(feats, adj)

    return new_graphs

def slice_graph(args, graphs, adj, feats):
    world_size = args['world_size']
    rank = args['rank']
    num_parts = len(graphs) // world_size

    sliced_graphs = graphs[rank*num_parts: (rank+1)*num_parts]
    sliced_adj = adj[rank*num_parts: (rank+1)*num_parts]
    sliced_feats = feats[rank*num_parts: (rank+1)*num_parts]

    return sliced_graphs, sliced_adj, sliced_feats
