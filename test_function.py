import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn

from data_process import load_graphs
from data_process import get_data_example
from data_process import load_dataset
from data_process import slice_graph
from data_process import convert_graphs

from Model.DySAT import DySAT
from Model.MLP import Classifier
from sklearn.metrics import f1_score, roc_auc_score

class _My_DGNN(torch.nn.Module):
    def __init__(self, args, in_feats = None):
        super(_My_DGNN, self).__init__()
        self.dgnn = DySAT(args, num_features = in_feats)
        self.classificer = Classifier(in_feature = self.dgnn.out_feats)
    def forward(self, graphs, nids):
        final_emb = self.dgnn(graphs)
        # print(nids)
        # get embeddings of nodes in the last graph
        emb = final_emb[-1]

        # get target embeddings
        source_id = nids[:, 0]
        target_id = nids[:, 1]
        source_emb = emb[source_id]
        target_emb = emb[target_id]
        input_emb = source_emb.mul(target_emb)
        # print(input_emb)
        return self.classificer(input_emb)

# TODO: complete the global forward
def run_dgnn_distributed(args):
    mp_group = args['mp_group']
    dp_gropu = args['dp_group']

    # load graphs
    # TODO: Unevenly slice graphs
    load_g, load_adj, load_feats = slice_graph(*load_graphs(args))
    num_graph = len(load_g)
    print("Loaded {}/{} graphs".format(num_graph, args['time_steps']))

    local_dataset = load_dataset(*get_data_example(load_g, args, num_graph))

    Dataset = Data.TensorDataset(
                torch.tensor(local_dataset['train_data']), 
                torch.FloatTensor(local_dataset['train_labels'])
            )
    loader = Data.DataLoader(
        dataset = Dataset,
        batch_size = 128,
        shuffle = True,
        num_workers=0,
    )

    # TODO: How to take the global forward?


def run_dgnn(args):
    r"""
    run dgnn with one process
    """
    device = args['device']

    # TODO: Unevenly slice graphs
    # load graphs
    load_g, load_adj, load_feats = slice_graph(*load_graphs(args))
    num_graph = len(load_g)
    print("Loaded {}/{} graphs".format(num_graph, args['time_steps']))

    dataset = load_dataset(*get_data_example(load_g, args, num_graph))

    train_dataset = Data.TensorDataset(
                torch.tensor(dataset['train_data']), 
                torch.FloatTensor(dataset['train_labels'])
            )
    loader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = 128,
        shuffle = True,
        num_workers=0,
    )

    # convert graphs to dgl or pyg graphs
    graphs = convert_graphs(load_g, load_adj, load_feats, args['data_str'])

    model = _My_DGNN(args, in_feats=load_feats[0].shape[1]).to(device)

    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    # results info
    epochs_f1_score = []
    epochs_auc = []
    epochs_acc = []

    # train
    for epoch in range (args['epochs']):
        Loss = []
        for step, (batch_x, batch_y) in enumerate(loader):
            model.train()
            print(batch_y)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            graphs = [graph.to(device) for graph in graphs]
            out = model(graphs, batch_x)
            # print(out.squeeze(dim=-1))
            loss = loss_func(out.squeeze(dim=-1), batch_y)
            # Loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(out)
        # test
        if epoch % args['test_freq'] == 0:
            # test_source_id = test_data[:, 0]
            # test_target_id = test_data[:, 1]
            model.eval()
            graphs = [graph.to(device) for graph in graphs]
            test_result = model(graphs, torch.tensor(dataset['test_data']).to(device))
            prob_f1 = []
            prob_auc = []
            prob_f1.extend(np.argmax(test_result.detach().cpu().numpy(), axis = 1))
            prob_auc.extend(test_result[:, -1].detach().cpu().numpy())
            ACC = sum(prob_f1 == dataset['test_labels'])/len(dataset['test_labels'])
            F1_result = f1_score(dataset['test_labels'], prob_f1)
            AUC = roc_auc_score(dataset['test_labels'], prob_auc)
            epochs_f1_score.append(F1_result)
            epochs_auc.append(AUC)
            epochs_acc.append(ACC)
            print("Epoch {:<3}, Loss = {:.3f}, F1 Score {:.3f}, AUC {:.3f}, ACC {:.3f}".format(epoch,
                                                                0,
                                                                F1_result,
                                                                AUC,
                                                                ACC))

    # print the training result info
    best_f1_epoch = epochs_f1_score.index(max(epochs_f1_score))
    best_auc_epoch = epochs_auc.index(max(epochs_auc))
    best_acc_epoch = epochs_acc.index(max(epochs_acc))

    print("Best f1 score epoch: {}, Best f1 score: {}".format(best_f1_epoch, max(epochs_f1_score)))
    print("Best auc epoch: {}, Best auc score: {}".format(best_auc_epoch, max(epochs_auc)))
    print("Best acc epoch: {}, Best acc score: {}".format(best_acc_epoch, max(epochs_acc)))








