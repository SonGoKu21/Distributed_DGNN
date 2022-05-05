import copy
import os
import sys
import copy
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss

from Model.layers import StructuralAttentionLayer
from Model.layers import TemporalAttentionLayer


def _customized_embedding_comm(args, x, gate):
    r"""
    Gate: a [N, T] bool matrix, to identify the temporal dependecy for each snapshot
    """
    comm_method = 'gloo'
    # groups = args['mp_group']
    rank = args['rank']
    world_size = args['world_size']
    global_time_steps = args['time_steps']
    num_graph_per_worker = global_time_steps/world_size

    # re-construct the communication map according to the 'gate' matrix
    worker_list = torch.tensor(range(world_size))
    temporal_list = [worker_list[gate[:, i]].numpy() for i in range (global_time_steps)]
    # print('gate: ', gate)
    # print('temporal_list: ',temporal_list)
    mp_groups = [torch.distributed.new_group(
            ranks = temporal_list[i],
            backend = comm_method,
        ) for i in range (global_time_steps)
        ]

    comm_tensor = x.clone().detach()[:,0:1,:]
    result_list = []

    for i in range (global_time_steps):
        src = int(i//num_graph_per_worker)
        if rank < src:
            break
        if rank == src:
            comm_tensor = x.clone().detach()[:,int(i%num_graph_per_worker):int(i%num_graph_per_worker + 1),:]
        if len(temporal_list[i]) > 1 and rank in temporal_list[i]:
            # print('The {}-th graph need to be sent, sender {}, local rank {}'.format(i, int(src), rank))
            torch.distributed.broadcast(comm_tensor, src, group = mp_groups[i])
            if rank != src: # receive
                result_list.append(comm_tensor)
    
    if len(result_list) > 0:
        result_list.append(x)
        final = torch.cat(result_list, 1)
    else: final = x.clone()

    # print('rank: {} with fused tensor size {}'.format(rank, final.size()))

    return final


def _embedding_comm(args, x):
    mp_group = args['mp_group']
    rank = args['rank']
    world_size = args['world_size']

    comm_tensor = x.clone().detach()
    
    result_list = []
    for i in range (world_size - 1):
        if i > rank:
            break
        if rank <= i + 2:
            torch.distributed.broadcast(comm_tensor, i, group = mp_group[i])
        if i != rank:
            result_list.append(comm_tensor)
    if len(result_list) > 0:
        result_list.append(x)
        final = torch.cat(result_list, 1)
        # print('rank: {} with fused tensor {}'.format(rank, final))
    else: final = x.clone()

    return final


class DySAT(nn.Module):
    def __init__(self, args, num_features):
        '''
        Args:
            args: hyperparameters
            num_features: input dimension
            time_steps: total timesteps in dataset
            sample_mask: sample different snapshot graphs
            method: adding time interval information methods
        '''
        super(DySAT, self).__init__()
        structural_time_steps = args['structural_time_steps']
        temporal_time_steps = args['temporal_time_steps']
        args['window'] = -1
        self.args = args
        

        if args['window'] < 0:
            self.structural_time_steps = structural_time_steps # training graph per 'num_time_steps'
        else:
            self.structural_time_steps = min(structural_time_steps, args['window'] + 1)
        self.temporal_time_steps = temporal_time_steps
        self.num_features = num_features

        # network setting
        # self.structural_head_config = list(map(int, args.structural_head_config.split(","))) # num of heads per layer (structural layer)
        # self.structural_layer_config = list(map(int, args.structural_layer_config.split(","))) # embedding size (structural layer)
        # self.temporal_head_config = list(map(int, args.temporal_head_config.split(","))) # num of heads per layer (temporal layer)
        # self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(","))) # embedding size (temporal layer)
        # self.spatial_drop = args.spatial_drop
        # self.temporal_drop = args.temporal_drop

        self.structural_head_config = [8]
        self.structural_layer_config = [128]
        self.temporal_head_config = [8]
        self.temporal_layer_config = [128]
        self.spatial_drop = 0.1
        self.temporal_drop = 0.9
        self.out_feats = 128

        self.n_hidden = self.temporal_layer_config[-1]
        # self.method = method

        # construct layers
        self.structural_attn, self.temporal_attn = self.build_model()

        # loss function
        self.bceloss = BCEWithLogitsLoss()


    def forward(self, graphs, gate):
        # TODO: communicate the imtermediate embeddings after StructureAtt

        # Structural Attention forward
        structural_out = []
        for t in range(0, self.structural_time_steps):
            structural_out.append(self.structural_attn(graphs[t]))
        structural_outputs = [g.x[:,None,:] for g in structural_out] # list of [Ni, 1, F]

        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]
        out_dim = structural_outputs[-1].shape[-1]
        structural_outputs_padded = []
        for out in structural_outputs:
            zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]

        # print('rank: {} with tensor size {}'.format(self.args['rank'], structural_outputs_padded.size()))


        # Temporal Attention forward
        if self.args['connection']:
            comm_start = time.time()
            # exchange node embeddings
            if self.args['gate']:
                fuse_structural_output = _customized_embedding_comm(self.args, structural_outputs_padded, gate)
            else:
                fuse_structural_output = _embedding_comm(self.args, structural_outputs_padded)
            self.args['comm_cost'] += time.time() - comm_start
            # print('comm_cost in worker {} with time {}'.format(self.args['rank'], self.args['comm_cost']))
            temporal_time_start = time.time()
            temporal_out = self.temporal_attn(fuse_structural_output)
            self.args['temporal_cost'] += time.time() - temporal_time_start
        else: 
            temporal_time_start = time.time()
            temporal_out = self.temporal_attn(structural_outputs_padded)
            self.args['temporal_cost'] += time.time() - temporal_time_start


        return temporal_out

    # construct model
    def build_model(self):
        input_dim = self.num_features
        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(input_dim=input_dim,
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.args['residual'])
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]

        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(method=0,
                                           input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.temporal_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args['residual'],
                                           interval_ratio = self.args['interval_ratio'])
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        return structural_attention_layers, temporal_attention_layers

if __name__ == '__main__':
    _customized_embedding_comm()