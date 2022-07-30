# coding: utf-8

import scipy
import scipy.io as io
import numpy as np
import torch
from torch_geometric.data import Data
from copy import deepcopy

def add_dummy_mat(mat_fn, output_mat_fn):
    # add dummy node to .mat file
    mat = io.loadmat(mat_fn)

    network = mat['network']
    group = mat['group']

    # add dummy to both network and group
    # to network
    tmp_network = network.toarray()
    row, _ = tmp_network.shape
    tmp_network = np.concatenate([tmp_network, np.ones([row, 1])], axis=1)
    tmp_network = np.concatenate([tmp_network, np.ones([1, row+1])], axis=0)
    tmp_network[row][row] = 0
    tmp_network = scipy.sparse.csc_matrix(tmp_network)

    # to group
    tmp_group = group.toarray()
    _, n_lbl = tmp_group.shape
    tmp_lbl = np.zeros([1, n_lbl])
    tmp_lbl[0][0] = 1   # randomly assign one for the dummy node
    tmp_group = np.concatenate([tmp_group, tmp_lbl], axis=0)
    tmp_group = scipy.sparse.csc_matrix(tmp_group)

    mat['network'] = tmp_network
    mat['group'] = tmp_group

    # save to file
    io.savemat(output_mat_fn, mat)

def add_dummy_PyG_TU_data(data):
    ''' Add dummy node to pytorch-geometric TUDataset data (a graph).
    the input data is of type torch_geometric.data.data.Data

    #deprecated# n_features is the number of features of each node. For example n_feature=1 for the PROTEINS dataset.

    This function should be set as the ''pre_transform'' parameter when calling ''TUDataset'' class.
    '''
    # [STEP1] MODIFY node features & node labels information
    # modify the data.x, which is a [n, a+b] tensor, 
    # where n is the total number of nodes, a is number of features and b is number of node labels.

    n, m = data.x.size(0), data.x.size(1)
    # n_node_lbls = m - n_features

    
    ''' Implementation-1 [[X 0],[0 1]]'''
#     x = torch.cat([torch.cat([data.x, torch.zeros([n, 1])], dim=1),\
#                     torch.zeros([1, m+1])], dim=0)
#     x[n][m] = 1
    
    ''' Implementatoin-2 [[X], [0]]'''
    x = torch.cat([data.x, torch.zeros([1, m])], dim=0)

    # [STEP2] MODIFY edge information
    # add bi-directional connection between dummy node and other nodes inside the graph.
    original_edge_size = data.edge_index.size(1)
    
    dummy_index = n
    from_inds = torch.empty([1, n], dtype=torch.int64)
    from_inds.fill_(n)
    to_inds = torch.arange(n, dtype=torch.int64).unsqueeze(0)

    from_to = torch.cat([from_inds, to_inds], dim=1)
    to_from = torch.cat([to_inds, from_inds], dim=1)
    concat_inds = torch.cat([from_to, to_from], dim=0)
    edge_index = torch.cat([data.edge_index, concat_inds], dim=1)
    
    ''' Default returns '''
#     return Data(edge_index=edge_index, x=x, y=data.y)

    ''' Additional information for distinguishing the dummy-edges '''
    is_dummy_edge = torch.arange(edge_index.size(1)) >= original_edge_size
    return Data(edge_index=edge_index, x=x, y=data.y, is_dummy_edge=is_dummy_edge)

if __name__ == '__main__':
    add_dummy_mat('example_graphs/blogcatalog.mat', 'example_graphs/blogcatalog_dummy.mat')
