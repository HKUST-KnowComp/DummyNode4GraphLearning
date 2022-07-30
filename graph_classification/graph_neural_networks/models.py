import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp 

from DiffPool import DiffPool
    
def get_last_index_in_graph(data):
    # get indices for the last node in each graph
    n = data.batch.size(0)
    if n > 1:
        tmp = torch.zeros_like(data.batch)
        tmp[:n-1] = data.batch[1:]
        is_last_in_graph = tmp != data.batch
        index_last_in_graph = is_last_in_graph.nonzero().squeeze()
    else:
        is_last_in_graph = torch.tensor([True])
        index_last_in_graph = is_last_in_graph.nonzero().squeeze(0)

    return index_last_in_graph

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        
        # dummy edge weight
        if args.dummy_weight > 0:
            self.dummy_weight = torch.tensor(args.dummy_weight, requires_grad=True, device=self.args.device)
            self.use_edge_weight = True
        else:
            self.use_edge_weight = False
            
        # additional transformation for nodes
        self.use_transform_dummy = False
        if args.transform_dummy != '':
            self.use_transform_dummy = True
            self.lin_dummy = [torch.nn.Linear(self.nhid, self.nhid)]
            if args.transform_dummy == 'no_activation':
                pass
            else:
                self.lin_dummy.append(torch.nn.ReLU())
            self.lin_dummy = torch.nn.Sequential(*self.lin_dummy)
                
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        if self.use_edge_weight:
            # when trainable dummy edge weight is enabled
            edge_attr = torch.ones(data.is_dummy_edge.size()).to(self.args.device)
            edge_attr[data.is_dummy_edge.to(self.args.device)] = self.dummy_weight
            
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # additional transformation for nodes
        if self.use_transform_dummy:
            # the last node in each graph is the dummy node (in that graph)
            index_last_in_graph = get_last_index_in_graph(data)
            x[index_last_in_graph] = self.lin_dummy(x[index_last_in_graph])

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = gap(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x
    
class GCN_concat_readout(torch.nn.Module):
    def __init__(self, args):
        super(GCN_concat_readout, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        
        # dummy edge weight
        if args.dummy_weight > 0:
            self.dummy_weight = torch.tensor(args.dummy_weight, requires_grad=True, device=self.args.device)
            self.use_edge_weight = True
        else:
            self.use_edge_weight = False
            
        # additional transformation for nodes
        self.use_transform_dummy = False
        if args.transform_dummy != '':
            self.use_transform_dummy = True
            self.lin_dummy = [torch.nn.Linear(self.nhid, self.nhid)]
            if args.transform_dummy == 'no_activation':
                pass
            else:
                self.lin_dummy.append(torch.nn.ReLU())
            self.lin_dummy = torch.nn.Sequential(*self.lin_dummy)
                
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        
        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        if self.use_edge_weight:
            # when trainable dummy edge weight is enabled
            edge_attr = torch.ones(data.is_dummy_edge.size()).to(self.args.device)
            edge_attr[data.is_dummy_edge.to(self.args.device)] = self.dummy_weight
            
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # additional transformation for nodes
        if self.use_transform_dummy:
            # the last node in each graph is the dummy node (in that graph)
            index_last_in_graph = get_last_index_in_graph(data)
            x[index_last_in_graph] = self.lin_dummy(x[index_last_in_graph])
        
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
    
class GraphSAGE(torch.nn.Module):
    def __init__(self, args):
        # Mostly follows the implementation from the paper "A Fail Comparison of Graph Neural Networks for Graph Classification"
        # https://github.com/diningphil/gnn-comparison/tree/master/models/graph_classifiers
        super(GraphSAGE, self).__init__()
        from torch_geometric.nn import SAGEConv
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        
        if args.additional:
            import json
            config = json.loads(args.additional) # "
        else:
            config = {'num_layers': 2,
                      'aggregation': 'mean'}
        if config.get('aggregation', 'mean') == 'max':
            self.fc_max = nn.Linear(self.nhid, self.nhid)
        num_layers = config.get('num_layers', 2)
        self.aggregation = config.get('aggregation', 'mean')
        
        # dummy edge weight
        if args.dummy_weight > 0:
            self.dummy_weight = torch.tensor(args.dummy_weight, requires_grad=True, device=self.args.device)
            self.use_edge_weight = True
        else:
            self.use_edge_weight = False
            
        # additional transformation for nodes
        self.use_transform_dummy = False
        if args.transform_dummy != '':
            self.use_transform_dummy = True
            self.lin_dummy = [torch.nn.Linear(self.nhid, self.nhid)]
            if args.transform_dummy == 'no_activation':
                pass
            else:
                self.lin_dummy.append(torch.nn.ReLU())
            self.lin_dummy = torch.nn.Sequential(*self.lin_dummy)
                
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = self.num_features if i==0 else self.nhid
            conv = SAGEConv(dim_input, self.nhid)
            conv.aggr = self.aggregation
            self.layers.append(conv)
        
        self.fc1 = nn.Linear(num_layers * self.nhid, self.nhid)
        self.fc2 = nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        if self.use_edge_weight:
            # when trainable dummy edge weight is enabled
            edge_attr = torch.ones(data.is_dummy_edge.size()).to(self.args.device)
            edge_attr[data.is_dummy_edge.to(self.args.device)] = self.dummy_weight
        
        # SAGE forward
        x_all = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)
        x = torch.cat(x_all, dim=1)
        x = gmp(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)
        return x
        
class GIN(torch.nn.Module):
    def __init__(self, args):
        # Mostly follows the implementation from the paper "A Fail Comparison of Graph Neural Networks for Graph Classification"
        # https://github.com/diningphil/gnn-comparison/tree/master/models/graph_classifiers
        super(GIN, self).__init__()
        from torch.nn import BatchNorm1d
        from torch.nn import Sequential, Linear, ReLU
        from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
        
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout = args.dropout_ratio
        
        if args.additional:
            import json
            config = json.loads(args.additional) 
        else:
            config = {'train_eps': False,
                      'num_layers': 2,
                      'aggregation': 'sum'}
        if config.get('aggregation', 'sum') == 'sum':
            self.pooling = global_add_pool
        elif config.get('aggregation', 'sum') == 'mean':
            self.pooling = global_mean_pool
        train_eps = config.get('train_eps', args.epochs)
        
        # dummy edge weight
        if args.dummy_weight > 0:
            self.dummy_weight = torch.tensor(args.dummy_weight, requires_grad=True, device=self.args.device)
            self.use_edge_weight = True
        else:
            self.use_edge_weight = False
            
        # additional transformation for nodes
        self.use_transform_dummy = False
        if args.transform_dummy != '':
            self.use_transform_dummy = True
            self.lin_dummy = [torch.nn.Linear(self.nhid, self.nhid)]
            if args.transform_dummy == 'no_activation':
                pass
            else:
                self.lin_dummy.append(torch.nn.ReLU())
            self.lin_dummy = torch.nn.Sequential(*self.lin_dummy)
        
        self.embeddings_dim = [self.nhid for _ in range(config.get('num_layers', 2))]
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        
        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(Linear(self.num_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                          Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, self.num_classes))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                           Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))
                self.linears.append(Linear(out_emb_dim, self.num_classes))
        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        if self.use_edge_weight:
            # when trainable dummy edge weight is enabled
            edge_attr = torch.ones(data.is_dummy_edge.size()).to(self.args.device)
            edge_attr[data.is_dummy_edge.to(self.args.device)] = self.dummy_weight
        
        out = 0
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                x = self.convs[layer-1](x, edge_index, edge_attr)
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training = self.training)
        out = F.log_softmax(out, dim=-1)
        return out