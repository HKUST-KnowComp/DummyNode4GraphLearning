import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv, global_mean_pool as gap, global_max_pool as gmp 

class RGCN(torch.nn.Module):
    def __init__(self, args):
        super(RGCN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_relations = args.num_relations
                

        self.conv1 = RGCNConv(self.num_features, self.nhid, self.num_relations)
        self.conv2 = RGCNConv(self.nhid, self.nhid, self.num_relations)
        
        if args.additional:
            config = args.additional
            if 'weight_reg' in config and (config['weight_reg'] > 1.1):
                print('Dividing conv weights by {}'.format(config['weight_reg']))
                with torch.no_grad():
                    self.conv1.weight.div_(config['weight_reg'])
                    self.conv2.weight.div_(config['weight_reg'])
                    

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self.num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if edge_attr is not None:
            _, edge_type = edge_attr.max(dim=1)
        else:
            edge_type = torch.zeros([edge_index.size(1)]).to(x.device)
        
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = gap(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x
    
    

class RGIN(torch.nn.Module):
    def __init__(self, args):
        # Adapted from RGCN & GIN
        # https://github.com/diningphil/gnn-comparison/tree/master/models/graph_classifiers
        super(RGIN, self).__init__()
        from torch.nn import BatchNorm1d
        from torch.nn import Sequential, Linear, ReLU
        from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
        
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout = args.dropout_ratio
        self.num_relations = args.num_relations
        
        if args.additional:
            config = args.additional
        else:
            config = {'num_layers': 2}
        if config.get('aggregation', 'sum') == 'sum':
            self.pooling = global_add_pool
        elif config.get('aggregation', 'sum') == 'mean':
            self.pooling = global_mean_pool
        
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
                # modified to RGCN conv
                # self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))
                self.convs.append(RGCNConv(self.nhid, self.nhid, self.num_relations, aggr='add'))
                self.linears.append(Linear(out_emb_dim, self.num_classes))
        if ('weight_reg' in config) and (config['weight_reg'] > 1.1):
            print('Dividing conv weights by {}'.format(config['weight_reg']))
            with torch.no_grad():
                for conv in self.convs:
                    conv.weight.div_(config['weight_reg'])
        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)

    def forward(self, data):
        # get type related data
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if edge_attr is not None:
            _, edge_type = edge_attr.max(dim=1)
        else:
            edge_type = torch.zeros([edge_index.size(1)]).to(x.device)
            
        out = 0
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                x = self.convs[layer-1](x, edge_index, edge_type)
                # mlp
                x = self.nns[layer-1](x)
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training = self.training)
        out = F.log_softmax(out, dim=-1)
        return out