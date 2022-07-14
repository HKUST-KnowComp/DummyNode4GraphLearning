import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import GraphAdjModel
from .container import *
# from ..constants import *
# from ..utils.act import map_activation_str_to_layer
# from ..utils.init import init_weight, init_module
from constants import *
from utils.act import map_activation_str_to_layer
from utils.decomp import create_decomposed_weights
from utils.init import init_weight, init_module


class DecompMultiTransform(nn.Module):
    def __init__(self, input_dim, output_dim, num_transforms, regularizer="basis", num_bases=-1, bias=False):
        super(DecompMultiTransform, self).__init__()

        assert regularizer in ["none", "basis", "bdd", "diag", "scalar"]
        if num_bases <= 0:
            regularizer = "none"
            num_bases = -1

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_transforms = num_transforms
        self.regularizer = regularizer
        self.num_bases = num_bases

        self.weights = create_decomposed_weights(regularizer, input_dim, output_dim, num_transforms, num_bases)

        if bias:
            self.bias = nn.Parameter(th.Tensor(output_dim))
        else:
            self.register_parameter("bias", None)

        # init
        if bias:
            nn.init.zeros_(self.bias)

        self.weights = ParameterDict(self.weights)

    def forward(self, x, xtype):
        x_size = x.size()

        if self.regularizer == "none":
            x = x.view(-1, 1, self.input_dim)
            if xtype.dtype == th.long:
                w = self.weights["weight"].index_select(0, xtype.view(-1))
            else:
                w = th.matmul(xtype.view(-1, self.num_transforms), self.weights["weight"])
            w = w.view(-1, self.input_dim, self.output_dim)
            x = th.bmm(x, w)
        elif self.regularizer == "basis":
            x = x.view(-1, 1, self.input_dim)
            if x.size(0) >= self.num_transforms:
                # method 1
                # computation cost: num_trans * num_bases * (input * output) + bsz * (input * output)
                weight = th.matmul(self.weights["w_comp"], self.weights["weight"])
                if xtype.dtype == th.long:
                    w = weight.index_select(0, xtype.view(-1))
                else:
                    w = th.matmul(xtype.view(-1, self.num_transforms), weight)
                w = w.view(-1, self.input_dim, self.output_dim)
                x = th.bmm(x, w)
            else:
                # method 2
                # computation cost: bsz * num_bases * (input * output) + bsz * (input * output)
                if xtype.dtype == th.long:
                    w_c = self.weights["w_comp"].index_select(0, xtype.view(-1))
                else:
                    w_c = th.matmul(xtype.view(-1, self.num_transforms), self.weights["w_comp"])
                w = th.matmul(w_c, self.weights["weight"])
                w = w.view(-1, self.input_dim, self.output_dim)
                x = th.bmm(x, w)
        elif self.regularizer == "bdd":
            if self.num_bases > 0:
                submat_in = self.input_dim // self.num_bases
                submat_out = self.output_dim // self.num_bases
            else:
                submat_in = self.input_dim // self.num_transforms
                submat_out = self.output_dim // self.num_transforms
            x = x.view(-1, 1, submat_in)

            if xtype.dtype == th.long:
                w = self.weights["weight"].index_select(0, xtype.view(-1))
            else:
                w = th.matmul(xtype.view(-1), self.weights["weight"])
            w = w.view(-1, submat_in, submat_out)
            x = th.bmm(x, w)
        elif self.regularizer == "diag" or self.regularizer == "scalar":
            x = x.view(-1, self.input_dim)
            if xtype.dtype == th.long:
                w_c = self.weights["w_comp"].index_select(0, xtype.view(-1))
            else:
                w_c = th.matmul(xtype.view(-1, self.num_transforms), self.weights["w_comp"])
            w = th.matmul(w_c, self.weights["weight"])
            w = w.view(x.size(0), -1)
            x = th.mul(x, w)
        else:
            raise NotImplementedError

        x = x.view(x_size[:-1] + (-1, ))

        if self.bias is not None:
            x = x + self.bias

        return x

    def get_output_dim(self):
        return self.output_dim

    def extra_repr(self):
        summary = [
            "in=%d, out=%d, num_transforms=%s," % (self.input_dim, self.output_dim, self.num_transforms),
            "regularizer=%s, num_bases=%d," % (self.regularizer, self.num_bases),
        ]

        return "\n".join(summary)


class HeteroGraphTransLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_node_types=1,
        num_edge_types=1,
        regularizer="basis",
        num_bases=-1,
        num_heads=1,
        self_loop=True,
        bias=True,
        batch_norm=False,
        act_func="relu",
        dropout=0.0
    ):
        super(HeteroGraphTransLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.total_rel = num_node_types * num_edge_types * num_node_types
        self.regularizer = regularizer
        self.num_heads = num_heads
        self.self_loop = self_loop
        self.scale = (hidden_dim / num_heads) ** -0.5

        if regularizer == "none":
            self.num_bases = -1
        elif regularizer in ["diag", "scalar"]:
            self.num_bases = 1
        else:
            self.num_bases = num_bases

        self.k_transform = DecompMultiTransform(input_dim, hidden_dim, num_node_types, regularizer, num_bases, False)
        self.q_transform = DecompMultiTransform(input_dim, hidden_dim, num_node_types, regularizer, num_bases, False)
        self.v_transform = DecompMultiTransform(input_dim, hidden_dim, num_node_types, regularizer, num_bases, False)
        self.a_transform = DecompMultiTransform(input_dim, hidden_dim, num_node_types, regularizer, num_bases, False)

        d_k = hidden_dim // num_heads
        self.relation_pri = nn.Parameter(th.ones(num_edge_types, num_heads))
        self.relation_att = nn.Parameter(th.Tensor(num_edge_types, num_heads, d_k, d_k))
        self.relation_msg = nn.Parameter(th.Tensor(num_edge_types, num_heads, d_k, d_k))

        if self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(self.input_dim, self.hidden_dim))
        else:
            self.register_parameter("loop_weight", None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(self.hidden_dim))
        else:
            self.register_parameter("bias", None)

        if batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = None

        self.act = map_activation_str_to_layer(act_func)
        self.drop = nn.Dropout(dropout)

        # init
        init_weight(self.relation_att, activation=act_func, init="uniform")
        init_weight(self.relation_msg, activation=act_func, init="uniform")
        if self_loop:
            init_weight(self.loop_weight, activation=act_func, init="uniform")
        if bias:
            nn.init.zeros_(self.bias)

        # register functions
        self.node_init_func = self._node_init_func
        self.edge_init_func = self._edge_init_func
        self.node_message_func = self._node_message_func
        self.node_reduce_func = fn.sum(msg=NODEMSG, out=NODEAGG)
        self.node_update_func = self._node_update_func
        self.edge_update_func = None

    def _node_init_func(self, graph, node_feat=None):
        if isinstance(node_feat, dict):
            for ntype in graph.ntypes:
                nf = node_feat[ntype]
                num_nt = nf.size(0) if nf is not None else 0
                if num_nt == 0:
                    continue
                nt = th.tensor(int(ntype), device=nf.device, dtype=th.long).unsqueeze(0).expand(num_nt, -1)
                k = self.k_transform(nf, nt).view(num_nt, self.num_heads, -1)
                v = self.v_transform(nf, nt).view(num_nt, self.num_heads, -1)
                q = self.q_transform(nf, nt).view(num_nt, self.num_heads, -1)
                graph.nodes[ntype].data.update({"k_" + NODEFEAT: k, "v_" + NODEFEAT: v, "q_" + NODEFEAT: q})
                if self.self_loop:
                    h_loop = th.matmul(nf, self.loop_weight)
                    graph.nodes[ntype].data.update({"loop_" + NODEFEAT: h_loop})
        elif isinstance(node_feat, th.Tensor):
            node_type = None
            if NODETYPE in graph.ndata:
                node_type = NODETYPE
            elif NODELABEL in graph.ndata:
                node_type = NODELABEL
            else:
                node_type = NODETYPE

            num_nodes = node_feat.size(0)
            nt = graph.ndata[node_type]
            k = self.k_transform(node_feat, nt).view(num_nodes, self.num_heads, -1)
            v = self.v_transform(node_feat, nt).view(num_nodes, self.num_heads, -1)
            q = self.q_transform(node_feat, nt).view(num_nodes, self.num_heads, -1)
            graph.ndata.update({"k_" + NODEFEAT: k, "v_" + NODEFEAT: v, "q_" + NODEFEAT: q})
            if self.self_loop:
                h_loop = th.matmul(node_feat, self.loop_weight)
                graph.ndata.update({"loop_" + NODEFEAT: h_loop})
        else:
            raise NotImplementedError

        return node_feat

    def _edge_init_func(self, graph, edge_feat=None):
        if len(graph.canonical_etypes) == 1:
            edge_type = None
            if EDGETYPE in graph.edata:
                edge_type = EDGETYPE
            elif EDGELABEL in graph.edata:
                edge_type = EDGELABEL
            else:
                edge_type = EDGETYPE

            et = graph.edata[edge_type]
            r_att = self.relation_att[et]
            r_pri = self.relation_pri[et]
            # r_msg = self.relation_msg[et]

            graph.apply_edges(
                lambda edges: {
                    "e": edges.dst["q_" + NODEFEAT] * th.einsum("bij,bijk->bik", edges.src["k_" + NODEFEAT], r_att)
                }
            )
            e = graph.edata.pop("e").sum(dim=-1) * r_pri * self.scale
            a = dgl.ops.edge_softmax(graph, e)
            graph.edata[NORM] = a.unsqueeze(-1)
            # graph.edata[EDGEMSG] = r_msg
        else:
            for srctype, etype, dsttype in graph.canonical_etypes:
                eg = graph[srctype, etype, dsttype]
                et = int(etype)
                r_att = self.relation_att[et]
                r_pri = self.relation_pri[et]
                # r_msg = self.relation_msg[et]

                eg.apply_edges(
                    lambda edges: {
                        "e": edges.dst["q_" + NODEFEAT] * th.einsum("bij,ijk->bik", edges.src["k_" + NODEFEAT], r_att)
                    }
                )
                e = eg.edata.pop("e").sum(dim=-1) * r_pri * self.scale
                a = dgl.ops.edge_softmax(eg, e)
                eg.edata[NORM] = a.unsqueeze(-1)
                # eg.edata[EDGEMSG] = r_msg.unsqueeze(0).expand(eg.number_of_edges(), -1, -1, -1)

        return edge_feat

    def _node_message_func(self, edges, et=None):
        if et is None:
            r_msg = edges.data[EDGEMSG]
        else:
            r_msg = self.relation_msg[et]
        return {NODEMSG: edges.data[NORM] * th.einsum("bij,bijk->bik", edges.src["v_" + NODEFEAT], r_msg)}

    def _node_update_func(self, nodes):
        agg = nodes.data[NODEAGG].view(-1, self.hidden_dim)
        if self.self_loop:  # simulate the message func for loops
            out = agg + nodes.data["loop_" + NODEFEAT]
        else:
            out = agg
        if self.bias is not None:
            out = out + self.bias
        if self.bn is not None:
            out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)

        return {NODEOUTPUT: out}

    def forward(self, graph, node_feat, edge_feat=None):
        # g = graph.local_var()
        g = graph

        self.node_init_func(g, node_feat)
        self.edge_init_func(g, edge_feat)

        if len(graph.canonical_etypes) == 1:
            edge_type = None
            if EDGETYPE in graph.ndata:
                edge_type = EDGETYPE
            elif EDGELABEL in graph.ndata:
                edge_type = EDGELABEL
            else:
                edge_type = EDGETYPE

            et = graph.edata[edge_type]
            r_msg = self.relation_msg[et]

            graph.update_all(
                lambda edges: {
                    NODEMSG: edges.data[NORM] * th.einsum("bij,bijk->bik", edges.src["v_" + NODEFEAT], r_msg)
                },
                self.node_reduce_func,
                self.node_update_func
            )
        else:
            graph.multi_update_all(
                {
                    etype: (
                        lambda edges: {
                            NODEMSG: edges.data[NORM] * \
                                th.einsum("bij,ijk->bik", edges.src["v_" + NODEFEAT], self.relation_msg[int(etype[1])])
                        },
                        self.node_reduce_func
                    ) for etype in g.canonical_etypes
                },
                "mean",
                self.node_update_func
            )

        out = graph.ndata[NODEOUTPUT]

        return out

    def extra_repr(self):
        summary = [
            "in=%d, out=%d," % (self.input_dim, self.hidden_dim),
            "num_node_types=%d, num_edge_types=%d," % (self.num_node_types, self.num_edge_types),
            "regularizer=%s, num_bases=%d, num_heads=%d," % (self.regularizer, self.num_bases, self.num_heads),
            "self_loop=%s, bias=%s," % (self.self_loop, self.bias is not None),
        ]

        return "\n".join(summary)

    def get_output_dim(self):
        return self.hidden_dim


class HGT(GraphAdjModel):
    def __init__(self, **kw):
        super(HGT, self).__init__(**kw)

    def create_rep_net(self, type, **kw):
        if type == "graph":
            num_layers = kw.get("rep_num_graph_layers", 1)
            num_node_types = self.max_ngvl
            num_edge_types = self.max_ngel
        elif type == "pattern":
            if self.share_rep_net:
                return self.g_rep_net
            num_layers = kw.get("rep_num_pattern_layers", 1)
            num_node_types = self.max_npvl
            num_edge_types = self.max_npel
        regularizer = kw.get("rep_hgt_regularizer", "diag")
        num_bases = kw.get("rep_hgt_num_bases", -1)
        num_heads = kw.get("rep_hgt_num_heads", 4)
        batch_norm = kw.get("rep_hgt_batch_norm", False)
        act_func = kw.get("rep_act_func", "relu")
        dropout = kw.get("rep_dropout", 0.0)

        hgt = ModuleList()
        for i in range(num_layers):
            hgt.add_module(
                "%s_hgt_(%d)" % (type, i),
                HeteroGraphTransLayer(
                    self.hid_dim, self.hid_dim,
                    num_node_types=num_node_types,
                    num_edge_types=num_edge_types,
                    regularizer=regularizer, num_bases=num_bases,
                    num_heads=num_heads,
                    batch_norm=batch_norm, act_func=act_func,
                    dropout=dropout
                )
            )

        return ModuleDict({"hgt": hgt})

    def get_pattern_rep(self, pattern, p_emb, mask=None):
        if mask is not None:
            p_zero_mask = ~(mask)
            outputs = [p_emb.masked_fill(p_zero_mask, 0.0)]
            for layer in self.p_rep_net["hgt"]:
                o = layer(pattern, outputs[-1])
                outputs.append(o.masked_fill(p_zero_mask, 0.0))
        else:
            outputs = [p_emb]
            for layer in self.p_rep_net["hgt"]:
                o = layer(pattern, outputs[-1])
                outputs.append(o)

        return outputs[-1]

    def get_graph_rep(self, graph, g_emb, mask=None, gate=None):
        if mask is None and gate is None:
            outputs = [g_emb]
            for layer in self.g_rep_net["hgt"]:
                o = layer(graph, outputs[-1])
                outputs.append(o)
        else:
            if gate is None:
                gate = mask.float()
            elif mask is not None:
                gate = mask.float() * gate

            outputs = [g_emb * gate]
            for layer in self.g_rep_net["hgt"]:
                o = layer(graph, outputs[-1])
                outputs.append(o * gate)

        return outputs[-1]
