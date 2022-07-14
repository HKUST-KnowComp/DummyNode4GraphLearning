import dgl
import igraph as ig
import numba
import numpy as np
import torch as th

# from ..constants import *
from constants import *


def compute_norm(graph, self_loop):
    if isinstance(graph, dgl.DGLGraph):
        if INDEGREE in graph.ndata:
            in_deg = graph.ndata[INDEGREE].float()
        else:
            in_deg = graph.in_degrees().float()
        if self_loop:
            node_norm = (in_deg + 1).reciprocal().unsqueeze(-1)
        else:
            node_norm = in_deg.reciprocal().masked_fill_(in_deg == 0, 1.0).unsqueeze(-1)
        u, v = graph.all_edges(form="uv", order="eid")
        edge_norm = node_norm[v]
    elif isinstance(graph, ig.Graph):
        if INDEGREE in graph.vertex_attributes():
            in_deg = np.asarray(graph.vs[INDEGREE]).astype(np.float32)
        else:
            in_deg = np.asarray(graph.indegree()).astype(np.float32)
        if self_loop:
            node_norm = np.expand_dims(np.reciprocal((in_deg + 1)), -1)
        else:
            node_norm = np.expand_dims(np.nan_to_num(np.reciprocal(in_deg), nan=1.0, posinf=1.0, copy=False), -1)
        u, v = np.asarray(graph.get_edgelist()).T
        edge_norm = node_norm[v]
    else:
        raise ValueError

    return node_norm, edge_norm


def compute_largest_eigenvalues(graph):
    if isinstance(graph, dgl.DGLGraph):
        if INDEGREE in graph.ndata:
            in_deg = graph.ndata[INDEGREE].float()
        else:
            in_deg = graph.in_degrees().float()
        if OUTDEGREE in graph.ndata:
            out_deg = graph.ndata[OUTDEGREE].float()
        else:
            out_deg = graph.out_degrees().float()
        u, v = graph.all_edges(form="uv", order="eid")
        max_nd = (out_deg[u] + in_deg[v]).max()
        max_ed = (in_deg[u] + out_deg[v]).max()
    elif isinstance(graph, ig.Graph):
        if INDEGREE in graph.vertex_attributes():
            in_deg = np.asarray(graph.vs[INDEGREE]).astype(np.float32)
        else:
            in_deg = np.asarray(graph.indegree()).astype(np.float32)
        if OUTDEGREE in graph.vertex_attributes():
            out_deg = np.asarray(graph.vs[OUTDEGREE]).astype(np.float32)
        else:
            out_deg = np.asarray(graph.outdegree()).astype(np.float32)
        u, v = np.asarray(graph.get_edgelist()).T
        max_nd = (out_deg[u] + in_deg[v]).max()
        max_ed = (in_deg[u] + out_deg[v]).max()
    else:
        raise ValueError

    node_eigenv = max_nd
    edge_eigenv = max_ed

    return node_eigenv, edge_eigenv


def convert_conjugate_graph(graph):
    class_name = str(graph.__class__)

    if class_name in [
        "<class 'torchkits.data.gadj.Graph'>",
        "<class 'dgl.heterograph.DGLHeteroGraph'>",
        "<class 'dgl.graph.DGLGraph'>"
    ]:
        conj_graph = graph.__class__()

        # if two vertices with the same ids
        # merge them
        if EDGEID in graph.edata and graph.number_of_edges() > 0:
            eids = graph.edata[EDGEID].numpy()
            num_edges = eids.max().item() + 1
            id2vertex = [None] * num_edges
            for e, eid in enumerate(eids):
                if id2vertex[eid] is None:
                    id2vertex[eid] = e
                else:
                    id2vertex[eid] = min(id2vertex[eid], e)
            conj_graph.add_nodes(num_edges)
            for k, v in graph.edata.items():
                vv = th.zeros((num_edges, ) + v.size()[1:], dtype=v.dtype)
                for e in range(num_edges):
                    if id2vertex[e] is not None:
                        vv[e].copy_(v[id2vertex[e]])
                conj_graph.ndata[k] = vv
        else:
            num_edges = graph.number_of_edges()
            id2vertex = list(range(num_edges))
            conj_graph.add_nodes(num_edges)
            for k, v in graph.edata.items():
                conj_graph.ndata[k] = v
            if EDGEID not in graph.edata:
                conj_graph.ndata[EDGEID] = th.arange(graph.number_of_edges())
        incident_matrix = graph.incidence_matrix("in")
        edges = []
        edge_indices = []
        prev = -1
        # if the two edges with the same source id, the same target id, and the same edge label
        # merge them
        if EDGEID in graph.edata and graph.number_of_edges() > 0 and NODELABEL in graph.ndata:
            used_keys = set()
            for e, source in enumerate(graph.edges(form="uv", order="eid")[0].numpy()):
                vid = eids[e]
                elabel = graph.ndata[NODELABEL][source].item()
                if prev != source:
                    # incident_edges = incident_matrix[source].coalesce().indices().numpy()[0]
                    incident_edges = incident_matrix[source]._indices().numpy()[0]
                for incident_e in incident_edges:
                    uid = eids[incident_e]
                    key = (uid, elabel, vid)
                    if key not in used_keys:
                        used_keys.add(key)
                        edges.append((uid, vid))
                        edge_indices.append(source)
                prev = source
        else:
            for e, source in enumerate(graph.edges(form="uv", order="eid")[0].numpy()):
                if prev != source:
                    # incident_edges = incident_matrix[source].coalesce().indices().numpy()[0]
                    incident_edges = incident_matrix[source]._indices().numpy()[0]
                for incident_e in incident_edges:
                    edges.append((incident_e, e))
                    edge_indices.append(source)
                prev = source
        if len(edges) > 0:
            edges = th.tensor(edges)
            edge_indices = th.tensor(edge_indices)
            conj_graph.add_edges(edges[:, 0], edges[:, 1])
            for k, v in graph.ndata.items():
                conj_graph.edata[k] = v[edge_indices]
            if NODEID not in graph.ndata:
                conj_graph.edata[NODEID] = th.LongTensor(edge_indices)
        else:
            for k, v in graph.ndata.items():
                conj_graph.edata[k] = th.zeros((0, ) + tuple(v.size())[1:], dtype=v.dtype)
            if NODEID not in graph.ndata:
                conj_graph.edata[NODEID] = th.zeros((0, ), dtype=th.long)

        if NODEID != EDGEID:
            conj_graph.ndata[NODEID] = conj_graph.ndata[EDGEID]
            del conj_graph.ndata[EDGEID]
            conj_graph.edata[EDGEID] = conj_graph.edata[NODEID]
            del conj_graph.edata[NODEID]

        if NODELABEL != EDGELABEL:
            conj_graph.ndata[NODELABEL] = conj_graph.ndata[EDGELABEL]
            del conj_graph.ndata[EDGELABEL]
            conj_graph.edata[EDGELABEL] = conj_graph.edata[NODELABEL]
            del conj_graph.edata[NODELABEL]

        for v in id2vertex:
            if v is None:
                conj_graph.remove_nodes(th.tensor([v for v in range(len(id2vertex)) if id2vertex[v] is None]))
                break

        if "_ID" in conj_graph.ndata:
            conj_graph.ndata.pop("_ID")
        if "_ID" in conj_graph.edata:
            conj_graph.edata.pop("_ID")

    elif class_name == "<class 'igraph.Graph'>":

        conj_graph = ig.Graph(directed=True)

        # if two vertices with the same ids
        # merge them
        if EDGEID in graph.edge_attributes() and graph.ecount() > 0:
            eids = graph.es[EDGEID]
            num_edges = max(eids) + 1
            id2vertex = [None] * num_edges
            for e, eid in enumerate(eids):
                if id2vertex[eid] is None:
                    id2vertex[eid] = e
                else:
                    id2vertex[eid] = min(id2vertex[eid], e)
            conj_graph.add_vertices(num_edges)
            for k in graph.edge_attributes():
                v = graph.es[k]
                conj_graph.vs[k] = [v[id2vertex[e]] if id2vertex[e] is not None else v[0].__class__() for e in range(num_edges)]
        else:
            num_edges = graph.ecount()
            id2vertex = list(range(num_edges))
            conj_graph.add_vertices(num_edges)
            for k in graph.edge_attributes():
                v = graph.es[k]
                conj_graph.vs[k] = v
            if EDGEID not in graph.edge_attributes():
                conj_graph.vs[EDGEID] = list(range(graph.ecount()))

        edges = []
        edge_indices = []
        prev = -1

        # if the two edges with the same source id, the same target id, and the same edge label
        # merge them
        if EDGEID in graph.edge_attributes() and graph.ecount() > 0 and NODELABEL in graph.vertex_attributes():
            used_keys = set()
            for e in range(graph.ecount()):
                source = graph.es[e].source
                vid = eids[e]
                elabel = graph.vs[source][NODELABEL]
                if prev != source:
                    incident_edges = sorted(graph.incident(source, "in"))
                for incident_e in incident_edges:
                    uid = eids[incident_e]
                    key = (uid, elabel, vid)
                    if key not in used_keys:
                        used_keys.add(key)
                        edges.append((uid, vid))
                        edge_indices.append(source)
                prev = source
        else:
            for e in range(graph.ecount()):
                source = graph.es[e].source
                if prev != source:
                    incident_edges = sorted(graph.incident(source, "in"))
                for incident_e in incident_edges:
                    edges.append((incident_e, e))
                    edge_indices.append(source)
                prev = source

        if len(edges) > 0:
            conj_graph.add_edges(edges)

            for k in graph.vertex_attributes():
                v = graph.vs[k]
                conj_graph.es[k] = [v[i] for i in edge_indices]
            if NODEID not in graph.vertex_attributes():
                conj_graph.es[NODEID] = edge_indices
        else:
            for k in graph.vertex_attributes():
                conj_graph.es[k] = []
            if NODEID not in graph.vertex_attributes():
                conj_graph.es[NODEID] = []

        if NODEID != EDGEID:
            conj_graph.vs[NODEID] = conj_graph.vs[EDGEID]
            del conj_graph.vs[EDGEID]
            conj_graph.es[EDGEID] = conj_graph.es[NODEID]
            del conj_graph.es[NODEID]

        if NODELABEL != EDGELABEL:
            conj_graph.vs[NODELABEL] = conj_graph.vs[EDGELABEL]
            del conj_graph.vs[EDGELABEL]
            conj_graph.es[EDGELABEL] = conj_graph.es[NODELABEL]
            del conj_graph.es[NODELABEL]

        for v in id2vertex:
            if v is None:
                conj_graph.delete_vertices([v for v in range(len(id2vertex)) if id2vertex[v] is None])
                break

    elif class_name == "<class 'torchkits.data.eseq.EdgeSeq'>":
        assert graph._batch_num_tuples is None

        conj_graph = convert_conjugate_graph(graph.to_graph()).to_edgeseq()
        
    else:
        raise ValueError

    return conj_graph


@numba.jit(numba.int64(numba.int64[:], numba.int64, numba.int64, numba.int64), nopython=True)
def long_item_bisect_left(array, x, lo, hi):
    while lo < hi:
        mid = (lo + hi) // 2
        if array[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.jit(numba.int64[:, :](
    numba.int64[:], numba.int64[:], numba.int64[:],
    numba.int64[:], numba.int64[:], numba.int64[:],
    numba.int64[:, :]), nopython=True
)
def get_conjugate_subisomorphisms(p_u, p_v, p_el, g_u, g_v, g_el, subisomorphisms):
    p_len = len(p_el)
    g_len = len(g_el)
    conj_subisomorphisms = np.zeros((len(subisomorphisms), p_len), dtype=np.int64)

    # numba does not support tuples as keys
    # hence, we use p_u * (max_v + 1) + p_v as the key
    max_v = max([p_u.max(), p_v.max(), g_u.max(), g_v.max()])
    mod = max_v + 1
    pattern_keys = p_u * mod + p_v
    pattern_elabels = dict()
    i = 0
    while i < p_len:
        j = i + 1
        while j < p_len and pattern_keys[i] == pattern_keys[j]:
            j += 1
        p_els = p_el[i:j]
        pattern_elabels[pattern_keys[i]] = p_els
        i = j

    for i, subisomorphism in enumerate(subisomorphisms):
        for p_eid, key in enumerate(pattern_elabels.keys()):
            p_els = pattern_elabels[key]
            u, v = key // mod, key % mod
            u, v = subisomorphism[u], subisomorphism[v]
            u_i = long_item_bisect_left(g_u, u, 0, g_len)
            u_j = long_item_bisect_left(g_u, u + 1, 0, g_len)
            v_i = long_item_bisect_left(g_v, v, u_i, u_j)
            v_j = long_item_bisect_left(g_v, v + 1, v_i, u_j)
            # len_pels = len(p_els)
            for k in range(v_i, v_j):
                for e in p_els:
                    if e == g_el[k]:
                        conj_subisomorphisms[i, p_eid] = k
    return conj_subisomorphisms
