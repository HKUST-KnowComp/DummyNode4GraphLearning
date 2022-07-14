import argparse
import igraph as ig
import os
import torch
from collections import Counter
from itertools import chain
# from torch_geometric.data import InMemoryDataset
# from torch_geometric.datasets import TUDataset
# from torch_geometric.io import read_tu_data

# class MyOwnDataset(InMemoryDataset):
#     def __init__(
#         self,
#         root,
#         name,
#         transform=None,
#         pre_transform=None,
#         pre_filter=None,
#         use_node_attr=False,
#         use_edge_attr=False,
#         cleaned=False
#     ):
#         self.name = name
#         self.cleaned = cleaned
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#         if self.data.x is not None and not use_node_attr:
#             num_node_attributes = self.num_node_attributes
#             self.data.x = self.data.x[:, num_node_attributes:]
#         if self.data.edge_attr is not None and not use_edge_attr:
#             num_edge_attributes = self.num_edge_attributes
#             self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

#     @property
#     def raw_dir(self) -> str:
#         name = f'raw{"_cleaned" if self.cleaned else ""}'
#         return os.path.join(self.root, self.name, name)

#     @property
#     def processed_dir(self) -> str:
#         name = f'processed{"_cleaned" if self.cleaned else ""}'
#         return os.path.join(self.root, self.name, name)

#     @property
#     def num_node_labels(self) -> int:
#         if self.data.x is None:
#             return 0
#         for i in range(self.data.x.size(1)):
#             x = self.data.x[:, i:]
#             if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
#                 return self.data.x.size(1) - i
#         return 0

#     @property
#     def num_node_attributes(self) -> int:
#         if self.data.x is None:
#             return 0
#         return self.data.x.size(1) - self.num_node_labels

#     @property
#     def num_edge_labels(self) -> int:
#         if self.data.edge_attr is None:
#             return 0
#         for i in range(self.data.edge_attr.size(1)):
#             if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
#                 return self.data.edge_attr.size(1) - i
#         return 0

#     @property
#     def num_edge_attributes(self) -> int:
#         if self.data.edge_attr is None:
#             return 0
#         return self.data.edge_attr.size(1) - self.num_edge_labels

#     @property
#     def raw_file_names(self):
#         names = ['A', 'graph_indicator']
#         return [f'{self.name}_{name}.txt' for name in names]

#     @property
#     def processed_file_names(self) -> str:
#         return 'data.pt'

#     def process(self):
#         self.data, self.slices = read_tu_data(self.raw_dir, self.name)

#         if self.pre_filter is not None:
#             data_list = [self.get(idx) for idx in range(len(self))]
#             data_list = [data for data in data_list if self.pre_filter(data)]
#             self.data, self.slices = self.collate(data_list)

#         if self.pre_transform is not None:
#             data_list = [self.get(idx) for idx in range(len(self))]
#             data_list = [self.pre_transform(data) for data in data_list]
#             self.data, self.slices = self.collate(data_list)

#         torch.save((self.data, self.slices), self.processed_paths[0])

#     def __repr__(self) -> str:
#         return f'{self.name}({len(self)})'


def load_graph_labels_from_TUDatadir(data_dir):
    graph_labels = []
    for file_name in sorted(os.listdir(data_dir)):
        if file_name.endswith("_graph_labels.txt"):
            with open(os.path.join(data_dir, file_name), "r") as f:
                graph_labels.extend([line.strip() for line in f])
    return graph_labels


def load_graph_data_from_TUDatadir(data_dir, with_dummy=False):
    A = []
    graph_indicator = []
    node_labels = []
    edge_labels = []
    for file_name in sorted(os.listdir(data_dir)):
        if file_name.endswith("_A.txt"):
            with open(os.path.join(data_dir, file_name), "r") as f:
                A.extend([tuple(map(int, line.strip().replace(" ", "").split(","))) for line in f])
        elif file_name.endswith("_graph_indicator.txt"):
            with open(os.path.join(data_dir, file_name), "r") as f:
                graph_indicator.extend([int(line.strip()) for line in f])
        elif file_name.endswith("_node_labels.txt"):
            with open(os.path.join(data_dir, file_name), "r") as f:
                node_labels.extend([int(line.strip()) for line in f])
        elif file_name.endswith("_edge_labels.txt"):
            with open(os.path.join(data_dir, file_name), "r") as f:
                edge_labels.extend([int(line.strip()) for line in f])

    if len(node_labels) == 0:
        node_labels = [1] * len(graph_indicator)
    else:
        min_node_label = min(node_labels)
        if min_node_label == 0:
            node_labels = list(map(lambda x: x + 1, node_labels))
        elif min_node_label != 1:
            node_labels = list(map(lambda x: x - min_node_label + 1, node_labels))
    if len(edge_labels) == 0:
        edge_labels = [1] * len(A)
    else:
        min_edge_label = min(edge_labels)
        if min_edge_label == 0:
            edge_labels = list(map(lambda x: x + 1, edge_labels))
        elif min_edge_label != 1:
            edge_labels = list(map(lambda x: x - min_edge_label + 1, edge_labels))

    graph_ns = Counter(graph_indicator)  # 0 is an empty graph

    graphs = []
    i = 0
    j = 0
    k = len(A)
    pre_n = 0
    gd = graph_indicator[0]
    while j < k:
        while j < len(A) and gd == graph_indicator[A[j][0] - 1] and gd == graph_indicator[A[j][1] - 1]:
            j += 1

        n = graph_ns[gd]
        m = j - i
        graph = ig.Graph(directed=True)
        if with_dummy:
            graph.add_vertices(n + 1)
            graph.vs["LABEL"] = node_labels[pre_n:(pre_n + n)] + [0]
            graph.vs["IS_DUMMY"] = [0] * n + [1]
            graph.add_edges([(e[0] - pre_n - 1, e[1] - pre_n - 1) for e in A[i:j]])  # m edges
            graph.add_edges(chain.from_iterable([[(n, v), (v, n)] for v in range(0, n)]))  # 2n dummy edges
            graph.es["LABEL"] = edge_labels[i:j] + [0] * (2 * n)
            graph.es["IS_DUMMY"] = [0] * (j - i) + [1] * (2 * n)

            assert n + 1 == graph.vcount()
            assert m + 2 * n == graph.ecount()
        else:
            graph.add_vertices(n)
            graph.vs["LABEL"] = node_labels[pre_n:(pre_n + n)]
            graph.add_edges([(e[0] - pre_n - 1, e[1] - pre_n - 1) for e in A[i:j]])  # m edges
            graph.es["LABEL"] = edge_labels[i:j]

            assert n == graph.vcount()
            assert m == graph.ecount()
        graph.vs["ID"] = list(range(graph.vcount()))
        graph.es["ID"] = list(range(graph.ecount()))
        graphs.append(graph)
        gd += 1
        pre_n += n
        i = j

    return graphs


def convert_conjugate_graph_forward(graph):
    conj_graph = ig.Graph(directed=True)

    # if two vertices with the same ids
    # merge them
    if "ID" in graph.edge_attributes() and graph.ecount() > 0:
        eids = graph.es["ID"]
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
            conj_graph.vs[k] = [
                v[id2vertex[e]] if id2vertex[e] is not None else v[0].__class__() for e in range(num_edges)
            ]
    else:
        num_edges = graph.ecount()
        id2vertex = list(range(num_edges))
        conj_graph.add_vertices(num_edges)
        for k in graph.edge_attributes():
            v = graph.es[k]
            conj_graph.vs[k] = v
        if "ID" not in graph.edge_attributes():
            conj_graph.vs["ID"] = list(range(graph.ecount()))

    edges = list()
    edge_indices = list()
    prev = -1

    # if the two edges with the same source id, the same target id, and the same edge label
    # merge them
    if "ID" in graph.edge_attributes() and graph.ecount() > 0 and "LABEL" in graph.vertex_attributes():
        used_keys = set()
        for e in range(graph.ecount()):
            source = graph.es[e].source
            vid = eids[e]
            elabel = graph.vs[source]["LABEL"]
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

    # if two vertices are dummy
    # merge them
    # if one edge starting from dummy nodes and sinking to dummy nodes
    # remove it
    if "IS_DUMMY" in graph.edge_attributes():
        dummy_eids = list()
        new_edges = list()
        new_edge_indices = list()
        if "ID" in graph.edge_attributes():
            for e, flag in zip(eids, graph.es["IS_DUMMY"]):
                if flag:
                    dummy_eids.append(e)
        else:
            for e, flag in enumerate(graph.es["IS_DUMMY"]):
                if flag:
                    dummy_eids.append(e)
        if len(dummy_eids) > 0:
            for e in dummy_eids[1:]:
                id2vertex[e] = None
            prev = dummy_eids[0]
            dummy_eids = set(dummy_eids)
            used_keys = set([(prev, prev)])
            for e in range(len(edges)):
                uid, vid = edges[e]
                if uid in dummy_eids:
                    uid = prev
                if vid in dummy_eids:
                    vid = prev
                key = (uid, vid)
                if key not in used_keys:
                    used_keys.add(key)
                    new_edges.append(key)
                    new_edge_indices.append(edge_indices[e])
            edges, edge_indices = new_edges, new_edge_indices

    if len(edges) > 0:
        conj_graph.add_edges(edges)
        for k in graph.vertex_attributes():
            v = graph.vs[k]
            conj_graph.es[k] = [v[i] for i in edge_indices]
        if "ID" not in graph.vertex_attributes():
            conj_graph.es["ID"] = edge_indices
    else:
        for k in graph.vertex_attributes():
            conj_graph.es[k] = list()
        if "ID" not in graph.vertex_attributes():
            conj_graph.es["ID"] = list()

    for v in id2vertex:
        if v is None:
            conj_graph.delete_vertices([v for v in range(len(id2vertex)) if id2vertex[v] is None])
            break

    return conj_graph


def save_graph_labels(graph_labels, data_dir, prefix=""):
    if prefix == "":
        prefix = os.path.basename(data_dir) + "_"
    # prefix = os.path.join("raw", prefix)

    with open(os.path.join(data_dir, prefix + "graph_labels.txt"), "w") as f:
        for line in graph_labels:
            f.write(str(line))
            f.write("\n")


def save_graph_data(graphs, data_dir, prefix=""):
    if prefix == "":
        prefix = os.path.basename(data_dir) + "_"
    # prefix = os.path.join("raw", prefix)

    n = 1  # start from 1
    m = 0

    graph_nsum = [1]  # edge source/target starts from 1
    graph_indicator = []
    for gd, g in enumerate(graphs):
        graph_nsum.append(graph_nsum[-1] + g.vcount())
        graph_indicator.extend([gd + 1] * g.vcount())
    with open(os.path.join(data_dir, prefix + "graph_indicator.txt"), "w") as f:
        for line in graph_indicator:
            f.write(str(line))
            f.write("\n")

    with open(os.path.join(data_dir, prefix + "A.txt"), "w") as f:
        for gd, g in enumerate(graphs):
            for e in g.get_edgelist():
                f.write("%d,%d" % (e[0] + graph_nsum[gd], e[1] + graph_nsum[gd]))
                f.write("\n")

    with open(os.path.join(data_dir, prefix + "node_labels.txt"), "w") as f:
        for g in graphs:
            for line in g.vs["LABEL"]:
                f.write(str(line))
                f.write("\n")

    with open(os.path.join(data_dir, prefix + "edge_labels.txt"), "w") as f:
        for g in graphs:
            for line in g.es["LABEL"]:
                f.write(str(line))
                f.write("\n")

    with open(os.path.join(data_dir, prefix + "node_ids.txt"), "w") as f:
        for g in graphs:
            for line in g.vs["ID"]:
                f.write(str(line))
                f.write("\n")

    with open(os.path.join(data_dir, prefix + "edge_ids.txt"), "w") as f:
        for g in graphs:
            for line in g.es["ID"]:
                f.write(str(line))
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_data_dir", type=str,
        default="tu_data/PROTEINS"
    )
    parser.add_argument(
        "--save_dummy_data_dir", type=str,
        default="tu_data/DUMMY_PROTEINS"
    )
    parser.add_argument(
        "--save_line_data_dir", type=str,
        default="tu_data/LINE_PROTEINS"
    )
    parser.add_argument(
        "--save_conjugate_data_dir", type=str,
        default="tu_data/CONJ_PROTEINS"
    )
    args = parser.parse_args()

    graph_labels = load_graph_labels_from_TUDatadir(args.load_data_dir)
    graph_data = load_graph_data_from_TUDatadir(args.load_data_dir, with_dummy=False)
    dummy_graph_data = load_graph_data_from_TUDatadir(args.load_data_dir, with_dummy=True)
    line_graph_data = [convert_conjugate_graph_forward(graph) for graph in graph_data]
    conjugate_graph_data = [convert_conjugate_graph_forward(graph) for graph in dummy_graph_data]

    os.makedirs(args.save_dummy_data_dir, exist_ok=True)
    os.makedirs(args.save_line_data_dir, exist_ok=True)
    os.makedirs(args.save_conjugate_data_dir, exist_ok=True)

    save_graph_data(dummy_graph_data, args.save_dummy_data_dir)
    save_graph_data(line_graph_data, args.save_line_data_dir)
    save_graph_data(conjugate_graph_data, args.save_conjugate_data_dir)

    save_graph_labels(graph_labels, args.save_dummy_data_dir)
    save_graph_labels(graph_labels, args.save_line_data_dir)
    save_graph_labels(graph_labels, args.save_conjugate_data_dir)
