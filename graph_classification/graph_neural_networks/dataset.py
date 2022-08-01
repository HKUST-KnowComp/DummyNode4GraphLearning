import torch
import os
import shutil
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import TUDataset
from torch_geometric.io import read_tu_data
from torch_geometric.data import Data


class PYGDataset(InMemoryDataset):
    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets')

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_node_attr=False,
        use_edge_attr=False,
        cleaned=False,
        add_dummy=False,
        convert_conjugate=False
    ):
        self.name = name
        self.cleaned = cleaned
        self.add_dummy = add_dummy
        self.convert_conjugate = convert_conjugate
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self) -> str:
        if self.add_dummy and self.convert_conjugate:
            data_name = "CONJ_" + self.name
        elif self.add_dummy:
            data_name = "DUMMY_" + self.name
        elif self.convert_conjugate:
            data_name = "LINE_" + self.name
        else:
            data_name = self.name
        dir_name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, data_name, dir_name)

    @property
    def processed_dir(self) -> str:
        if self.add_dummy and self.convert_conjugate:
            data_name = "CONJ_" + self.name
        elif self.add_dummy:
            data_name = "DUMMY_" + self.name
        elif self.convert_conjugate:
            data_name = "LINE_" + self.name
        else:
            data_name = self.name
        dir_name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, data_name, dir_name)

    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = os.path.join(self.root, self.name)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        if not self.add_dummy and not self.convert_conjugate:
            shutil.rmtree(self.raw_dir)
            os.rename(os.path.join(folder, self.name), self.raw_dir)
        else:
            raise ValueError("Please use tu_data_processing.py to add_dummy and convert_conjugate")

    def set_dummy_flags(self, data):
        if self.add_dummy:
            is_dummy_node = data.x[:, self.num_node_attributes].bool()
            if hasattr(data, "edge_attr"):
                is_dummy_edge = getattr(data, "edge_attr")[:, self.num_edge_attributes].bool()
            else:
                is_dummy_edge = torch.index_select(is_dummy_node, dim=0, index=data.edge_index[0])
                is_dummy_edge |= torch.index_select(is_dummy_node, dim=0, index=data.edge_index[1])
        else:
            is_dummy_node = torch.zeros(data.x.shape[0], dtype=torch.bool)
            is_dummy_edge = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)

        data = Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
            y=data.y,
            is_dummy_node=is_dummy_node,
            is_dummy_edge=is_dummy_edge
        )

        return data

    def process(self):
        if self.add_dummy and self.convert_conjugate:
            data_name = "CONJ_" + self.name
        elif self.add_dummy:
            data_name = "DUMMY_" + self.name
        elif self.convert_conjugate:
            data_name = "LINE_" + self.name
        else:
            data_name = self.name

        self.data, self.slices = read_tu_data(self.raw_dir, data_name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        # set dummy flags
        data_list = [self.get(idx) for idx in range(len(self))]
        data_list = [self.set_dummy_flags(data) for data in data_list]
        self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
