from collections.abc import Sequence
from optparse import Option
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

from zmq import device

from custom_neighbor_sample_fn import custom_sample_fn, custom_hetero_sample_fn, custom_skip_sample_fn, custom_weighted_sample_fn, custom_skip_hetero_sample_fn
import torch
from torch import Tensor
import copy
import math
from typing import Dict, Optional, Tuple, Union
from torch_geometric.typing import EdgeType, OptTensor
import numpy as np
from scipy.sparse import csc_matrix

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.utils import (edge_type_to_str, filter_data,
                                          filter_hetero_data, to_csc, to_hetero_csc)
from utils import to_weighted_csc, to_weighted_hetero_csc
from torch_geometric.typing import EdgeType, InputNodes

NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]



class NeighborSampler:
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: NumNeighbors,
        replace: bool = False,
        directed: bool = True,
        labeled: bool = False, 
        input_node_type: Optional[str] = None,
        skip: bool = False, 
        weight_func: Optional[str] = None,
        device: Optional[str] = 'cpu',
        exp: Optional[bool] = False
    ):
        self.data_cls = data.__class__
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed
        self.labeled = labeled
        self.data = data
        self.skip = skip
        self.weight_func = weight_func
        self.exp = exp
        

        if isinstance(data, Data):
            # Convert the graph data into a suitable format for sampling.
            if data.edge_weight is not None:
                print('edge weights recognized')
                self.colptr, self.row, self.perm, self.edge_weight = to_weighted_csc(data, device=device)
            else:
                self.colptr, self.row, self.perm = to_csc(data, device=device)
            #assert isinstance(num_neighbors, (list, tuple))

        elif isinstance(data, HeteroData):
            # Convert the graph data into a suitable format for sampling.
            # NOTE: Since C++ cannot take dictionaries with tuples as key as
            # input, edge type triplets are converted into single strings.

            if len(self.data.edge_weight_dict) > 0:
                print('edge weight dict recognized')
                self.colptr_dict, self.row_dict, self.perm_dict, self.edge_weight_dict = to_weighted_hetero_csc(data, device=device)
            else:
                out = to_hetero_csc(data, device=device)
                self.colptr_dict, self.row_dict, self.perm_dict = out

            self.node_types, self.edge_types = data.metadata()
            if isinstance(num_neighbors, (list, tuple)):
                num_neighbors = {key: num_neighbors for key in self.edge_types}
            assert isinstance(num_neighbors, dict)
            self.num_neighbors = {
                edge_type_to_str(key): value
                for key, value in num_neighbors.items()
            }
            

            self.num_hops = max([len(v) for v in self.num_neighbors.values()])

            assert isinstance(input_node_type, str)
            self.input_node_type = input_node_type

        else:
            raise TypeError(f'NeighborLoader found invalid type: {type(data)}')

       
    def __call__(self, index: Union[List[int], Tensor]):
        #print('call')
        #print(index)
        if not isinstance(index, torch.LongTensor):
            index = torch.LongTensor(index)

        if issubclass(self.data_cls, Data):

            if self.weight_func is not None:
                if self.skip:
                    sample_fn = custom_skip_sample_fn
                    node, row, col, edge = sample_fn(
                        self.colptr,
                        self.row,
                        self.edge_weight,
                        index,
                        self.num_neighbors,
                        self.replace,
                        self.directed,
                        self.weight_func
                    )
                else:
                    sample_fn = custom_weighted_sample_fn
                    node, row, col, edge = sample_fn(
                        self.colptr,
                        self.row,
                        self.edge_weight,
                        index,
                        self.num_neighbors,
                        self.replace,
                        self.directed,
                    )

            else:
                #print(index)
                #sample_fn = torch.ops.torch_sparse.neighbor_sample
                sample_fn = custom_sample_fn
                node, row, col, edge = sample_fn(
                    self.colptr,
                    self.row,
                    index,
                    self.num_neighbors,
                    self.replace,
                    self.directed,
                )
            
            return node, row, col, edge, index.numel()

        elif issubclass(self.data_cls, HeteroData):

            if self.weight_func is not None :
                if self.skip:
                    sample_fn = custom_skip_hetero_sample_fn
                    node_dict, row_dict, col_dict, edge_dict = sample_fn(
                    self.node_types,
                    self.edge_types,
                    self.colptr_dict,
                    self.row_dict,
                    self.edge_weight_dict,
                    {self.input_node_type: index},
                    self.num_neighbors,
                    self.num_hops,
                    self.replace,
                    self.directed,
                    self.labeled,
                    self.weight_func,
                    self.exp
                )
            else: 
                sample_fn = custom_hetero_sample_fn
                node_dict, row_dict, col_dict, edge_dict = sample_fn(
                    self.node_types,
                    self.edge_types,
                    self.colptr_dict,
                    self.row_dict,
                    {self.input_node_type: index},
                    self.num_neighbors,
                    self.num_hops,
                    self.replace,
                    self.directed,
                    self.labeled
                )
            return node_dict, row_dict, col_dict, edge_dict, index.numel()

        


class NeighborLoader(torch.utils.data.DataLoader):
    r"""A data loader that performs neighbor sampling as introduced in the
    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, :obj:`num_neighbors` denotes how much neighbors are
    sampled for each node in each iteration.
    :class:`~torch_geometric.loader.NeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    .. code-block:: python

        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import NeighborLoader

        data = Planetoid(path, name='Cora')[0]

        loader = NeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=data.train_mask,
        )

        sampled_data = next(iter(loader))
        print(sampled_data.batch_size)
        >>> 128

    By default, the data loader will only include the edges that were
    originally sampled (:obj:`directed = True`).
    This option should only be used in case the number of hops is equivalent to
    the number of GNN layers.
    In case the number of GNN layers is greater than the number of hops,
    consider setting :obj:`directed = False`, which will include all edges
    between all sampled nodes (but is slightly slower as a result).

    Furthermore, :class:`~torch_geometric.loader.NeighborLoader` works for both
    **homogeneous** graphs stored via :class:`~torch_geometric.data.Data` as
    well as **heterogeneous** graphs stored via
    :class:`~torch_geometric.data.HeteroData`.
    When operating in heterogeneous graphs, more fine-grained control over
    the amount of sampled neighbors of individual edge types is possible, but
    not necessary:

    .. code-block:: python

        from torch_geometric.datasets import OGB_MAG
        from torch_geometric.loader import NeighborLoader

        hetero_data = OGB_MAG(path)[0]

        loader = NeighborLoader(
            hetero_data,
            # Sample 30 neighbors for each node and edge type for 2 iterations
            num_neighbors={key: [30] * 2 for key in hetero_data.edge_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=128,
            input_nodes=('paper', hetero_data['paper'].train_mask),
        )

        sampled_hetero_data = next(iter(loader))
        print(sampled_hetero_data['paper'].batch_size)
        >>> 128

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.NeighborLoader`, see
        `examples/hetero/to_hetero_mag.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py>`_.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: NumNeighbors,
        input_nodes: InputNodes = None,
        replace: bool = False,
        directed: bool = True,
        labeled: bool = False,
        transform: Callable = None,
        skip: bool = False,
        weight_func: Optional[str] = None,
        neighbor_sampler: Optional[NeighborSampler] = None,
        device: Optional[str] = 'cpu',
        exp: Optional[bool] = False,
        **kwargs,
    ):
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning:
        self.data = data
        self.num_neighbors = num_neighbors
        self.input_nodes = input_nodes
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = neighbor_sampler
        self.device = device
        self.exp = exp

        if neighbor_sampler is None:
            input_node_type = get_input_node_type(input_nodes)
            self.neighbor_sampler = NeighborSampler(data, num_neighbors,
                                                    replace, directed, labeled,
                                                    input_node_type, skip=skip, weight_func = weight_func, exp=exp, device=device)

        super().__init__(get_input_node_indices(self.data, input_nodes),
                                collate_fn=self.neighbor_sampler, **kwargs)

    def transform_fn(self, out: Any) -> Union[Data, HeteroData]:
        if isinstance(self.data, Data):
            node, row, col, edge, batch_size = out
            data = filter_data(self.data, node, row, col, edge,
                               self.neighbor_sampler.perm)

            if self.data.edge_weight is not None:
                data.edge_weight = self.data.edge_weight[edge]
            data.batch_size = batch_size

        elif isinstance(self.data, HeteroData):
            node_dict, row_dict, col_dict, edge_dict, batch_size = out
            data = filter_hetero_data(self.data, node_dict, row_dict, col_dict,
                                      edge_dict,
                                      self.neighbor_sampler.perm_dict)

            if len(self.data.edge_weight_dict) > 0:
                 for edge_type in self.data.edge_types:
                    data[edge_type].edge_weight = self.data[edge_type].edge_weight[edge_dict[edge_type_to_str(edge_type)]]
            data[self.neighbor_sampler.input_node_type].batch_size = batch_size

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


###############################################################################


def get_input_node_type(input_nodes: InputNodes) -> Optional[str]:
    if isinstance(input_nodes, str):
        return input_nodes
    if isinstance(input_nodes, (list, tuple)):
        assert isinstance(input_nodes[0], str)
        return input_nodes[0]
    return None


def get_input_node_indices(data: Union[Data, HeteroData],
                           input_nodes: InputNodes) -> Sequence:
    if isinstance(data, Data) and input_nodes is None:
        return range(data.num_nodes)
    if isinstance(data, HeteroData):
        if isinstance(input_nodes, str):
            input_nodes = (input_nodes, None)
        assert isinstance(input_nodes, (list, tuple))
        assert len(input_nodes) == 2
        assert isinstance(input_nodes[0], str)
        if input_nodes[1] is None:
            return range(data[input_nodes[0]].num_nodes)
        input_nodes = input_nodes[1]

    if isinstance(input_nodes, Tensor):
        if input_nodes.dtype == torch.bool:
            input_nodes = input_nodes.nonzero(as_tuple=False).view(-1)
        input_nodes = input_nodes.tolist()

    assert isinstance(input_nodes, Sequence)
    return input_nodes

#def to_hetero_csc(
#    data: HeteroData,
#    device: Optional[torch.device] = None,
#    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    # Convert the heterogeneous graph data into a suitable format for sampling
    # (CSC format).
    # Returns dictionaries holding `colptr` and `row` indices as well as edge
    # permutations for each edge type, respectively.
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets are converted into single strings.

#    colptr_dict, row_dict, perm_dict = {}, {}, {}

#    for store in data.edge_stores:

        # We need to make sure our csc matrix has num_rows = num unique nodes and num_cols = num unique nodes
#        key = edge_type_to_str(store._key)
#       (row, col) = store.edge_index
#        values = np.array([1 for i in zip(row.numpy(), col.numpy())])
#
#        num_rows = data[store._key[0]].x.shape[0]
#        num_cols = data[store._key[2]].x.shape[0]
#
#        sp_mat = csc_matrix((values, (row.numpy(), col.numpy())), shape = (num_rows, num_cols))
#        size = store.size()
#        perm = (col * size[0]).add_(row).argsort()
#        colptr_dict[key] = torch.from_numpy(sp_mat.indptr).to(torch.long)
#        row_dict[key] = torch.from_numpy(sp_mat.indices).to(torch.long)
#        perm_dict[key] = perm

#    return colptr_dict, row_dict, perm_dictm