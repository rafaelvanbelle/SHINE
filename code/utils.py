import torch
import os


import numpy as np
import torch


from typing import Dict, Optional, Tuple, Union, List
from torch import Tensor

from torch_geometric.typing import PairTensor



from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.typing import OptTensor
from copy import deepcopy
from torch_geometric.utils import index_to_mask, maybe_num_nodes
from torch_geometric.typing import EdgeType, OptTensor

def bipartite_subgraph(
    subset: Union[PairTensor, Tuple[List[int], List[int]]],
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    relabel_nodes: bool = False,
    size: Tuple[int, int] = None,
    return_edge_mask: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Returns the induced subgraph of the bipartite graph
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.
    Args:
        subset (Tuple[Tensor, Tensor] or tuple([int],[int])): The nodes
            to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        size (tuple, optional): The number of nodes.
            (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    Examples:
        >>> edge_index = torch.tensor([[0, 5, 2, 3, 3, 4, 4, 3, 5, 5, 6],
        ...                            [0, 0, 3, 2, 0, 0, 2, 1, 2, 3, 1]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        >>> subset = (torch.tensor([2, 3, 5]), torch.tensor([2, 3]))
        >>> bipartite_subgraph(subset, edge_index, edge_attr)
        (tensor([[2, 3, 5, 5],
                [3, 2, 2, 3]]),
        tensor([ 3,  4,  9, 10]))
        >>> bipartite_subgraph(subset, edge_index, edge_attr,
        ...                    return_edge_mask=True)
        (tensor([[2, 3, 5, 5],
                [3, 2, 2, 3]]),
        tensor([ 3,  4,  9, 10]),
        tensor([False, False,  True,  True, False, False, False, False,
                True,  True,  False]))
    """

    device = edge_index.device

    if isinstance(subset[0], (list, tuple)):
        subset = (torch.tensor(subset[0], dtype=torch.long, device=device),
                  torch.tensor(subset[1], dtype=torch.long, device=device))

    if subset[0].dtype == torch.bool or subset[0].dtype == torch.uint8:
        size = subset[0].size(0), subset[1].size(0)
    else:
        if size is None:
            size = (maybe_num_nodes(edge_index[0]),
                    maybe_num_nodes(edge_index[1]))
        subset = (index_to_mask(subset[0], size=size[0]),
                  index_to_mask(subset[1], size=size[1]))

    node_mask = subset
    edge_mask = node_mask[0][edge_index[0]] & node_mask[1][edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx_i = torch.zeros(node_mask[0].size(0), dtype=torch.long,
                                 device=device)
        node_idx_j = torch.zeros(node_mask[1].size(0), dtype=torch.long,
                                 device=device)
        node_idx_i[node_mask[0]] = torch.arange(node_mask[0].sum().item(),
                                                device=device)
        node_idx_j[node_mask[1]] = torch.arange(node_mask[1].sum().item(),
                                                device=device)
        edge_index = torch.stack(
            [node_idx_i[edge_index[0]], node_idx_j[edge_index[1]]])

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr
        
def edge_type_to_str(edge_type: Union[EdgeType, str]) -> str:
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets need to be converted into single strings.
    return edge_type if isinstance(edge_type, str) else '__'.join(edge_type)

def to_csc(
    data: Union[Data, EdgeStorage],
    device: Optional[torch.device] = None,
    share_memory: bool = False,
    is_sorted: bool = False,
) -> Tuple[Tensor, Tensor, OptTensor]:
    # Convert the graph data into a suitable format for sampling (CSC format).
    # Returns the `colptr` and `row` indices of the graph, as well as an
    # `perm` vector that denotes the permutation of edges.
    # Since no permutation of edges is applied when using `SparseTensor`,
    # `perm` can be of type `None`.
    perm: Optional[Tensor] = None

    if hasattr(data, 'adj_t'):
        colptr, row, _ = data.adj_t.csr()

    elif hasattr(data, 'edge_index'):
        (row, col) = data.edge_index
        if not is_sorted:
            size = data.size()
            perm = (col * size[0]).add_(row).argsort()
            row = row[perm]
        colptr = torch.ops.torch_sparse.ind2ptr(col[perm], size[1])
    else:
        raise AttributeError("Data object does not contain attributes "
                             "'adj_t' or 'edge_index'")

    colptr = colptr.to(device)
    row = row.to(device)
    perm = perm.to(device) if perm is not None else None

    if not colptr.is_cuda and share_memory:
        colptr.share_memory_()
        row.share_memory_()
        if perm is not None:
            perm.share_memory_()

    return colptr, row, perm


def to_hetero_csc(
    data: HeteroData,
    device: Optional[torch.device] = None,
    share_memory: bool = False,
    is_sorted: bool = False,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, OptTensor]]:
    # Convert the heterogeneous graph data into a suitable format for sampling
    # (CSC format).
    # Returns dictionaries holding `colptr` and `row` indices as well as edge
    # permutations for each edge type, respectively.
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets are converted into single strings.
    colptr_dict, row_dict, perm_dict = {}, {}, {}

    for store in data.edge_stores:
        key = edge_type_to_str(store._key)
        out = to_csc(store, device, share_memory, is_sorted)
        colptr_dict[key], row_dict[key], perm_dict[key] = out

    return colptr_dict, row_dict, perm_dict


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf'), output_path='', filename=''
    ):
        self.best_valid_loss = best_valid_loss
        self.output_path = output_path
        self.filename = filename
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(self.output_path, self.filename + '_model.pth'))





def to_weighted_csc(
    data: Union[Data, EdgeStorage],
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor, OptTensor]:
    # Convert the graph data into a suitable format for sampling (CSC format).
    # Returns the `colptr` and `row` indices of the graph, as well as an
    # `perm` vector that denotes the permutation of edges.
    # Since no permutation of edges is applied when using `SparseTensor`,
    # `perm` can be of type `None`.
    if hasattr(data, 'adj_t'):
        colptr, row, _ = data.adj_t.csr()
        return colptr.to(device), row.to(device), None

    elif hasattr(data, 'edge_index'):
        (row, col) = data.edge_index
        size = data.size()
        perm = (col * size[0]).add_(row).argsort()
        colptr = torch.ops.torch_sparse.ind2ptr(col[perm], size[1])

        # if edge_weights are present, return them with the csc matrix
        if data.edge_weight is not None:
            edge_weight = data.edge_weight
            print('it s me')
            return colptr.to(device), row[perm].to(device), perm.to(device), edge_weight[perm].to(device)

        raise AttributeError(
            "Data object does not contain attributes 'edge_weight'")
        
        
        
        
    raise AttributeError(
        "Data object does not contain attributes 'adj_t' or 'edge_index'")




def to_hetero_csc(
    data: HeteroData,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, OptTensor]]:
    # Convert the heterogeneous graph data into a suitable format for sampling
    # (CSC format).
    # Returns dictionaries holding `colptr` and `row` indices as well as edge
    # permutations for each edge type, respectively.
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets are converted into single strings.
    colptr_dict, row_dict, perm_dict = {}, {}, {}

    for store in data.edge_stores:
        key = edge_type_to_str(store._key)
        colptr_dict[key], row_dict[key], perm_dict[key] = to_csc(store, device)

    return colptr_dict, row_dict, perm_dict


def to_weighted_hetero_csc(
    data: HeteroData,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, OptTensor]]:
    # Convert the heterogeneous graph data into a suitable format for sampling
    # (CSC format).
    # Returns dictionaries holding `colptr` and `row` indices as well as edge
    # permutations for each edge type, respectively.
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets are converted into single strings.
    colptr_dict, row_dict, perm_dict, edge_weight_dict = {}, {}, {}, {}

    for store in data.edge_stores:
        key = edge_type_to_str(store._key)
        colptr_dict[key], row_dict[key], perm_dict[key], edge_weight_dict[key] = to_weighted_csc(store, device)

    return colptr_dict, row_dict, perm_dict, edge_weight_dict

def HeteroSubgraph(data, node_mask_dict, weighted=False):
    
    new_data = deepcopy(data)
    for edge_type in data.edge_types:
           if edge_type[0] in node_mask_dict and edge_type[-1] in node_mask_dict:
                if weighted:
                    new_edge, new_edge_weight = bipartite_subgraph((node_mask_dict[edge_type[0]], node_mask_dict[edge_type[-1]]), data[edge_type].edge_index, data[edge_type].edge_weight)
                    new_data[edge_type].edge_weight = new_edge_weight
                else:
                    new_edge, _ = bipartite_subgraph((node_mask_dict[edge_type[0]], node_mask_dict[edge_type[-1]]), data[edge_type].edge_index)

                new_data[edge_type].edge_index = new_edge
                
    return new_data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


    