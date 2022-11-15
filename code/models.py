import torch
from torch_geometric.nn import HeteroConv
from torch_geometric.nn import SAGEConv, Linear
import torch.nn.functional as F


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, num_features, sage_aggr='add', project=False, dropout=False, normalize=False):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.project = project
        self.dropout = dropout
        self.sage_aggr = sage_aggr
        self.num_features  = num_features
        self.hidden_channels = hidden_channels
        self.normalize = normalize
        
        first_conv = SAGEConv((self.num_features, self.num_features), self.hidden_channels, aggr=self.sage_aggr, project=self.project, normalize=self.normalize)
           
        self.convs.append(first_conv)

        for _ in range(num_layers-1):
            self.convs.append(SAGEConv((self.hidden_channels, self.hidden_channels), hidden_channels, aggr=self.sage_aggr, project=self.project, normalize=self.normalize))
        

        self.linear = Linear(self.hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)

        return x



class HeteroGNN_concat(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, sage_aggr='add', project=False):
        super().__init__()

        #self.convs = torch.nn.ModuleList()
        self.conv1 = HeteroConv({
            ('cardholder', 'pays', 'transaction'): SAGEConv((1,85), hidden_channels, aggr=sage_aggr, project=project),
            ('merchant', 'receives', 'transaction'): SAGEConv((1, 85), hidden_channels, aggr=sage_aggr,project=project),
            ('transaction', 'rev_pays', 'cardholder'): SAGEConv((85, 1), hidden_channels, aggr=sage_aggr,project=project),
            ('transaction', 'rev_receives', 'merchant'): SAGEConv((85, 1), hidden_channels, aggr=sage_aggr,project=project)
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('cardholder', 'pays', 'transaction'): SAGEConv((hidden_channels,hidden_channels), hidden_channels, aggr=sage_aggr, project=project),
            ('merchant', 'receives', 'transaction'): SAGEConv((hidden_channels,hidden_channels), hidden_channels, aggr=sage_aggr, project=project),
            ('transaction', 'rev_pays', 'cardholder'): SAGEConv((hidden_channels,hidden_channels), hidden_channels, aggr=sage_aggr, project=project),
            ('transaction', 'rev_receives', 'merchant'): SAGEConv((hidden_channels,hidden_channels), hidden_channels, aggr=sage_aggr, project=project)
        }, aggr=None)
        
        self.lin = Linear(128, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        x_dict['transaction'] = torch.reshape(x_dict['transaction'], (x_dict['transaction'].size()[0],128))
        x_dict['merchant'] = torch.reshape(x_dict['merchant'], (x_dict['merchant'].size()[0],64))
        x_dict['cardholder'] = torch.reshape(x_dict['cardholder'], (x_dict['cardholder'].size()[0],64))

        return self.lin(x_dict['transaction'])


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, num_features, metadata, sage_aggr='mean', hetero_aggr = 'sum', project=False, dropout=False, normalize=False):
        super().__init__()

        self.project = project
        self.dropout = dropout
        self.sage_aggr = sage_aggr
        self.hetero_aggr = hetero_aggr
        self.hidden_channels = hidden_channels
        self.normalize = normalize

        self.convs = torch.nn.ModuleList()
        edge_types = metadata[1]
        first_conv = HeteroConv({ edge_type: SAGEConv((num_features[edge_type[0]],num_features[edge_type[2]]), hidden_channels, aggr=sage_aggr, project=project, normalize=self.normalize) for edge_type in edge_types}, aggr='sum')
           
        self.convs.append(first_conv)

        for _ in range(num_layers-1):
            conv = HeteroConv({
                ('cardholder', 'pays', 'transaction'): SAGEConv((hidden_channels,hidden_channels), hidden_channels, aggr=self.sage_aggr, project=self.project, normalize=self.normalize),
                ('merchant', 'receives', 'transaction'): SAGEConv((hidden_channels,hidden_channels), hidden_channels, aggr=self.sage_aggr,project=self.project, normalize=self.normalize),
                ('transaction', 'rev_pays', 'cardholder'): SAGEConv((hidden_channels,hidden_channels), hidden_channels, aggr=self.sage_aggr,project=self.project, normalize=self.normalize),
                ('transaction', 'rev_receives', 'merchant'): SAGEConv((hidden_channels,hidden_channels), hidden_channels, aggr=self.sage_aggr,project=self.project, normalize=self.normalize),
            }, aggr=self.hetero_aggr)
            self.convs.append(conv)

        
        self.lin = Linear(-1, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for layer, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            if self.dropout:
                x_dict = {key: F.dropout(x, p=0.5, training=self.training)  for key, x in x_dict.items()}
            if (self.hetero_aggr == None) & (layer > 0):
                # these tensors have one extra dimensions because no hetero_aggr was applied. 
                # These tensors are reshaped in order to concatenate the last two dimensions into one.
                x_dict['transaction'] = torch.reshape(x_dict['transaction'], (-1,2*self.hidden_channels))
                x_dict['merchant'] = torch.reshape(x_dict['merchant'], (-1,self.hidden_channels))
                x_dict['cardholder'] = torch.reshape(x_dict['cardholder'], (-1,self.hidden_channels))

        return self.lin(x_dict['transaction'])

from typing import Tuple, Union

from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class GraphConvCustom(MessagePassing):
    r"""The graph neural network operator from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot \mathbf{x}_j

    where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
    target node :obj:`i` (default: :obj:`1`)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = 'add',
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_rel = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_root = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_rel(out)

        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_root(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        #print('message')
        #print('shape', edge_weight.view(-1, 1).size())
        #print('sum', edge_weight.view(-1, 1).sum())
        #print(edge_weight.view(-1, 1)/edge_weight.view(-1, 1).sum())
        return x_j if edge_weight is None else (edge_weight.view(-1, 1) * x_j) / edge_weight.view(-1, 1).sum()

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)