import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch.nn.init import xavier_uniform_, zeros_


def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU(init=0.5)
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError


class EGCNConv(MessagePassing):
    # form https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv
    def __init__(self, in_channels, out_channels,
                 edge_channels, edge_mode, activation:str, 
                 shared_weight:bool,improved=False, 
                 cached=False,bias=True, **kwargs):
        super(EGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.edge_mode = edge_mode
        self.activation_layer = get_activation(activation)
        self.shared_weight = shared_weight

        if shared_weight :
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        else :
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
            self.edge_weight = Parameter(torch.Tensor(edge_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(shared_weight)

    def reset_parameters(self,shared_weight:bool):
        xavier_uniform_(self.weight)
        if not shared_weight :
            xavier_uniform_(self.edge_weight)
        zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    def norm(self,edge_index, num_nodes, edge_weight=None, improved=False,dtype=None):

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)+1
        self.deg = deg

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        weight_val = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, weight_val


    def forward(self, x, edge_attr, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
        if self.shared_weight :
            edge_attr = torch.matmul(edge_attr, self.weight)
        else :
            edge_attr = torch.matmul(edge_attr, self.edge_weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)


    def message(self, x_i, x_j, edge_attr, norm):
        if self.edge_mode == 0:
            attention = self.attention_lin(torch.cat((x_i,x_j, edge_attr),dim=-1))
            m_j = attention * x_j
        elif self.edge_mode == 1:
            m_j = (x_j) * torch.sigmoid(edge_attr)

        return norm.view(-1, 1) * m_j

    def update(self, aggr_out, x):
        
        if self.bias is not None:
            
            
            
            aggr_out = aggr_out + self.bias + (x * torch.unsqueeze(self.deg.pow(-1.),dim=1))
        if self.edge_mode == 0:
            aggr_out = aggr_out + x
        return self.activation_layer(aggr_out)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)