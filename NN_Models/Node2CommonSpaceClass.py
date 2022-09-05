import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from typing import List,Tuple
import math

class Node2CommonSpace(nn.Module) :

    def __init__(self, common_space : int, node_type_info : List[Tuple[str,int]]) :
        
        super(Node2CommonSpace,self).__init__()

        self.common_space = common_space
        self.node_type_info = node_type_info
        self.node_matrices = {}

        for node_,node_dim in node_type_info :
            self.node_matrices[node_] = nn.Parameter( torch.Tensor(self.common_space,node_dim) )
            self.node_matrices[node_ + "_bias"] = nn.Parameter( torch.Tensor(self.common_space,1).squeeze() )
    
            self.reset_weight(self.node_matrices[node_])
            self.reset_bias(self.node_matrices[node_],self.node_matrices[node_+"_bias"])

        self.node_matrices = nn.ParameterDict(self.node_matrices)

    def reset_bias(self,weight,bias) -> None :
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)

    def reset_weight(self,weight) -> None:
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        return weight          

    def forward(self, node_matrix : List[torch.tensor], node_types : List[str]) :
        num_nodes = len(node_types)
        ans = torch.zeros((num_nodes,self.common_space))
        for idx,type_ in enumerate(node_types) :
            weight = self.get_node_type_matrix(type_)
            bias = self.get_node_type_matrix(type_+"_bias")
            input_ = node_matrix[idx].float()
            if len(input_.size()) == 1:
                input_ = torch.unsqueeze(input_,dim=0)
            ans[idx] = F.leaky_relu_(F.linear(input_,weight,bias).squeeze(),0.3)
                
        return ans 

    def get_node_type_matrix(self, node_type:str) :
        return self.node_matrices[node_type]

    def get_matrix(self,node_type:str) :
        ans = self.node_matrices[node_type]
        return ans

    def get_common_space(self) -> int :
        return self.common_space

    def get_common_node_space(self,node) :
        node_type = node.get_node_type()
        node_features = node.get_node_features().float()
        return self.forward([node_features],[node_type])







