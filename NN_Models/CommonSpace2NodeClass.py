import torch
import math
from torch import nn
import torch.nn.functional as F
from typing import List,Tuple
from torch.nn import init

class CommonSpace2Node(nn.Module) :

    def __init__(self, common_space : int, node_type_info : List[Tuple[str,int]]) :
        
        super(CommonSpace2Node,self).__init__()

        self.common_space = common_space
        self.node_type_info = node_type_info
        self.node_matrices = {}

        for node_,node_dim in node_type_info :
            self.node_matrices[node_] = nn.Parameter( torch.Tensor(node_dim,self.common_space) )
            self.node_matrices[node_ + "_bias"] = nn.Parameter( torch.Tensor(node_dim,1).squeeze() )
    
            self.reset_weight(self.node_matrices[node_])
            self.reset_bias(self.node_matrices[node_],self.node_matrices[node_+"_bias"])

        self.node_matrices = nn.ParameterDict(self.node_matrices)

    def forward(self, node_matrix : torch.tensor, 
                node_types : List[str]) -> List[torch.tensor] :
        
        node_ans = []
        for idx,type_ in enumerate(node_types) :
            weight = self.get_node_type_matrix(type_)
            bias = self.get_node_type_matrix(type_+"_bias")
            input_ = node_matrix[idx].float()
            if len(input_.size()) == 1:
                input_ = torch.unsqueeze(input_,dim=0)
            node_ans.append(F.leaky_relu_(F.linear(input_,weight,bias).squeeze(),0.3))
        
        return node_ans 

    def reset_bias(self,weight,bias) -> None :
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)

    def reset_weight(self,weight) -> None:
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        return weight

    def get_node_type_matrix(self, node_type:str) :
        return self.node_matrices[node_type]

    def get_common_space(self) -> int :
        return self.common_space







