import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from typing import List,Tuple
import math

class CommonSpace2Edge(nn.Module) :

    def __init__(self, common_space:int, edge_type_info : List[Tuple[str,int]]) :
        
        super(CommonSpace2Edge,self).__init__()

        #print(edge_type_info)

        self.common_space = common_space
        self.edge_type_info = edge_type_info
        self.edge_matrices = {}

        for edge_type,edge_dim in edge_type_info :
            self.edge_matrices[edge_type] = nn.Parameter( torch.Tensor(edge_dim,self.common_space) )
            self.edge_matrices[edge_type + "_bias"] = nn.Parameter( torch.Tensor(edge_dim,1).squeeze() )
    
            self.reset_weight(self.edge_matrices[edge_type])
            self.reset_bias(self.edge_matrices[edge_type],self.edge_matrices[edge_type+"_bias"])

        self.edge_matrices = nn.ParameterDict(self.edge_matrices)

    def get_matrix(self,edge_type:str) :
        ans = self.edge_matrices[edge_type]
        return ans

    def reset_bias(self,weight,bias) -> None :
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)

    def reset_weight(self,weight) -> None:
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        return weight

    def forward(self, edge_features : torch.tensor, edge_types : List[str]) -> List[torch.tensor] :
        ans = []
        for idx,type_ in enumerate(edge_types) :
            weight = self.get_matrix(type_)
            bias = self.get_matrix(type_+"_bias")
            input_ = edge_features[idx].float()
            if len(input_.size()) == 1:
                input_ = torch.unsqueeze(input_,dim=0)
            #ans.append(F.leaky_relu_(F.linear(input_,weight,bias).squeeze(),0.3))
            ans.append(F.linear(input_,weight,bias).squeeze())
        return ans 