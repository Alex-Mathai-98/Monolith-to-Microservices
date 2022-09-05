from typing import List
from Nodes.NodeClass import Node
import numpy as np
import torch

class Edge() :
    def __init__(self, edge_type:str, 
                    src_node : Node, 
                    tgt_node : Node,
                    edge_features : np.ndarray) :
        
        self.edge_type = edge_type
        self.src_node = src_node
        self.tgt_node = tgt_node
        # what does edge_features do ??
        self.edge_features = edge_features

    def get_role(self) :
        if self.edge_type == "CRUD" :
            assert(self.edge_features.shape[0] == 4)
            if np.sum(self.edge_features*np.array([1.,0.,1.,1.])) > 0. :
                return "updater"
            else :
                return "reader"
        else :
            return "NA"

    def set_edge_weight(self,weight:float) :
        self.weight = weight

    def get_edge_weight(self) :
        return self.weight

    def get_edge_features(self) :
        return torch.tensor(self.edge_features)

    def get_type(self) -> str :
        return self.edge_type

    def get_other_node(self,node:Node) -> Node :
        node_id = node.get_id()
        if self.src_node.node_id == node_id :
            return self.tgt_node
        elif self.tgt_node.node_id == node_id :
            return self.src_node
        else :
            return None


