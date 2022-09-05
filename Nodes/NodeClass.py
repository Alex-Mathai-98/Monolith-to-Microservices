from typing import List
import numpy as np
import torch

class Node(object) :

    def __init__(self, node_id : str,
                    node_type :str, 
                    node_features : np.ndarray) :
        
        self.node_id = node_id
        self.node_type = node_type
        self.in_edges = []
        self.out_edges = []
        self.node_features = node_features

    def add_in_edge(self, edge) :
        self.in_edges.append(edge)

    def add_out_edge(self, edge) :
        self.out_edges.append(edge)

    def get_id(self) :
        return self.node_id
    
    def get_node_type(self):
        return self.node_type

    def get_node_features(self):
        return torch.tensor(self.node_features)

    def get_neighbouring_edges(self,edge_direction:str) :
        if edge_direction=="in":
            return self.in_edges
        elif edge_direction=="out":
            return self.out_edges
        else :
            return None







