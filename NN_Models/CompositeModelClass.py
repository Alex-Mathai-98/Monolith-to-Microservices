from Nodes.NodeClass import Node
from Edges.EdgeClass import Edge
from NN_Models.Node2CommonSpaceClass import Node2CommonSpace
from NN_Models.CommonSpace2NodeClass import CommonSpace2Node
from NN_Models.Edge2CommonSpaceClass import Edge2CommonSpace
from NN_Models.CommonSpace2EdgeClass import CommonSpace2Edge
import torch
from torch import nn
import numpy as np
from typing import List
from NN_Models.GNNModel.model import AE_EGCN_Separate,AE_EGCN_Shared

class CompositeModel(nn.Module) :

    def __init__(self, node_model : Node2CommonSpace, 
                    edge_model : Edge2CommonSpace,
                    reverse_node_model : CommonSpace2Node,
                    reverse_edge_model : CommonSpace2Edge,
                    strategy : str,
                    config : dict):

        super(CompositeModel,self).__init__()
    
        self.node_model = node_model
        self.edge_model = edge_model
        self.strategy = strategy
        self.config = config


        hidden_layer1 = self.config["model_config"]["hidden1"]
        hidden_layer2 = self.config["model_config"]["hidden2"]


        # if strategy in ["homogenous","Homo_Edges_GCNAE","Homo_Nodes_GCNAE"] :
        #     self.gnn_model = GCNAE(self.node_model.get_common_space(), hidden_layer1, hidden_layer2, 0.1)
        #     self.use_edge_features=False
        if strategy == "AE_EGCN_Separate" :
            #print(hidden_layer1)
            #print(hidden_layer2)
            self.gnn_model = AE_EGCN_Separate(self.node_model.get_common_space(), hidden_layer1, hidden_layer2)
            self.use_edge_features=True
        elif strategy == "AE_EGCN_Shared" : 
            self.gnn_model = AE_EGCN_Shared(self.node_model.get_common_space(), hidden_layer1, hidden_layer2)
            self.use_edge_features=True
        
        self.reverse_node_model = reverse_node_model
        self.reverse_edge_model = reverse_edge_model
        self.plotted_graph=True

        # data pre-processing must be done once 
        self.node_matrix = None
        self.node_types = None
        self.node_names = None
        self.edge_matrix = None
        self.edge_features = None
        self.edge_types = None
        self.adj_matrix = None
        self.done_once = False

    def get_node_type_info(self) :
        """ Gets node type info : ex. [("program",10),("resource",5)] """
        return self.node_model.node_type_info

    def get_edge_type_info(self) :
        """ Gets edge type info : ex. [("CALLS",2),("CRUD",4)] """
        return self.edge_model.edge_type_info

    def original_to_commonspace(self,node_matrix,node_types,edge_features,edge_types) :
        node_matrix = self.node_model(node_matrix,node_types)
        edge_features = self.edge_model(edge_features,edge_types)
        return node_matrix,edge_features

    def commonspace_to_original(self,node_matrix,node_types,edge_features,edge_types) :
        node_ans = self.reverse_node_model(node_matrix,node_types)
        edge_ans = self.reverse_edge_model(edge_features,edge_types)
        return node_ans,edge_ans

    def add_graph(self,writer):
        self.plotted_graph=False
        self.writer = writer

    def train_func(self) :
        self.node_model.train()
        self.edge_model.train()
        self.gnn_model.train()
        self.reverse_node_model.train()
        self.reverse_edge_model.train()

    def eval(self) :
        self.node_model.eval()
        self.edge_model.eval()
        self.gnn_model.eval()
        self.reverse_node_model.eval()
        self.reverse_edge_model.eval()

    def forward(self,input_node_matrix : List[torch.tensor],
                    node_types : List[str], 
                    adj_matrix : torch.tensor, 
                    input_edge_features : List[torch.tensor],
                    edge_types : List[str],
                    training:bool=True,
                    edge_weight : List[float] = None) :

        node_matrix, edge_features = self.original_to_commonspace(input_node_matrix,node_types,input_edge_features,edge_types)

        #print("====> Node Matrix <====")
        #print(node_matrix.size())
        

        #print("====> Edge Matrix <====")
        #print(adj_matrix.size())
        

        #print("====> Edge Features <====")
        #print(edge_features.size())
        

        if self.use_edge_features==False :
            # Pass through the model
            decoded_state,encoded_state = self.gnn_model(node_matrix,adj_matrix,training)
            #print("====> Ans <====")
            #print("Encoded State Size : {}".format(encoded_state.size()))
            #print("Decoded State Size : {}".format(decoded_state.size()))
            

            # Convert back to original space
            node_ans, edge_ans = self.commonspace_to_original(decoded_state,node_types,edge_features,edge_types)
            edge_ans = None

        else :
            # Pass through the model
            decoded_state, encoded_state, output_edge_features = self.gnn_model(node_matrix, edge_features, adj_matrix, edge_weight)
            #print("====> Ans <====")
            #print("Encoded State Size : {}".format(encoded_state.size()))
            #print("Decoded State Size : {}".format(decoded_state.size()))
            #print("Edge Features Size : {}".format(output_edge_features.size()))

            # Convert back to original space
            node_ans, edge_ans = self.commonspace_to_original(decoded_state,node_types,output_edge_features,edge_types)

        if self.plotted_graph==False :
            # self.writer.add_graph(self.gnn_model, (node_matrix, edge_features,adj_matrix))
            self.plotted_graph=True

        return node_ans, edge_ans, encoded_state 


    def prepare_adjacency_matrix(self, edge_index:torch.tensor, num_nodes:int) -> torch.tensor :

        ground_truth = np.zeros((num_nodes,num_nodes))
        src_s = edge_index[0]
        dst_s = edge_index[1]
        for src,dst in zip(src_s,dst_s) :
            ground_truth[src][dst] = 1.0

        if self.strategy in ["homogenous","AE_GCNConv","AE_EGCN_Shared","AE_EGCN_Separate"] :
            #print("Pre-process Adjacency Matrix")
            ground_truth = ground_truth + np.eye(ground_truth.shape[0])
            rowsum = np.array(ground_truth.sum(1))
            degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5))
            ground_truth =  np.dot(np.dot(degree_mat_inv_sqrt, ground_truth),degree_mat_inv_sqrt).astype(np.float32)
        elif self.strategy in ["AE_GRAPE"] :
            print("Keep Ajacency Matrix Untouched")

        ground_truth = torch.tensor(ground_truth)
        return ground_truth

    def get_node_dictionary(self) :
        """ Returns the node dictionary to Graph Object """
        return self.node_dict

    def get_reverse_node_dictionary(self) :
        """ Returns the reverse node dictionary to Graph Object """
        return self.reverse_node_dict

    def prepare_mini_batch(self, edge_list : List[Edge]) :

        with torch.no_grad() :

            if self.done_once :
                #print("Re-using the pre-processed data")
                node_matrix = self.node_matrix
                node_types = self.node_types
                node_names = self.node_names
                edge_matrix = torch.tensor(self.edge_matrix)
                edge_features = self.edge_features
                edge_types = self.edge_types
                adj_matrix = torch.tensor(self.adj_matrix)
                edge_weight = torch.tensor(self.edge_weight)

            else :
                # preprocessing the data for the first time

                node_dict = {} # stores all the nodes in the format : {node_name : (node_index,node_object)}
                reverse_node_dict = {}
                node_id = -1

                ############ Creating the Adjacency List : dimension is [2,number of edges] and the Node dictionary ############
                edge_matrix = torch.zeros((2,len(edge_list))).long()
                for edge_idx,edge in enumerate(edge_list) :
                    # source node
                    src_node = edge.src_node
                    src_id = src_node.get_id()
                    if node_dict.get(src_id,-1) == -1 :
                        node_id += 1
                        node_dict[src_id] = (node_id,src_node)
                        reverse_node_dict[node_id] = src_id
                    # target node
                    tgt_node = edge.tgt_node
                    tgt_id = tgt_node.get_id()
                    if node_dict.get(tgt_id,-1) == -1 :
                        node_id += 1
                        node_dict[tgt_id] = (node_id,tgt_node)
                        reverse_node_dict[node_id] = tgt_id
                    # fill the edge matrix
                    edge_matrix[0,edge_idx] = node_dict[src_id][0]
                    edge_matrix[1,edge_idx] = node_dict[tgt_id][0]
                
                # assert(False)

                # saving the node dictionary 
                #print(node_dict)
                self.node_dict = node_dict
                self.reverse_node_dict = reverse_node_dict

                ############ Creating the node features ############
                node_keys = []
                node_vals = []
                for node_key,node_val in node_dict.items() :
                    node_keys.append(node_key)
                    node_vals.append(node_val[0])

                node_keys = np.array(node_keys)
                node_vals = np.array(node_vals)
                #print("Node Keys : {}".format(node_keys))
                #print("Node Values : {}".format(node_vals))
                sorted_order = np.argsort(node_vals)

                # sort the result
                node_keys = node_keys[sorted_order]
                node_vals = node_vals[sorted_order]

                node_matrix = []
                node_types = []
                node_names = []
                for node_key,node_val in zip(node_keys,node_vals) :
                    node_matrix.append(node_dict[node_key][1].get_node_features())
                    if self.strategy == "Homo_Nodes_GCNAE" :
                        node_types.append("program")
                    else :
                        node_types.append(node_dict[node_key][1].get_node_type())
                    node_names.append(node_dict[node_key][1].get_id())
                    
                num_nodes = len(node_dict)
                assert(len(node_matrix)==num_nodes)

                ############ Creating the edge_features ############
                edge_features = []
                edge_types = []
                edge_weight = []
                for edge_idx,edge in enumerate(edge_list) : 
                    if self.strategy == "Homo_Edges_GCNAE" :
                        edge_types.append("CALLS")
                        edge_features.append(torch.tensor([1.,0.]))
                    else :
                        edge_types.append(edge.get_type())
                        edge_features.append(edge.get_edge_features())
                    
                    edge_weight.append(edge.get_edge_weight())
                edge_weight = torch.tensor(edge_weight)

                for i in range(len(edge_list)) :
                    src = node_names[edge_matrix[0,i]]
                    tgt = node_names[ edge_matrix[1,i] ]
                    #print(src + "==>" + tgt)
                    if torch.sum(edge_features[i]) > 1. :
                        pass
                        #print(edge_types[i])
                        #print(edge_features[i])
                        #print("\n\n\n")

                # prepate the adjacency matrix
                adj_matrix = self.prepare_adjacency_matrix(edge_matrix,len(node_types))

                self.node_matrix = node_matrix
                self.node_types = node_types
                self.node_names = node_names
                self.edge_matrix = edge_matrix.detach().cpu().numpy()
                self.edge_features = edge_features
                self.edge_types = edge_types
                self.adj_matrix = adj_matrix.detach().cpu().numpy()
                self.done_once = True
                self.edge_weight = edge_weight.detach().cpu().numpy()

        return node_matrix,node_types,node_names,edge_matrix,edge_features,edge_types,adj_matrix,edge_weight
    