from Nodes.NodeClass import Node
from Edges.EdgeClass import Edge
from NN_Models.CompositeModelClass import CompositeModel
from NN_Models.Losses.loss_fns import compute_attribute_loss,compute_structure_loss
from NN_Models.Losses.kmeans import Clustering
from data_layer.DataModel import DataModel
from convert_outputs import OutputFormatter
from typing import List
import torch
from torch import optim
import time
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import json
import os
import numpy as np
import sys
import pickle
import argparse
from metric import get_modularity,get_ned,get_coverage,get_structural_modularity
from time import sleep

class Graph(object) :

    def __init__(self, data_model : DataModel, model : CompositeModel, config:dict, k:int=9) :
        self.data_model = data_model
        self.nodes = data_model.get_nodes()
        self.edges = data_model.get_edges()
        self.model = model
        self.create_node_dict()
        self.config = config
        self.num_clusters = k

        if data_model.get_seeds() :
            self.node_mapping = self.get_node_mapping()
            self.seeds = data_model.get_seeds()
            # simple sanity check
            assert(len(self.seeds)-1 == self.num_clusters)
            self.kmeans = Clustering(seeds=self.get_seed_mapping())
            #print("Seeds Found")
            #assert(0)
        else :
            self.seeds = None
            self.node_mapping = None
            self.kmeans = Clustering(K=k)
            #print("No Seeds Found")
            #assert(0)

        self.traced_graph=False
        

    def get_seed_mapping(self) -> List[List[float]] :
        """Gets the idx mapping of every seed passed in the seed list.

        Returns:
            List[List[float]]: An array of array of indices that are seeds for each cluster.
        """
        #print(self.seeds)
        num_clusters = self.seeds["num_clusters"]
        seed_list = [[] for ctr in range(num_clusters)]
        for idx in range(num_clusters) :
            arr = self.seeds[idx]        
            seed_list[idx].extend([ self.node_mapping[ele][0] for ele in arr if self.node_mapping.get(ele,-1) != -1 ])
        #print("Seed List : {}".format(seed_list))
        return seed_list


    def get_node_mapping(self) :
        """ Gets the node mapping from CompositeModel : its format is {node_name : (node_index,node_object)} """
        self.model.prepare_mini_batch(self.edges)

        old_node_set = []
        for node in self.nodes :
            old_node_set.append(node.get_id())
        old_node_set = set(old_node_set)

        new_node_set = set(self.model.node_dict.keys())
        #print("Difference : {}".format(old_node_set-new_node_set))
        #assert(0)
        return self.model.get_node_dictionary()

    def create_node_dict(self) :
        self.node_dict = {}
        for node in self.nodes :
            self.node_dict[node.get_id()] = node

    def get_node(self,node_id:str) :
        return self.node_dict.get(node_id,None)

    def get_edge_label_match_accuracy(self,features,reconstructed) :
        """Checks how accuracte the model is about the labels predicted for each edge"""
        numerator=0
        denominator=0
        for feature_ele,recon_ele in zip(features,reconstructed) :
            if len(recon_ele) == 4 :
                labels = np.around((feature_ele).cpu().data.numpy(),2)
                predictions = np.around(torch.sigmoid(recon_ele).cpu().data.numpy(),2)
                predictions = np.round_(predictions)
                numerator += np.sum(labels==predictions)
                denominator += 4

        metric = numerator/(denominator+1)
        if type(metric) == float :
            metric = torch.tensor(metric)
        return metric

    def get_loss(self,input_node_matrix : List[torch.tensor],output_node_matrix : List[torch.tensor],encoded_state : torch.tensor,adj_matrix : torch.tensor,input_edge_features : List[torch.tensor]=None,output_edge_features : List[torch.tensor]=None,edge_loss_alpha : float = 0.,o_1 : torch.tensor=None,o_2 : torch.tensor=None) :

        # get the node loss
        node_loss = compute_attribute_loss(input_node_matrix,output_node_matrix,o_1,mode="Node")        

        # get the structure loss
        structure_loss = compute_structure_loss(adj_matrix, encoded_state, o_2)
        
        # get the clustering loss
        self.kmeans.cluster(encoded_state)
        if self.seeds :
            clustering_loss = self.kmeans.get_loss(encoded_state)
            clustering_loss = clustering_loss - self.kmeans.get_seed_loss(encoded_state)
        else :
            clustering_loss = self.kmeans.get_loss(encoded_state)

        # edge loss
        edge_loss = None
        if edge_loss_alpha>0 and not(input_edge_features is None) and not(output_edge_features is None) : 
            # edge_loss = compute_edge_loss(input_edge_features,output_edge_features)
            edge_loss = compute_attribute_loss(input_edge_features,output_edge_features,mode="Edge")
            edge_metric = self.get_edge_label_match_accuracy(input_edge_features,output_edge_features)
            self.edge_metric = edge_metric
        

        return node_loss, structure_loss, clustering_loss, edge_loss

    def add_graph(self) :
        self.model.add_graph(self.writer)

    def perform_loop(self,edge_list : List[Edge],node_loss_alpha:float=0.4,structure_loss_alpha:float=0.4,edge_loss_alpha:float=0.2,clustering_loss_alpha:float=0.,training:bool=True,o_1 : torch.tensor=None,o_2 : torch.tensor=None,use_edge_weights:bool=False) :

        input_node_matrix,node_types,node_names,edge_index,input_edge_features,edge_types,adj_matrix,edge_weight = self.model.prepare_mini_batch(edge_list)

        if not use_edge_weights :
            edge_weight = torch.ones((len(edge_weight)))

        output_node_matrix, output_edge_features, encoded_state = self.model(input_node_matrix,node_types,edge_index,input_edge_features,edge_types,training,edge_weight=edge_weight)
        
        # some stuff for updating outliers
        self.input_node_matrix = input_node_matrix
        self.output_node_matrix = output_node_matrix

        self.encoded_state = encoded_state
        self.adj_matrix = adj_matrix

        if self.model.use_edge_features :
            self.input_edge_features = input_edge_features
            self.output_edge_features = output_edge_features
        else :
            self.input_edge_features = None
            self.output_edge_features = None

        self.node_names = node_names

        final_loss = None
        clusters= None
        if training :
            
            node_loss, structure_loss, clustering_loss, edge_loss = self.get_loss(input_node_matrix,output_node_matrix,encoded_state,adj_matrix,input_edge_features,output_edge_features,edge_loss_alpha=edge_loss_alpha,o_1=o_1,o_2=o_2)
            node_loss_component = node_loss_alpha * node_loss
            structure_loss_component = structure_loss_alpha * structure_loss
            clustering_loss_component = clustering_loss_alpha * clustering_loss
            if edge_loss is None :
                edge_loss_component = 0
            else :
                edge_loss_component = edge_loss_alpha * edge_loss

            # node loss
            self.node_loss_list.append((node_loss_component).item())
            #print("Node Loss : {}".format(node_loss_component))
            self.writer.add_scalar("Node-Loss",node_loss,global_step=self.epoch)

            # structure loss
            self.struct_loss_list.append((structure_loss_component).item())
            #print("Structure Loss : {}".format(structure_loss_component))
            self.writer.add_scalar("Struct-Loss",structure_loss,global_step=self.epoch)

            # clustering loss
            self.clustering_loss_list.append((clustering_loss_component).item())
            #print("Clustering Loss : {}".format(clustering_loss_component))
            self.writer.add_scalar("Cluster-Loss",clustering_loss,global_step=self.epoch)

            # edge loss
            if not (edge_loss is None) : 
                #self.edge_loss_list.append((edge_loss_component).item())
                self.edge_loss_list.append((self.edge_metric).item())
                if edge_loss_alpha > 0 :
                    # print("Edge Metric : {}".format(self.edge_metric))
                    # print("Edge Loss : {}".format(edge_loss.item()))
                    pass
                self.writer.add_scalar("Edge-Loss",edge_loss,global_step=self.epoch)
            else :
                edge_loss = torch.tensor([0.])

            # final loss
            if self.seeds :
                if clustering_loss_alpha > 0 :
                    # print("Subtracting the clustering loss")
                    pass
                final_loss =  node_loss_component +  structure_loss_component + clustering_loss_component + edge_loss_component
            else :
                final_loss =  node_loss_component +  structure_loss_component + clustering_loss_component + edge_loss_component
            
            #print("Final Loss : {}\n\n".format(final_loss))
            self.final_loss_list.append(final_loss.item())
        
        else :
            #print("In Test Mode")
            self.kmeans.cluster(encoded_state)
            clusters = self.kmeans.M

        return final_loss,node_names,clusters

    def train(self,epoch_start=0,epoch_end=300,node_loss_alpha:float=0.5,structure_loss_alpha:float=0.5,edge_loss_alpha:float=0.,clustering_loss_alpha:float=0,training:bool=True,lr:float=0.01,use_edge_weights:bool=False) :
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epoch_start,epoch_end):
            self.epoch = epoch
            print("\tEpoch : {}".format(epoch))
            self.model.train_func()
            loss,node_names,clusters = self.perform_loop(self.edges,node_loss_alpha,structure_loss_alpha,edge_loss_alpha,clustering_loss_alpha,training,None,None,use_edge_weights)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return node_names,clusters

    def get_cluster(self, node_names, cluster_names, run_num=None) :
        if run_num is None :
            run_num = 1

        cluster_dict = {}
        for cluster,node_ in zip(cluster_names,node_names) :
            cluster = str(cluster)
            if cluster_dict.get(cluster,-1) == -1:
                cluster_dict[cluster] = []
            cluster_dict[cluster].append(node_)

        #print(cluster_dict)
        
        json_file = "clusters" + self.config["code"] + ".json"
        if run_num == 1 :
            with open( os.path.join(self.data_model.basepath,json_file), "w")  as f:
                json.dump(cluster_dict,f,indent=4)
        elif run_num > 1 :
            with open( os.path.join(self.data_model.basepath,json_file[:-5]+"_{}".format(run_num)+".json"), "w")  as f:
                json.dump(cluster_dict,f,indent=4)

        icu_file = os.path.join(self.data_model.basepath,"temp",self.config["icu_path"])
        callgraph_file = os.path.join(self.data_model.basepath,"temp",self.config["callgraph_path"])
        service_file = os.path.join(self.data_model.basepath,"temp",self.config["service_path"])
        resource_file = os.path.join(self.data_model.basepath,"temp",self.config["db_path"])
        output_file = os.path.join(self.data_model.basepath,"clusters" + self.config["code"] + "_cma_format.json")
        if run_num > 1 :
            # if we are going to train the model multiple times
            output_file = output_file[:-5] + "_{}".format(run_num) + ".json"

        formatter = OutputFormatter(cluster_dict, icu_file, callgraph_file, service_file, resource_file)
        formatter.write_to_file(output_file)

        self.metric_calculation(run_num)

        return cluster_dict

    def metric_calculation(self,run_num=None) :

        if run_num is None :
            run_num = 1

        if run_num == 1 :
            result_file = os.path.join(self.data_model.basepath,"clusters" + self.config["code"] + "_cma_format.json")
            mod = get_modularity(result_file)
            ned = get_ned(result_file)
            coverage = get_coverage(result_file)
            struct_mod = get_structural_modularity(result_file)
        else :
            result_file = os.path.join(self.data_model.basepath,"clusters" + self.config["code"] + "_cma_format_{}.json".format(run_num))
            mod = get_modularity(result_file)
            ned = get_ned(result_file)
            coverage = get_coverage(result_file)
            struct_mod = get_structural_modularity(result_file)
        
        
        if run_num == 1 :
            # first time
            with open(os.path.join(self.data_model.basepath,"metrics.txt"), "w") as f :
                f.write("Run No\tMod\tNED\tStruct Mod\tCoverage\n{}\t{}\t{}\t{}\t{}\n".format(run_num,mod,ned,struct_mod,coverage))
        else :
            with open(os.path.join(self.data_model.basepath,"metrics.txt"), "a") as f :
                f.write("{}\t{}\t{}\t{}\t{}\n".format(run_num,mod,ned,struct_mod,coverage))

    def test(self,run_num=None) :
        with torch.no_grad() :
            self.model.eval()
            loss,node_names,clusters = self.perform_loop(self.edges,training=False)
        ans = self.get_cluster(node_names,clusters,run_num)
        return ans

    def training_strategy(self,use_edge_weights:bool=False,run_num=None) :

        self.node_loss_list = []
        self.struct_loss_list = []
        self.edge_loss_list = []
        self.clustering_loss_list = []
        self.final_loss_list = []

        self.writer = SummaryWriter("./runs/visuals")

        ### pre-training phase ###
        node_names,clusters = self.train(0,100,node_loss_alpha=self.config["pre_train"]["node_loss_alpha"],structure_loss_alpha=self.config["pre_train"]["structure_loss_alpha"],edge_loss_alpha=self.config["pre_train"]["edge_loss_alpha"],clustering_loss_alpha=self.config["pre_train"]["clustering_loss_alpha"],lr=self.config["pre_train"]["lr"],use_edge_weights=use_edge_weights)
        #print("\n\n\n")        
        ### training phase ###
        node_names,clusters = self.train(100,200,node_loss_alpha=self.config["train"]["node_loss_alpha"],structure_loss_alpha=self.config["train"]["structure_loss_alpha"],edge_loss_alpha=self.config["train"]["edge_loss_alpha"],clustering_loss_alpha=self.config["train"]["clustering_loss_alpha"],lr=self.config["train"]["lr"],use_edge_weights=use_edge_weights)

        self.writer.close()
        return self.test(run_num)


if __name__ == '__main__' :

    import numpy as np
    from NN_Models.Node2CommonSpaceClass import Node2CommonSpace
    from NN_Models.CommonSpace2NodeClass import CommonSpace2Node
    from NN_Models.Edge2CommonSpaceClass import Edge2CommonSpace
    from NN_Models.CommonSpace2EdgeClass import CommonSpace2Edge

    sys.setrecursionlimit(10**6)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",help="model type. ex. AE_EGCN_Separate")
    parser.add_argument("--data", help="name of the app. ex. acme")
    parser.add_argument("--code",help="one of ['with_edge_loss','without_edge_loss']")
    parser.add_argument("--saved_data_model",help="boolean flag to re-use the pre-processed data. One of ['true','false']",default='false')

    args = parser.parse_args()

    model = args.model
    data = args.data
    code = args.code
    saved_data_model = args.saved_data_model.lower()

    if saved_data_model == 'false' :
        saved_data_model = False
    else :
        saved_data_model = True

    basepath = os.path.join(os.getcwd(),"data_layer/utilities/data/{}_{}/".format(data,model))

    # load the config
    with open( os.path.join(basepath,"custom/gcnconfig_" + code + ".json"), "r") as f :
        config = json.load(f)

    # load the parameters
    node_dimension = config["node_dimension"]
    num_clusters = config["num_clusters"]
    common_space = config["model_config"]["common_space"]

    multiple_runs=30
    for i in range(1,multiple_runs+1) : 

        print("************** RUN ", i, " **************")

        if saved_data_model :
            # re-use the preprocessed data
            dm_saving_path = os.path.join(basepath,"data_model.pkl")
            with open(dm_saving_path,"rb") as f :
                data_model = pickle.load(f)
            #print("Re-used the data model")
            sleep(5)
        else :
            # instantiate the data model
            data_model = DataModel(basepath,"gcnconfig_" + code + ".json",False)
            data_model.compute_features()
            data_model.collect_nodes_and_edges()
            #assert(0)

            # save the data model
            if i == 1:
                dm_saving_path = os.path.join(data_model.basepath,"data_model.pkl")
            else :
                dm_saving_path = os.path.join(data_model.basepath,"data_model_{}.pkl".format(i))
            with open(dm_saving_path,"wb") as f :
                pickle.dump(data_model,f)
        
        #print("Number of nodes : {}".format(len(data_model.nodes)))
        #print("Number of edges : {}".format(len(data_model.edges)))

        # procedure to create the GNN model and the Graph object
        node_type_info = [("program",node_dimension),("resource",node_dimension)]
        n_two_cs = Node2CommonSpace(common_space,node_type_info)
        cs_two_n = CommonSpace2Node(common_space,node_type_info)

        edge_type_info = [("CALLS",2),("CRUD",4)]
        e_two_cs = Edge2CommonSpace(common_space,edge_type_info)
        cs_two_e = CommonSpace2Edge(common_space,edge_type_info)

        #print(data_model.config)
        gnn_model = CompositeModel(n_two_cs,e_two_cs,cs_two_n,cs_two_e,model,data_model.config)

        #print(gnn_model.parameters)
        #print("==========")

        graph = Graph(data_model,gnn_model,data_model.config,num_clusters)

        # train the gnn
        ans = graph.training_strategy(use_edge_weights=False,run_num=i)
        
        # save the gnn
        if i > 1 :
            saving_path = os.path.join(data_model.basepath,"params_{}.pt".format(i))
            torch.save({'gnn_model_state_dict': gnn_model.state_dict()}, saving_path)
        else :
            saving_path = os.path.join(data_model.basepath,"params.pt")
            torch.save({'gnn_model_state_dict': gnn_model.state_dict()}, saving_path)