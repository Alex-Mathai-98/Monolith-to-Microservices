from data_layer.utilities.monolith import Monolith
from typing import List
import numpy as np
from Nodes.NodeClass import Node
from Edges.EdgeClass import Edge
from data_layer.utilities import gcncmautilities as gcnutils
import os
import json
import pickle
from copy import deepcopy

class DataModel() :

    def __init__(self, basepath:str, config:dict, graph_directed:bool=True) :
        self.basepath = basepath
        self.config = self.read_config(config)
        self.app = self.create_app()
        self.nodes = []
        self.edges = []
        self.graph_directed = graph_directed

    def read_config(self,config_path:str):
        "Reads the config file"
        with open(os.path.join(self.basepath,"custom",config_path),"r") as f :
            json_file = json.load(f)
        return json_file

    def create_app(self) -> Monolith :
        "Creates an instance of the 'monolith' app"
        icu_file = os.path.join(self.basepath,"temp",self.config["icu_path"])
        callgraph_file = os.path.join(self.basepath,"temp",self.config["callgraph_path"])
        service_file = os.path.join(self.basepath,"temp",self.config["service_path"])
        resource_file = os.path.join(self.basepath,"temp",self.config["db_path"])
        seed_file = os.path.join(self.basepath,"custom",self.config["seeds_path"])
        self.create_seeds_dict(seed_file)
        build_cooc_matrix=True
        return Monolith(icu_file,callgraph_file,service_file, resource_file, build_cooc_matrix)

    def create_seeds_dict(self,seed_file:str) :
        """ Creates the list of artifacts that needs to be present in each cluster exclusively

        Args:
            seed_file (str): path to the seed file
        """

        try :
            self.seed_dict = {} 
            num_clusters = 0
            with open(seed_file,"r") as f :
                lines = f.readlines()
                num_clusters = len(lines)
                # for each cluster, save the list of seeds
                for idx,line in enumerate(lines) :
                    self.seed_dict[idx] = []
                    line = line.strip()
                    seeds = line.split(",")
                    for seed in seeds :
                        self.seed_dict[idx].append(seed.strip())
                    # in case the same seed appears twice in a seed set
                    self.seed_dict[idx] = list(set(self.seed_dict[idx]))
            # save the number of clusters
            self.seed_dict["num_clusters"] = num_clusters
            
        except Exception as e :
            self.seed_dict = None

        #print(self.seed_dict)
        

    def get_idxs_to_ignore(self) :
        "Creates a list of programs to ignore during the analysis"
        self.ignore_overrides = []
        ignorefiles_override = os.path.join(self.basepath,"custom",self.config["ignore_file"])
        if ignorefiles_override is not None:
            ignore_files = gcnutils.read_filecontents(ignorefiles_override)
            if len(ignore_files) > 0:
                self.ignore_overrides = ignore_files
                #print("Override Ignore Files : " + str(self.ignore_overrides))
        self.idxs_to_ignore = [self.c2imap[c] for c in self.ignore_overrides]

    def compute_struct(self) :
        """ Creates the struct.csv file which has the adjacency matrix """
        self.get_idxs_to_ignore()
        count_noICUconns = 0
        struct_matrix = []
        for idx, full_classname in enumerate(self.all_classes):
            row = np.zeros((self.num_classes,), dtype=np.int)
            for c in self.app.get_classes_used_by(full_classname):
                index = self.c2imap[c]
                if index not in self.idxs_to_ignore:
                    row[index] = 1
            for c in self.app.get_classes_using(full_classname):
                index = self.c2imap[c]
                if index not in self.idxs_to_ignore:
                    row[index] = 1
            row = row.reshape(1,-1)
            
            if np.sum(row) == 0:
                
                count_noICUconns += 1
            struct_matrix.append(row[0])
        #print(count_noICUconns, "classes/programs have no connections in the ICU.")

        # Prepare the matrix - we use double precision now, use float32 if memory becomes an issue
        struct_matrix = np.array(struct_matrix, dtype=np.float64)
        #print("Initial struct matrix:", struct_matrix.shape)
        # Identify classes not in the call graph
        self.unused_cooc_idxs = []
        for i, row in enumerate(self.app.graph.cooccurence_matrix):
            if i in self.idxs_to_ignore:
                continue
            if np.sum(row) == 0:
                self.unused_cooc_idxs.append(i)

        #print("Number of unused classes in Cooc matrix:", len(self.unused_cooc_idxs))
    
        self.struct_matrix = struct_matrix
        #print("Struct Matrix has been created")        

    def create_remove_node_dict(self) :
        """Creates the dictionary of nodes to be removed"""

        # remove the node
        #print("Created the remove_node dictionary")
        self.remove_node = {}
        for node_idx in self.idxs_to_remove :
            node_id = self.i2cmap[node_idx]
            self.remove_node[node_id] = 1

    def remove_zero_rows(self) :
        """ Removes the empty rows in the matrix. """

        #print("\nRemoving zero rows .. \n")
        #print("Initial shape:", self.features.shape)
        # Find idxs that need to be removed because we dont have any information about them
        unsuitable_class_idxs = gcnutils.analyze_outputs(self.features, self.struct_matrix, self.i2cmap)
        self.idxs_to_remove = list(set(unsuitable_class_idxs + self.idxs_to_ignore))
        self.create_remove_node_dict()
        #print("Row Idxs to remove : {}".format(self.idxs_to_remove))
        #print("To be Removed : {}".format(self.remove_node))
        # Remove the zero rows
        self.old_features = deepcopy(self.features)
        self.struct_matrix, self.features, self.idxs_to_remove = gcnutils.correct_matrices(self.struct_matrix, self.all_features, self.idxs_to_remove)
        #print("Rows have been removed")
        #print("New Shape : {}\n".format(self.features.shape))

    def remove_zero_columns(self) :
        """ Removes the empty columns in the matrix. """

        #print("\nRemoving zero columns ....\n")
        #print("Initial shape:", self.features.shape)
        zero_features = np.where(np.sum(self.features, axis=0) == 0)[0]
        
        self.unsuitable_feat_idxs = [i for i in zero_features]
        #print("Column Idxs to remove:", len(self.unsuitable_feat_idxs))
        
        # Remove the zero columns
        self.features = np.delete(self.features, self.unsuitable_feat_idxs, axis=1)
        self.old_features = np.delete(self.old_features, self.unsuitable_feat_idxs, axis=1)
        self.num_features = self.features.shape[1]
        #print("Columns have been removed")
        #print("New Shape : {}\n".format(self.features.shape))


    def compute_features(self) :
        """ Computes all the features for the GNN. """
        
        # feature list
        features_to_compute = self.config['gcn_features']

        # Identify classes to use
        self.all_classes = self.app.filtered_classes
        self.num_classes = len(self.all_classes)

        # Generate forward and reverse maps
        self.c2imap, self.i2cmap = {}, {}
        for idx,full_classname in enumerate(self.all_classes):
            self.c2imap[full_classname] = idx
            self.i2cmap[idx] = full_classname
        #print(self.c2imap)
        #print("\n\n")
        
        # compute structure.csv
        self.compute_struct()

        #print("\nGenerating features .. \n")
        inh_feat, pcc_feat, ep_feat, db_feat = None, None, None, None

        # inheritance features
        if "INH" in features_to_compute:
            inh_feat = gcnutils.compute_inherit_features(self.c2imap, self.app.implements)
        # co-occurence features
        if "PCC" in features_to_compute:
            pcc_feat = gcnutils.compute_pcc_features(self.c2imap, self.app.graph.cooccurence_matrix)
        # entry point features
        if "EP" in features_to_compute:
            ep_feat = gcnutils.compute_entrypoint_features(self.c2imap, self.all_classes, self.app.graph)

        all_features = (inh_feat, pcc_feat, ep_feat, db_feat)
        features_list = []
        for f in all_features:
            if f is not None:
                features_list.append(f)

        # approximate features for the missing classes using adj matrix
        self.all_features = all_features
        
        #self.all_features = gcnutils.approximate_features(all_features, self.struct_matrix, self.unused_cooc_idxs)
        
        self.features = np.concatenate(features_list, axis=1)

        # removing the zero rows
        self.remove_zero_rows()

        # removing the zero columns
        # self.remove_zero_columns()

        self.write_struct_file()
        self.write_content_file()

    def write_struct_file(self) :
        #print("Struct shape:", self.struct_matrix.shape)
        struct_filename = os.path.join(self.basepath,"temp",self.config['struct_path'])
        np.savetxt(struct_filename, self.struct_matrix, delimiter=",")

        with open(struct_filename[:-4]+".pkl", "wb") as f:
            pickle.dump(np.array(self.struct_matrix, dtype=np.float64), f)

    def write_content_file(self) :
        #print("Content/Features shape:", self.features.shape)
        self.num_features = self.features.shape[1]
        content_filename = os.path.join(self.basepath,"temp",self.config['content_path'])
        np.savetxt(content_filename, self.features, delimiter=",")

        with open(content_filename[:-4]+".pkl", "wb") as f:
            pickle.dump(np.array(self.features, dtype=np.float64), f)

    def get_edge_data(self,edge_type:str,edge_action:str) -> np.ndarray :

        if edge_type == "CALLS" :
            edge_data = np.array([1.,0.])
        
        elif edge_type == "IMPLEMENTS" :
            edge_data = np.array([0.,1.])
        
        # (res)
        # (p1,p2,p3 ...)

        elif edge_type == "CRUD" :
            if edge_action == "C" :
                edge_data = np.array([1.,0.,0.,0.])
            elif edge_action == "R" :
                edge_data = np.array([0.,1.,0.,0.])
            elif edge_action == "U" :
                edge_data = np.array([0.,0.,1.,0.])
            elif edge_action == "D" :
                edge_data = np.array([0.,0.,0.,1.])
            else :
                # multiple CRUD actions in one string. ex "CR" ==> create and read
                edge_action = edge_action.strip()
                actions = [action for action in edge_action]
                edge_data = np.array([0.,0.,0.,0.])
                pos_dict = {"C":0,"R":1,"U":2,"D":3}
                for action in actions :
                    edge_data[ pos_dict[action] ] = 1.0

        return edge_data

    def update_edge_data(self,edge_old:np.ndarray,edge_new:np.ndarray) -> np.ndarray :
        final_edge = edge_old + edge_new
        final_edge = np.sign(final_edge)
        return final_edge

    def create_edge(self,node_A:Node,node_B:Node,edge_direction:str,edge_type:str="CALLS",edge_action:str=None):
        """Creates an edge

        Args:
            node_A (Node): Source Node
            node_B (Node): Target Node
            edge_direction (str): "in" or "out"
            edge_type (str, optional): type of the edge. Defaults to "CALLS". Can be "CRUD"
            edge_action (str, optional): If edge type is "CRUD" then action can be "R","U","C" and "D". Defaults to None.

        Returns:
            [edge]: returns an edge
        """

        
        
        

        # examples of key 
        # LGICDB01_*_db2.customer-res
        key = node_A.get_id() + "_*_" + node_B.get_id()
        # examples of reverse key
        # db2.customer-res_*_LGICDB01
        reverse_key = node_B.get_id() + "_*_" + node_A.get_id()

        # if this edge or even the reverse of this edge has been populated.
        if (self.edge_dict.get(key,-1) != -1) and (self.edge_dict.get(reverse_key,-1) != -1) :

            # get old edge feature
            edge_old = self.edge_dict[key].edge_features
            # get the new edge feature
            edge_new = self.get_edge_data(edge_type,edge_action)

            # if a program reads and updates to the same resource
            # edge_old can be [0.,1.,0.,0.] (read array) and edge_new [0.,0.,1.,0.] (update array)
            # updated_edge will be [0.,1.,1.,0.] (read and update array)
            updated_edge = self.update_edge_data(edge_old,edge_new)

            if (not np.array_equal(updated_edge,edge_old)) :
                pass
                #print("Got a new CRUD operation")
                #print("Old array : {}".format(edge_old))
                #print("New array : {}".format(edge_new))
                #print("Updated array : {}\n".format(updated_edge))

            self.edge_dict[key].edge_features =  np.copy(updated_edge)
            self.edge_dict[reverse_key].edge_features = np.copy(updated_edge)

            return self.edge_dict.get(key,None)

        # if this edge or even the reverse of this edge has never been populated.
        elif (self.edge_dict.get(key,-1) == -1) and (self.edge_dict.get(reverse_key,-1) == -1):
            
            # create the edge
            edge_data = self.get_edge_data(edge_type,edge_action)
            edge = Edge(edge_type, node_A, node_B, edge_data)
            reverse_edge = Edge(edge_type, node_B, node_A, edge_data)

            self.edge_dict[key] = edge
            self.edge_dict[reverse_key] = edge

            if edge_direction == "in" :
                node_A.add_in_edge(edge)
                node_B.add_out_edge(edge)

                if self.graph_directed==False :
                    # if the graph is undirected
                    node_A.add_out_edge(edge)
                    node_B.add_in_edge(edge)

            elif edge_direction == "out" :
                node_A.add_out_edge(edge)
                node_B.add_in_edge(edge)

                if self.graph_directed==False :
                    # if the graph is undirected
                    node_A.add_in_edge(edge)
                    node_B.add_out_edge(edge)
            
            # add the edge only once
            self.edges.append(edge)

            if self.graph_directed==False :
                self.edges.append(reverse_edge)

            return edge

        # only the reverse edge is stored
        elif self.edge_dict.get(key,-1) == -1 :
            return self.edge_dict.get(reverse_key,None) 
        
        # only the forward edge is stored
        elif self.edge_dict.get(reverse_key,-1) == -1 :
            return self.edge_dict.get(key,None)

    def create_node(self, node_id:str, node_type:str, node_data:np.ndarray=np.ones((10))) -> Node :
        """ Creates a node with name=node_id and type=node_type and data=node_data. """
        if self.node_dict.get(node_id,-1) == -1:
            self.node_dict[node_id] = Node(node_id,node_type,node_data)
            self.nodes.append(self.node_dict[node_id])
        return self.node_dict[node_id]
    
    def delete_node(self, node_id:str) -> None :
        if self.node_dict.get(node_id,-1) != -1:
            del self.node_dict[node_id]

    def get_node_feature(self, node_id:str) -> np.ndarray :
        node_name = self.parse_real_name(node_id)
        node_idx = self.c2imap[node_name]
        return self.old_features[node_idx]

    def add_edges(self, node_id:str, raw_edges_dict:dict, edge_direction:str) :
        """ Adds the incoming/outgoing edges connected to a node
        Args :
            node_id : name of the node
            raw_edges_dict : edge dictionary from monolith class
            edge_direction : One of "in"/"out"        
        """

        # extract node type
        parts = node_id.split("_#_")
        nodetype = parts[0]
        # if we cant find a string with format <type>_#_<name>, default to program type
      
        if len(parts) == 1:
            nodetype = "program"
        else :
            assert(nodetype in ["resource"])

        #print("Source Node : {}".format(node_id))
        parsed_name = self.parse_real_name(node_id)


        # create the node
        node = self.create_node(parsed_name, node_type=nodetype, node_data=self.get_node_feature(node_id))
        #print("Source Type in Name : {} and Saved Source Type : {}".format(nodetype,self.node_dict[parsed_name].get_node_type()))
        assert(self.node_dict[parsed_name].get_node_type() == nodetype)

        # look at its neighbours and the connecting edges
        neig_nodes = raw_edges_dict[node_id][edge_direction]
        edge_types = raw_edges_dict[node_id][edge_direction+"_types"]
        #print("Nieghbour Nodes : {}".format(neig_nodes))

        for neig_id,edge_type_ele in zip(neig_nodes,edge_types) :

            #print("Target : {}".format(neig_id))

            neig_parsed_name = self.parse_real_name(neig_id)
            neig_parts = neig_id.split("_#_")
            neig_nodetype = neig_parts[0]
            # if we cant find a string with format <type>_#_<name>, default to program type

            # if ("db2" in neig_id) or ("vsam" in neig_id) :
            #     neig_nodetype = "resource"
            if len(neig_parts) == 1:
                neig_nodetype = "program"
            else : 
                assert(neig_nodetype in ["resource"])

            #print("Target Type : {}\n".format(neig_nodetype))
            # assert(False)

            if self.remove_node.get(neig_parsed_name,-1) == -1:
                # create the neighbour node
                neig_node = self.create_node(neig_parsed_name, node_type=neig_nodetype, node_data=self.get_node_feature(neig_id))
                #print("Target Type in Name : {} and Saved Target Type : {}".format(neig_nodetype,self.node_dict[neig_parsed_name].get_node_type()))
                assert(self.node_dict[neig_parsed_name].get_node_type() == neig_nodetype)

                if "CRUD_" in edge_type_ele :
                    # CRUD_R split to "CRUD" and "R"
                    edge_type, edge_action = edge_type_ele.split("_")
                else :
                    edge_type = "CALLS"
                    edge_action = None

                # create the edge
                self.create_edge(node_A=node,node_B=neig_node,edge_direction=edge_direction,edge_type=edge_type,edge_action=edge_action)

        #print("=================")

    def populate_edge_weight(self) :
        """Does nothing but initialize all weights to 1"""
        for edge in self.edges :
            edge.set_edge_weight(1.0)

    def parse_real_name(self, node_id) :
        """ Changes the resource name like resource_#_db2.customer-res to db2.customer-res.
            Leaves program names as is. """
        node_name = node_id
        parts = node_id.split("_#_")
        if len(parts) > 1:
            node_name = parts[1]
        return node_name

    def collect_nodes_and_edges(self) :
        """ Collects the nodes and edges from the Monolith class and stores it into the heterogenous graph. """

        raw_edges_dict = self.app.get_all_edges()
        self.node_dict = {}
        self.edge_dict = {}

        #print(raw_edges_dict)
        for node_id in raw_edges_dict.keys() :
            parsed_name = self.parse_real_name(node_id)
            if self.remove_node.get(parsed_name,-1) == -1 :
                # add the incoming edges
                self.add_edges(node_id, raw_edges_dict, "in")
                # add the outgoing edges
                self.add_edges(node_id, raw_edges_dict, "out")
            else :
                # skip the nodes that have been removed during pre-processing
                continue
        
        # populate the edge weight of each edge
        self.populate_edge_weight()

    def get_num_nodes(self) :
        return len(self.nodes)
    
    def get_num_edges(self) :
        return len(self.edges)

    def get_num_undirected_edges(self) :
        ans = 0
        for node in self.nodes :
            ans += len(node.in_edges)
        return ans
    
    def get_nodes(self) :
        """ Returns the nodes to the Graph Object """
        return self.nodes
    
    def get_edges(self) :
        """ Returns the edges to the Graph Object """
        return self.edges

    def get_seeds(self) :
        """ Returns the seeds to the Graph Object """
        return self.seed_dict