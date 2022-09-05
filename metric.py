import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import json
import math


def get_node_names(graph,node_ids) :
    node_ids = list(node_ids)
    names = { graph.nodes[id]["name"] for id in node_ids }
    return names

def read_result_json(result_file, graph) :
    """ Networkx module calculating modularity """

    with open(result_file,"r") as f :
        cluster_ans = json.load(f)

    # array of nodes
    nodes_arr = cluster_ans["nodes"]
    for node in nodes_arr :
        node_id = node["id"]
        node_name = node["label"]
        # add the node to the graph
        graph.add_node(node_id)
        graph.nodes[node_id]["name"] = node_name
    
    # array of edges
    edges_arr = cluster_ans["edges"][0]["relationship"]
    for edge in edges_arr :
        src = edge["id"].split("_")[0]
        tgt = edge["id"].split("_")[1]
        out_edge = [(src,tgt)]
        graph.add_edges_from(out_edge)

    # if a separate entry exists for program to resource relationship
    if len(cluster_ans["edges"]) > 1 :
        edges_arr = cluster_ans["edges"][1]["relationship"]
        for edge in edges_arr :
            src = edge["id"].split("_")[0]
            tgt = edge["id"].split("_")[1]
            out_edge = [(src,tgt)]
            graph.add_edges_from(out_edge)

    
    
    # array of clusters
    communities = []
    cluster_arr = cluster_ans["clusters"]
    for cluster in cluster_arr :
        nodes_set = set(cluster["nodes"])
        communities.append(nodes_set)

    

    return graph,communities

def get_modularity(result_file) :
    """Calculates the modularity of the microservices"""
    graph = nx.Graph()
    graph,communities = read_result_json(result_file, graph)
    modularity = nx_comm.modularity(graph,communities)
    
    return modularity

def get_ned(result_file, eps=0.5):
    """Calculates the NED score of the microservices"""
    total_length = 0

    with open(result_file,"r") as f :
        data = json.load(f)

    for i in data["nodes"]:
        if i["entity_type"] == 'class':
            total_length += 1
        else :
            #print("Entity type : {}".format(i["entity_type"]))
            total_length += 1
            # assert(False)

    num_clusters = len(data["clusters"])
    avg_cluster_size = total_length/num_clusters

    low_lim = math.floor(avg_cluster_size*(1-eps))
    high_lim = math.ceil(avg_cluster_size*(1+eps))
    
    

    valid_len = 0
    comm_data = data["clusters"]
    for i in comm_data:
        if len(i["nodes"]) >= low_lim and len(i["nodes"]) <= high_lim:
            valid_len += len(i["nodes"])

    total_score = valid_len / total_length
    
    return total_score

def get_IFN(result_file) :
    """Calculates the average interface number of the microservices"""
    graph = nx.Graph()
    graph,communities = read_result_json(result_file, graph)

    ans = 0
    for comm_set in communities :
        comm = list(comm_set)
        
        comm_names = get_node_names(graph,comm_set)
        

        comm_ans = 0
        for node in comm :
            neig = graph[node]
            neig_set = set(neig)
            
            
            outside_node_set = neig_set - comm_set
            
            outside_names = get_node_names(graph,outside_node_set)
            
            comm_ans += len(outside_node_set)
            
        
        ans += comm_ans
        
    
    ans /= len(communities)
    ans /= 2
    
    return ans


def get_coverage(result_file) :
    graph = nx.Graph()
    graph,communities = read_result_json(result_file,graph)
    coverage = nx_comm.coverage(graph,communities)
    
    return coverage

def get_structural_coupling(result_file) :
    
    graph = nx.Graph()
    graph,communities = read_result_json(result_file, graph)
    total_ans = 0
    num_clusters = len(communities)

    idx_arr = [ i for i in range(len(communities))]
    for i in idx_arr :
        for j in idx_arr :
            if i >= j :
                continue
            else :
                cross_edges = 0
                # cluster i
                cluster_i_set = communities[i]
                cluster_i_list = list(cluster_i_set)
                num_i = len(cluster_i_set)
                
                # cluster j
                cluster_j_set = communities[j]
                
                num_j = len(cluster_j_set)
                # find coupling
                for ele_i in cluster_i_list :
                    neig = graph[ele_i]
                    
                    interection = set(neig) & cluster_j_set
                    
                    # finding the sigma_ij
                    cross_edges += len(interection)

                # final formula : sigma_ij/(2 * N_i * N_j)
                cross_edges = cross_edges/(2*num_i*num_j)

            total_ans += cross_edges

    denom = (num_clusters*(num_clusters-1))/2
    total_ans = total_ans/denom 

    return total_ans

def get_structural_cohesivity(result_file) :

    graph = nx.Graph()
    graph,communities = read_result_json(result_file, graph)
    num_clusters = len(communities)
    total_ans = 0
    for comm_set in communities :
        comm_list = list(comm_set)
        
        community_graph = graph.subgraph(comm_list)
        # node count of community
        comm_nodes_cnt = community_graph.number_of_nodes()
        # edge count of community
        comm_edges_cnt = community_graph.number_of_edges()*2
        # final value = (u_i/N_i**2)
        comm_ans = comm_edges_cnt/(comm_nodes_cnt**2)
        # community edges
        comm_edges = community_graph.edges()
        
        total_ans += comm_ans
    # averaging
    total_ans /= num_clusters

    return total_ans

def get_structural_modularity(result_file) : 
    cohesivity = get_structural_cohesivity(result_file)
    coupling = get_structural_coupling(result_file)
    SM =  cohesivity - coupling
    
    return SM

    


