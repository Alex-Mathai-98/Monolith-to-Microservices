import sys
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
from typing import List
from sklearn.cluster import KMeans
import numpy as np

def compute_edge_loss(features : List[torch.tensor], 
                    recon : List[torch.tensor]) :

    num_nodes = len(features)
    loss = torch.zeros((num_nodes))

    index = 0
    for feature_ele,recon_ele in zip(features,recon) :
        prediction = F.log_softmax(recon_ele,dim=0)
        losses = -1*torch.sum(prediction*feature_ele)
        loss[index] = losses
        index += 1

    
    #print("Edge Loss Size : {}".format(loss.size()))
    # assert(False)
    return torch.sum(loss)

def compute_attribute_loss(features : List[torch.tensor], 
                    recon : List[torch.tensor], 
                    outlier_wt:torch.tensor=None,
                    mode:str=None):

    num_nodes = len(features)
    loss = torch.zeros((num_nodes))

    if mode == "Node" : 
        index = 0
        for feature_ele,recon_ele in zip(features,recon) :
            loss[index] = torch.sum((feature_ele-recon_ele)**2)
            index += 1
    elif mode == "Edge" :
        index = 0
        for feature_ele,recon_ele in zip(features,recon) :
            #loss[index] = torch.sum(-1*(feature_ele*F.log_softmax(recon_ele)))
            loss[index] = torch.sum((feature_ele-torch.sigmoid(recon_ele))**2)
            # old code below
            #loss[index] = torch.sum((feature_ele-recon_ele)**2)
            index += 1
    else :
        assert(False)

    if outlier_wt is None :
        outlier_wt = [1.0/num_nodes]*num_nodes
        outlier_wt = torch.tensor(outlier_wt)

    outlier_wt = torch.log(1/outlier_wt)

    #print("{} Loss Size : {}".format(mode,loss.size()))

    # we dont need to multiply with outlier scores during pretraining
    # but since the scores are fixed, it doesnt change the result and keeps the loss
    # in the same relative scale compared to when we introduce outlier scores
    attr_loss = torch.sum(torch.mul(outlier_wt, loss))

    return attr_loss

def compute_structure_loss(adj : torch.tensor, 
                        embed : torch.tensor, 
                        outlier_wt:torch.tensor=None):

    # to compute F(x_i).F(x_j)
    embeddot = torch.mm(embed, torch.transpose(embed, 0, 1))

    # ground_truth = ground_truth.to_dense()
    # positive_entries = torch.gt(adj_tensor, 0.0)

    # # to identify i,j embeddot which must be used in summation
    # embeddot = torch.mul(embeddot, positive_entries)

    # compute A_ij - F(x_i)*F(x_j)
    difference = adj - embeddot
    # square difference and sum
    loss = torch.mean(torch.mul(difference, difference), dim=1)

    if outlier_wt is None :
        num_nodes = len(adj)
        outlier_wt = [1.0/num_nodes]*num_nodes
        outlier_wt = torch.tensor(outlier_wt)

    outlier_wt = torch.log(1/outlier_wt)

    # we dont need to multiply with outlier scores during pretraining
    # but since the scores are fixed, it doesnt change the result and keeps the loss
    # in the same relative scale compared to when we introduce outlier scores
    #print("Structure Loss Size : {}".format(loss.size()))
    #print("\t==>Adj Size : {}".format(adj.size()))
    #print("\t==>Ground Truth Size : {}".format(adj.size()))
    struct_loss = torch.sum(torch.mul(outlier_wt, loss))
    return struct_loss

def update_o1(adj_matrix, embed):
    # to compute F(x_i).F(x_j)
    embed = embed.data
    embeddot = torch.mm(embed, torch.transpose(embed, 0, 1))

    # adj_tensor = adj.to_dense()
    # positive_entries = torch.gt(adj_tensor, 0.0)

    # # to identify i,j embeddot which must be used in summation
    # embeddot = torch.mul(embeddot, positive_entries)

     # compute A_ij - F(x_i)*F(x_j)
    difference = adj_matrix - embeddot
    # square difference and sum
    error = torch.sum(torch.mul(difference, difference), dim=1)

    # compute the denominator
    normalization_factor = torch.sum(error)

    # normalize the errors
    o1 = error/normalization_factor
    
    return o1

def update_o2(features : List[torch.tensor], recon : List[torch.tensor]):
    
    """
    features = features.data
    recon = recon.data
    # error = x - F(G(x))
    error = features - recon
    # error now = (x - F(G(x)))^2, summed across dim 1
    error = torch.sum(torch.mul(error, error), dim=1)
    """

    num_nodes = len(features)
    error = torch.zeros((num_nodes))
    index = 0
    for feature_ele,recon_ele in zip(features,recon) :
        error[index] = torch.sum((feature_ele-recon_ele)**2)
        index += 1

    # compute the denominator
    normalization_factor = torch.sum(error)

    # normalize the errors
    o2 = error/normalization_factor
    
    return o2