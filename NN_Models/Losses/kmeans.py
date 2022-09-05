import sys
import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from copy import deepcopy

class Clustering(object):

    def __init__(self, seeds=None, K=None, maxiter=100, num_classes=None):
        self.seeds = seeds
        self.K = K
        self.max_iter = maxiter

        if(seeds is None and K is None):
            #print("Assertion ERROR: both the Seeds and K are None.")
            #print("Using a default K, but it is advised to check the inputs.")
            if num_classes is not None:
                k = int(np.sqrt(num_classes))
                if k > 50:
                    k = 50
                if k < 3:
                    k = 3
                self.K = k
            else:
                self.K = 5

        self.u = None
        self.M = None

    def cluster(self, embed, isgpu=True):
        
        embed_np = embed.detach().cpu().numpy()

        # If seeds are available, call SeededKmeans else use default Kmeans to cluster
        if self.seeds is not None:
            
            clustering = SeededKmeans(n_iter=self.max_iter)
            clustering.fit(embed_np, self.seeds)
        else:
            clustering = KMeans(n_clusters=self.K, n_init=5, max_iter=self.max_iter)
            clustering.fit(embed_np)

        self.M = clustering.labels_
        self.u = self._compute_centers(self.M, embed_np)

    def get_loss(self, embed):
        loss = torch.Tensor([0.])
        #TODO: This may be slightly inefficient, we can fix it later to use matrix multiplications
        for i, clusteridx in enumerate(self.M):
            x = embed[i]
            c = self.u[clusteridx]
            difference = x - c
            err = torch.sum(torch.mul(difference, difference))
            loss += err

        return loss

    def get_seed_loss(self, embed):
        # if we end up calling seed loss when seeds are not present, return 0.0
        if self.seeds is None:
            return torch.Tensor([0.0])
        centers = []
        
        
        

        for seed_arr in self.seeds:
            centers.append(torch.mean(embed[seed_arr], dim=0))

        centers = torch.stack(centers)
        
        # loss = torch.mean(torch.cdist(centers, centers))
        # 
        # #return torch.sqrt(loss)
        # return loss

        return torch.mean(_pairwise_distances(centers, centers))

    def get_membership(self):
        return self.M

    def _compute_centers(self,labels, embed):
        """
        sklearn kmeans may not give accurate cluster centers in some cases (see doc), so we compute ourselves
        """
        clusters = {}
        for i,lbl in enumerate(labels):
            if clusters.get(lbl) is None:
                clusters[lbl] = []
            clusters[lbl].append(torch.FloatTensor(embed[i]))

        centers = {}
        for k in clusters:
            all_embed = torch.stack(clusters[k])
            center = torch.mean(all_embed, 0)
            centers[k] = center

        return centers

class SeededKmeans():
    """ Kmeans with seeds provided and cluster-membership fixed """

    def __init__(self, n_iter=100, tol=1e-6):
        self.n_iter = n_iter
        self.labels_ = []
        self.cluster_centers_ = []
        self.tol = tol

    def fit(self, X, seed_idxs):
        """ remember: seed_idxs is an array of arrays (there could be multiple seeds per row)"""
        X = np.array(X)

        all_seeds = []
        for seed_arr in seed_idxs:
            all_seeds.extend(seed_arr)

        self.cluster_centers_ = self._compute_initial_centers(X, seed_idxs)

        points = range(X.shape[0])

        for iter in range(self.n_iter):

            self.labels_ = [0] * X.shape[0] # assign all points to cluster 0 by default
            # compute distances to centers
            distances = euclidean_distances(self.cluster_centers_, X)

            # already assign seeds to the fixed clusters
            clusters = [deepcopy(idxs) for idxs in seed_idxs]
            # seeds 0 is assigned to cluster 0, seeds 1 to cluster 1 and so on
            for i,seeds in enumerate(seed_idxs):
                for s in seeds:
                    self.labels_[s] = i

            # assign the rest of the points to the nearest center, skipping seeds
            for j in range(distances.shape[1]):
                if j in all_seeds:
                    continue
                d = distances[:,j]
                closest_cluster = np.argmin(d)
                clusters[closest_cluster].append(j)
                self.labels_[j] = closest_cluster

            # recompute centers
            prev_centers = self.cluster_centers_
            self.cluster_centers_ = [np.mean(np.array(X[c]), axis=0) for c in clusters]

            # check termination condition
            diffs = euclidean_distances(prev_centers, self.cluster_centers_)
            diffs = [diffs[i,i] for i in range(diffs.shape[0])]
            maxdiff = np.max(diffs)
            if maxdiff < self.tol or iter == self.n_iter:
                break

        

    def _compute_initial_centers(self, X, seed_idxs):
        centers = []
        for seeds in seed_idxs:
            subarr = X[seeds]
            centers.append(np.mean(subarr, axis=0))

        return np.array(centers)

def _pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    FROM: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)
