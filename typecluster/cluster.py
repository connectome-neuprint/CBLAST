"""Neuron type clustering.

Contains a set of functions for clustering neurons and visualing results.
"""
    
import json
import numpy as np
import pandas as pd


def compute_distance_matrix(features):
    """Compute a distance matrix between the neurons.

    Args:
        features (dataframe): matrix of features
    Returns:
        (dataframe): A 2d distance matrix represented as a table
    """

    # compute pairwise distance and put in square form
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    dist_matrix = squareform(pdist(features.values))

    return pd.DataFrame(dist_matrix, index=features.index.values.tolist(), columns=features.columns.values.tolist())

class HCluster:
    """Simple wrapper class for cluster output to preserve labeling.
    """
    def __init__(self, cluster, labels):
        self.cluster = cluster
        self.labels = labels

    def compute_vi(self, gtmap, num_parts=0):
        """Return variation of information based on provided ground truth maps.

        Args:
            gtmap (dict): body/neuronid => label or type
            num_parts (int): optional specification of number of parititons to use (default=0, find optimal)
        Returns:
            (float64, float64, dataframe, numparts): merge vi, split vi, bodyids and score, and num partitionse
        """

        # find optimal match unless one is specified
        from scipy.cluster.hierarchy import cut_tree
       
        partitions = None
        if num_parts > 0:
            partitions = cut_tree(self.cluster, n_clusters=num_parts)
            return _vi_wrapper(list(partitions[:.0]), self.labels, gtmap) 

        # iterate through all partitionings and find best match
        bestmatch = 99999999999999
        partitions = cut_tree(self.cluster)
        bestres = None
        for colid in range(0, len(partitions[0,:])):
            merge, split, bodyrank = _vi_wrapper(list(partitions[:,colid]), self.labels, gtmap) 
            if (merge + split) < bestmatch:
                bestmatch = (merge+split)
                bestres = (merge, split, bodyrank, len(gtmap)-colid)

        return bestres

class KCluster:
    """Simple wrapper class for cluster output to preserve labeling.
    """
    def __init__(self, cluster, labels):
        self.cluster = cluster
        self.labels = labels

    def compute_vi(self, gtmap):
        """Return variation of information based on provided ground truth maps.

        Args:
            gtmap (dict): body/neuronid => label or type
        Returns:
            (float64, float64, dataframe): merge vi, split vi, bodyids and score
        """


        return _vi_wrapper(self.cluster.labels_, self.labels, gtmap) 


def _vi_wrapper(clabels, indices, gtmap):
    # determine mapping of index to type
    cid = 1
    typemap = {}
    cidmap = {}
    gtlist = []
    for bodyid in indices:
        currclass = gtmap[bodyid]
        if currclass in typemap:
            gtlist.append(typemap[currclass])
        else:
            gtlist.append(cid)
            typemap[currclass] = cid
            cidmap[cid] = currclass
            cid += 1

   
    (merge_vi, split_vi, bodydata) = _compute_vi(clabels, gtlist)

    # relabel the index to the provided cluster names
    newindices = {}
    cidlist = bodydata.index.values.tolist()
    for cid in cidlist:
        newindices[cid] = cidmap[cid]

    bodydata.rename(newindices, inplace = True)

    return merge_vi, split_vi, bodydata


# ?! TODO print number of mistakes -- edit distance
def _compute_vi(labels, gt):
    """Compute variation of information for set of labels compared to groundtruth.

    Args:
        labels (list): integer array providing cluster ids for each index location
        gt (list): array of ground truth labels for each index location
    Returns:
        (float64, float64, (list,list)): false split, false merge VI and a list VI norm, VI unnorm
    """
    from math import log

    # vi for each gt label
    gtvi = {}

    # construct groups
    label_groups = {}
    gt_groups = {}

    for idx, label in enumerate(labels):
        if label not in label_groups:
            label_groups[label] = {}
        if gt[idx] not in label_groups[label]:
            label_groups[label][gt[idx]] = 0
        label_groups[label][gt[idx]] += 1

    for idx, label in enumerate(gt):
        if label not in gt_groups:
            gt_groups[label] = {}
        if labels[idx] not in gt_groups[label]:
            gt_groups[label][labels[idx]] = 0
        gt_groups[label][labels[idx]] += 1

    # false merge VI
    merge_vi = 0
    gtvi = {}

    for label, group in label_groups.items():
        tot = 0
        for label2, count in group.items():
            tot += count
        for label2, count in group.items():
            info = count/len(labels)*(log(tot/count)/log(2))
            merge_vi += info
            if label2 not in gtvi:
                gtvi[label2] = 0
            gtvi[label2] += info

    # false split VI
    split_vi = 0
    for label, group in gt_groups.items():
        tot = 0
        for label2, count in group.items():
            tot += count
        for label2, count in group.items():
            info = count/len(labels)*(log(tot/count)/log(2))
            split_vi += info
            if label not in gtvi:
                gtvi[label] = 0
            gtvi[label] += info

    # normalize gtvi
    
    # get size of each cluster
    gtsize = {}
    for label in gt:
        if label not in gtsize:
            gtsize[label] = 0
        gtsize[label] += 1

    vitable = np.zeros((len(gtvi), 2))
    iter1 = 0
    indices = []
    for label, vi in gtvi.items():
        normvi = vi / gtsize[label]
        vitable[iter1] = [vi, normvi]
        indices.append(label)
        iter1 += 1
    final_table = pd.DataFrame(vitable, index=indices, columns=["VI", "VI-norm"])

    return merge_vi, split_vi, final_table





def hierarchical_cluster(features):
    """Compute a hierarchical clustering for a given set of features.
    
    Args:
        features (dataframe): matrix of features
    Returns:
        (HClusterobject): wraps cluster resultss
    """
    from scipy.cluster.hierarchy import linkage
    
    # compute distance function
    #distm = compute_distance_matrix(features)
    
    clustering = linkage(features.values, 'ward')
    
    return HCluster(clustering, features.index.values.tolist())


def kmeans_cluster(features, num_clusters):
    """Run kmeans clustering.

    Args:
        features (dataframe): matrix of features
    Returns:
        (KMeans): kmeans object result
    """
    from sklearn.cluster import KMeans

    return KCluster(KMeans(n_clusters=num_clusters, random_state=0).fit(features), features.index.values.tolist())


def sort_distance_matrix(distance):
    """Sort the distance matrix based on hierarchical clustering.

    Ars:
        distance (dataframe): NxN dataframe
    Returns:
        (dataframe): distance matrix with sorted indices
    """

    # ?! needs condensed matrix !! 
    clustering = linkage(distance.values, 'ward')
    sorted_indices = _sort_indices(clustering, len(clustering), len(clustering)*2-2)
    bodyids = distance.columns.values.tolist()
    neworder = []
    for idx in sorted_indices:
        neworder.append(bodyids[idx])
    return distance.reindex(index=neworder, columns=neworder)

def find_closest_neurons(neuron, features):
    """Find closest neurons to the given neuron.

    Args:
        neuron (int): neuron id
        features (dataframe): array contain featurs describing the neuron and related neurons.
    Returns:
        (series): data series for body ids and distance based on features
    """
    dist = compute_distance_matrix(features)
    closest_neurons = dist[neuron]
    return dist[neuron].sort_values(ascending=True)


""" -- example for showing matrix
plt.pcolormesh(dist_mat)
plt.xlim([0,N])
plt.ylim([0,N])
plt.show()
"""

def _sort_indices(Z, N, cur_index):
    """Extracts ordered indices from dendrogram.
    
    From: https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))






