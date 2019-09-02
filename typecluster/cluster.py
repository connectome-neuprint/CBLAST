"""Neuron type clustering.

Contains a set of functions for clustering neurons and visualing results.
"""
    
import json
import numpy as np
import pandas as pd



# ?! mod hierarchical cluster partition result to also return the max dist, allow restriction based on distance (when comparing two different features check distance for similar partition and find diffs and see how different it clusters when distance constrained -- if the distances are believed to be equivalent-ish





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

    return pd.DataFrame(dist_matrix, index=features.index.values.tolist(), columns=features.index.values.tolist())


def report_diffs(self, part1, part2, features1 = None, features2 = None)
    """Return the bodies that are in different clusters. Ordered by distance if features provided.
    """
    
    # get part1 fragments
    # get type => [bodyid...]
    type2bodies = {}
    for bodyid, cluster in part1.items():
        if cluster not in type2bodies:
            type2bodies[cluster] = []
        type2bodies[cluster].append(bodyid)


    falsepart1 = []
    for cluster, bodyids in type2bodies.items():
        for iter1 in range(len(bodyids)):
            for iter2 in  range(iter1+1, len(bodyids)):
                if part2[bodyids[iter1]] != part2[bodyids[iter2]]:
                    if features2 is not None:
                        v1 = features2.loc[bodyids[iter1]].values
                        v2 = features2.loc[bodyids[iter2]].values
                        diffvec = (v1-v2)**2
                        diff = (diffvec.sum())**(1/2)
                    falsepart1.append((diff, bodyids[iter1], bodyids[iter2]))

    # get part2 fragments
    type2bodies = {}
    for bodyid, cluster in part2.items():
        if cluster not in type2bodies:
            type2bodies[cluster] = []
        type2bodies[cluster].append(bodyid)

    falsepart2 = []
    for cluster, bodyids in type2bodies.items():
        for iter1 in range(len(bodyids)):
            for iter2 in  range(iter1+1, len(bodyids)):
                if part1[bodyids[iter1]] != part1[bodyids[iter2]]:
                    if features1 is not None:
                        v1 = features1.loc[bodyids[iter1]].values
                        v2 = features1.loc[bodyids[iter2]].values
                        diffvec = (v1-v2)**2
                        diff = (diffvec.sum())**(1/2)
                    falsepart2.append((diff, bodyids[iter1], bodyids[iter2]))
                  
    falsepart1.sort()
    falsepart2.sort()

    return falsepart1, falsepart2

    
class HCluster:
    """Simple wrapper class for cluster output to preserve labeling.
    """
    def __init__(self, cluster, labels, features):
        self.cluster = cluster
        self.labels = labels
        self.features = features

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

    def _get_max_dist(self, res):
        """Gets max distance between features in the same cluster.

        Args:
            res (dict): cluster id -> [body1, body2, ... ]
        Returns:
            (float, tuple): maximum distance between bodies in a cluster
        """
        max_dist = 0
        max_pair = None
        for idx, group in res.items():
            for iter1 in range(len(group)):
                for iter2 in range(iter1+1, len(group)):
                    v1 = self.features.loc[group[iter1]].values
                    v2 = self.features.loc[group[iter2]].values
                    diffvec = (v1-v2)**2
                    diff = (diffvec.sum())**(1/2)
                    if diff > max_dist:
                        max_dist = diff
                        max_pair = (group[iter1], group[iter2])
        return max_dist, max_pair

    def get_partitions_dist_constraint(self, dist):
        """Returns cluster partitions for the specified distance constraint.

        Args:
            dist (float): maximum distance to allow between bodies in a cluter 
        Returns:
            (dict, dataframe, float, tuple): {cluster id: [body1, body2,...]}, "bodyid", "cluster id",
            distance between farthest bodies, farthest bodies in the same cluster
        """
        from scipy.cluster.hierarchy import cut_tree

        previous_result = None
        partitions = cut_tree(self.cluster)
        for colid in range(0, len(partitions[0,:])):
            labels = list(partitions[:,colid])
            mapping = pd.DataFrame(list(zip(self.labels, labels)), columns=["bodyid", "type"])
            res = {}

            for idx, label in enumerate(labels):
                if label not in res:
                    res[label] = []
                res[label].append(self.labels[idx])
        
            max_dist, max_pair = self._get_max_dist(res)
            if max_dist > dist:
                break
            previous_result = (res, mapping, max_dist, max_pair) 

        return previous_result


    def get_partitions(self, num_parts, return_max=False):
        """Returns cluster partitions for specified number of clusters.

        Args:
            num_parts (int): number of cluster partitions
            return_max (boolean): if true return the maximum distance between body ids in one cluster
        Returns:
            (dict, dataframe): {cluster id: [body1, body2,...]}, "bodyid", "cluster id"
            optional (dict, dataframe, float, tuple): includes maximum distance between bodies in a cluster
            and those body ids
        """
        from scipy.cluster.hierarchy import cut_tree
        partitions = cut_tree(self.cluster, n_clusters=num_parts)
        res = {}

        labels = list(partitions[:,0])
        mapping = pd.DataFrame(list(zip(self.labels, labels)), columns=["bodyid", "type"])
        for idx, label in enumerate(labels):
            if label not in res:
                res[label] = []
            res[label].append(self.labels[idx])
        
        if return_max:
            max_dist, max_pair = self._get_max_dist(res)
            return (res, mapping, max_dist, max_pair)
        return (res, mapping)


class KCluster:
    """Simple wrapper class for cluster output to preserve labeling.
    """
    def __init__(self, cluster, labels, features):
        self.cluster = cluster
        self.labels = labels
        self.features = features

    def compute_vi(self, gtmap):
        """Return variation of information based on provided ground truth maps.

        Args:
            gtmap (dict): body/neuronid => label or type
        Returns:
            (float64, float64, dataframe): merge vi, split vi, bodyids and score
        """

        return _vi_wrapper(self.cluster.labels_, self.labels, gtmap) 

    def get_partitions(self):
        """Returns cluster partitions where the first element is the closest to center.

        Returns:
            (dict, dataframe): {cluster id: [body1, body2,...]}, "bodyid", "cluster id"
        """
        res = {}
        
        for idx, label in enumerate(self.cluster.labels_):
            if label not in res:
                res[label] = []
            res[label].append(idx)
        
        mapping = pd.DataFrame(list(zip(self.labels, self.cluster.labels_)), columns=["bodyid", "type"])

        finalres = {}
        # find representative element for each cluster
        for idx in range(0, len(self.cluster.cluster_centers_)):
            cf = self.cluster.cluster_centers_[idx] 
            mindist = 999999999999999999999
            minidx = -1
            for idx2 in res[idx]:
                dist = np.linalg.norm(self.features.iloc[idx2].values-cf)
                if dist < mindist:
                    mindist = dist
                    minidx = idx2 
            
            finalres[idx] = [self.labels[minidx]]
            for idx2 in res[idx]:
                if idx2 != minidx:
                    finalres[idx].append(self.labels[idx2])

        return (finalres, mapping)

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
    splitvi = {}

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
            if label not in splitvi:
                splitvi[label] = 0
            gtvi[label] += info
            splitvi[label] += info

    # normalize gtvi
    
    # get size of each cluster
    gtsize = {}
    for label in gt:
        if label not in gtsize:
            gtsize[label] = 0
        gtsize[label] += 1

    vitable = np.zeros((len(gtvi), 4))
    iter1 = 0
    indices = []
    for label, vi in gtvi.items():
        normvi = vi / gtsize[label]
        split = 0
        if label in splitvi:
            split = splitvi[label]
        vitable[iter1] = [vi, normvi, vi-split, split]
        indices.append(label)
        iter1 += 1
    final_table = pd.DataFrame(vitable, index=indices, columns=["VI", "VI-norm", "false merge", "false split"])

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
    # distm = compute_distance_matrix(features)
    
    clustering = linkage(features.values, 'ward')
    
    return HCluster(clustering, features.index.values.tolist(), features)


def kmeans_cluster(features, num_clusters):
    """Run kmeans clustering.

    Args:
        features (dataframe): matrix of features
    Returns:
        (KMeans): kmeans object result
    """
    from sklearn.cluster import KMeans

    return KCluster(KMeans(n_clusters=num_clusters, random_state=0).fit(features), features.index.values.tolist(), features)


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






