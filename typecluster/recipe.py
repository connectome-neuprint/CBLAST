"""Recipes for clustering neurons.
"""

from . import features
from . import cluster

def iterative_features(neuronlist, dataset, npclient, est_neuron_per_cluster, projection_init=True, iterative=True, collapse_types=False, saved_projection=""):
    """Generate features through speculative iteration.

    Args:
        neuronlist (list): list of body ids
        dataset (str): name of neuprint dataset
        npclient (object): neuprint client object
        est_neuron_per_cluster (int): estimated number of neurons per cluster (set high?)
        projection_init (boolean): use projection features to seed initial clustering
        iterative (boolean): iteratively refine features based on connectivity cluster results
        collapse_types (boolean): true means to collapse types, false means to sort
        saved_projection (str): location of file to store or restore features
   
    Return:
       (dataframe): Features for neuron list

    Note:
        Evaluating results could be done by looking at the worst distance for different levels of clustering.
        In addition, features computed with different recipes can be compared using 'report_diffs'.
        It probably make sense to produce final cluster results at different levels (conservative,
        mediumn, aggressive, etc).
    
    """
    # TODO potentially use a distance threshold for initial clustering (or connectome clustering)

    assert not collapse_types, "Does not support cell type collapsing"

    neuron2cluster = None
    if projection_init:

        features_pro = None
        if saved_projection != "":
            try:
                features_pro = features.load_features(saved_projection)
            except:
                pass

        # if not features are read from disk, compute features
        if features_pro is None:
            # generate projection features or body id connectivity features
            roiquery = f"MATCH (m :Meta) WHERE m.dataset=\"{dataset}\" RETURN m.superLevelRois AS rois"                    
            roires = npclient.fetch_custom(roiquery)                                                                       
            superrois = set(roires["rois"].iloc[0])

            features_pro = features.extract_projection_features(neuronlist, dataset, npclient, roilist=list(superrois), leftright=True)

            if saved_projection != "":
                features.save_features(features_pro, saved_projection)

        # create (aggressive?) clusters
        hcluster = cluster.hierarchical_cluster(features_pro)
        cluster2neuron, neuron2cluster = hcluster.get_partitions(len(neuronlist)//est_neuron_per_cluster)
        print("Computed projection clusters")

    features_type = None
    lastdist = 9999999999

    # get and save all ccnnectivity features
    replay_data = features.compute_connection_similarity_features(neuronlist, dataset, npclient, sort_types=(not collapse_types), dump_replay=True)

    # compute type features based on predicted clusters
    while True:
        # TODO: potentially set the cluster name to be a representative element
        # from the cluster
        customtypes = {}
        if neuron2cluster is not None:
            customtypes =  dict(zip(neuron2cluster["bodyid"], neuron2cluster["type"]))
        
        # add new cell type groupings
        p1, p2, p3, p4, old_body2type = replay_data
        old_body2type.update(customtypes)
        replay_data_mod = (p1,p2,p3,p4,old_body2type)

        # compute type features based on predicted clusters
        features_type = features.compute_connection_similarity_features(neuronlist, dataset, npclient, customtypes=customtypes,  sort_types=(not collapse_types), replay_data=replay_data_mod)
        # iteratively refine
        if not iterative:
            break

        hcluster = cluster.hierarchical_cluster(features_type)
        cluster2neuron, neuron2cluster, dist, bpair  = hcluster.get_partitions(len(neuronlist)//est_neuron_per_cluster, return_max=True)
        print(f"Computed type clusters: {dist}")

        # keep iterating until distance gets larger or doesn't change much
        diff = abs(dist - lastdist)
        if dist >= lastdist or diff < 0.0001:
            break
        lastdist = dist

    return features_type

    
