"""Recipes for clustering neurons.
"""

from . import features
from . import cluster
from . import utils

def cblast_workflow_simple(npclient, dataset, neuronlist, est_neuron_per_cluster,
        cluster_neighbors=False, iterations=0, postprocess_pro=None, postprocess_conn=None,
        saved_projection=""):
    """Generate features through speculative iteration.

    The algorithm first generates features using neuron projections.  These
    results are then fed into an interative calling of postprocess_conn.  For
    more customization, call cblast_workflow.

    Args:
        npclient (object): neuprint client object
        dataset (str): name of neuprint dataset
        neuronlist (list): list of body ids
        est_neuron_per_cluster (int): estimated number of neurons per cluster (set high?)
        cluster_neighbors (boolean): use the upstream/downstream partners for initial  clustering with projection
        iterations (int): number of iterations to iteratively refine connectivity reesults
        (default 0: automatically determine the number of iterations):
        postprocess_pro (func): *_process function option for projection features
        postprocess_conn (func): *_process function option for connectivity features
        
        saved_projection (str): location of file to store or restore features
   
    Return:
       (dataframe): Features for neuron list

    Note:
        Evaluating results could be done by looking at the worst distance for different levels of clustering.
        In addition, features computed with different recipes can be compared using 'report_diffs'.
        It probably make sense to produce final cluster results at different levels (conservative,
        mediumn, aggressive, etc).
    
    """
    neuron2cluster = None

    features_pro = None
    if saved_projection != "":
        try:
            features_pro = features.load_features(saved_projection)
        except:
            pass

    num_neurons = len(neuronlist)
    # if not features are read from disk, compute features
    if features_pro is None:
        new_neuronlist = neuronlist
        if cluster_neighbors:
            new_neuronlist = utils.extract_neighbors(npclient, dataset, neuronlist) 
        num_neurons = len(new_neuronlist)
        # generate projection features or body id connectivity features
        if postprocess_pro is not None:
            features_pro = features.extract_projection_features(npclient, dataset, new_neuronlist, postprocess=postprocess_pro)
        else:
            features_pro = features.extract_projection_features(npclient, dataset, new_neuronlist)

        if saved_projection != "":
            features.save_features(features_pro, saved_projection)

    # create (aggressive?) clusters
    hcluster = cluster.hierarchical_cluster(features_pro)
    cluster2neuron, neuron2cluster = hcluster.get_partitions(num_neurons//est_neuron_per_cluster)
    print("Computed projection clusters")

    features_type = None
    lastdist = 9999999999

    # get and save all ccnnectivity features
    replay_data = None
    if postprocess_conn is not None:
        replay_data = features.compute_connection_similarity_features(npclient, dataset, neuronlist, use_saved_types=False, postprocess=postprocess_conn, dump_replay=True)
    else:
        replay_data = features.compute_connection_similarity_features(npclient, dataset, neuronlist, use_saved_types=False, dump_replay=True)

    # compute type features based on predicted clusters
    num_iterations = 0
    while True:
        num_iterations += 1
        customtypes = {}
        if neuron2cluster is not None:
            customtypes =  dict(zip(neuron2cluster["bodyid"], neuron2cluster["type"]))

        # ?! if I support fanin/fanout for pro features, need to keep those types around
        # add new cell type groupings
        p1, p2, p3, p4, old_body2type = replay_data
        old_body2type.update(customtypes)
        replay_data_mod = (p1,p2,p3,p4,old_body2type)

        # compute type features based on predicted clusters
        if postprocess_conn is not None:
            features_type = features.compute_connection_similarity_features(npclient, dataset, neuronlist, use_saved_types=False, postprocess=postprocess_conn, customtypes=customtypes, replay_data=replay_data_mod)
        else:
            features_type = features.compute_connection_similarity_features(npclient, dataset, neuronlist, use_saved_types=False, customtypes=customtypes, replay_data=replay_data_mod)

        # iteratively refine to specified limit or when no improvement found
        if num_iterations == iterations:
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

    
