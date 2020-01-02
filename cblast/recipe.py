"""Recipes for clustering neurons.
"""

from . import features
from . import cluster
from . import utils

def cblast_workflow_simple(npclient, dataset, neuronlist, npc, premode="pro",
        cluster_neighbors=False, neighbor_npc=10, iterations=0, postprocess_pro=None, postprocess_conn=None,
        saved_projection=""):
    """Generate features through speculative iteration.

    The algorithm first generates features using neuron projections.  These
    results are then fed into an interative calling of postprocess_conn.  For
    more customization, call cblast_workflow.

    Args:
        npclient (object): neuprint client object
        dataset (str): name of neuprint dataset
        neuronlist (list): list of body ids
        premode (str): feature initialization mode ("pro", "conn", or "nonit" for projectome, connectivity,
        or non iterative features")
        npc (int): estimated number of neurons per cluster (set high?) for the list of neurons
        cluster_neighbors (boolean): use the upstream/downstream partners for initial  clustering with projection
        neighbor_npc (int): npc when considering all neighbors
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
        npc_round1 = npc
        if cluster_neighbors:
            npc_round1 = neighbor_npc
            new_neuronlist = utils.extract_neighbors(npclient, dataset, neuronlist) 
        num_neurons = len(new_neuronlist)

        if premode == "pro": 
            # generate projection features or body id connectivity features
            if postprocess_pro is not None:
                features_pro = features.extract_projection_features(npclient, dataset, new_neuronlist, postprocess=postprocess_pro)
            else:
                features_pro = features.extract_projection_features(npclient, dataset, new_neuronlist)
        elif premode == "conn":
            if postprocess_pro is not None:
                features_pro = features.compute_connection_similarity_features(npclient, dataset, new_neuronlist, use_saved_types=False, postprocess=postprocess_pro)
            else:
                features_pro = features.compute_connection_similarity_features(npclient, dataset, new_neuronlist, use_saved_types=False)
        else:
            features_pro = non_iterative_features(npclient, dataset, new_neuronlist)
        
        if saved_projection != "":
            features.save_features(features_pro, saved_projection)

    # create (aggressive?) clusters
    hcluster = cluster.hierarchical_cluster(features_pro)
    cluster2neuron, neuron2cluster = hcluster.get_partitions(num_neurons//npc_round1)
    print("Computed initial clusters")

    features_type = None
    lastdist = 9999999999
    last_features_type = None

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

        p1, p2, p3, p4, old_body2type = replay_data
        old_body2type.update(customtypes)
        replay_data = (p1,p2,p3,p4,old_body2type)

        # compute type features based on predicted clusters
        if postprocess_conn is not None:
            features_type = features.compute_connection_similarity_features(npclient, dataset, neuronlist, use_saved_types=False, postprocess=postprocess_conn, customtypes=customtypes, replay_data=replay_data)
        else:
            features_type = features.compute_connection_similarity_features(npclient, dataset, neuronlist, use_saved_types=False, customtypes=customtypes, replay_data=replay_data)

        # iteratively refine to specified limit or when no improvement found
        if num_iterations == iterations:
            break

        hcluster = cluster.hierarchical_cluster(features_type)
        cluster2neuron, neuron2cluster, dist, bpair  = hcluster.get_partitions(len(neuronlist)//npc, return_max=True)
        print(f"Computed type clusters: {dist}")

        # keep iterating until distance gets larger or doesn't change much if no iteration is specified
        if iterations < 1:
            diff = abs(dist - lastdist)
            # ignore when change is less than 1% and choose features closer to start
            if diff < (dist*0.01):
                if last_features_type is None:
                    raise RuntimeError("only ran one iteration") 
                features_type = last_features_type
                break
            lastdist = dist
            last_features_type = features_type

    return features_type

def cblast_workflow(npclient, dataset, neuronlist, npc,
        prev_features=None, iterations=0,
        use_saved_types=True, customtypes={}, postprocess=features.scaled_process(0.5, 0.5, [0.4,0.4,0.2]),
        sort_types=True, minconn=3, roi_restriction=None):
    """Wrapper for "features.compute_connections_similarity_features" that allow users to provide
    initial features (instead of customtypes) and number of iterations.

    Note: the cluster types produced by "prev_features" have higher priority than customtypes
    which have higher priority to the types stored in neuprint already if enabled. 

    Args:
        npclient (object): neuprint client object
        dataset (str): name of neuprint dataset
        neuronlist (list): list of body ids
        npc (int): estimated number of neurons per cluster (set high?)
        prev_features (dataframe): features to use for clustering if available
        iterations (int): number of iterations to iteratively refine connectivity reesults
        (default 0: automatically determine the number of iterations):
        use_saved_types(boolean): use types stored in neuprint already to guide clustering
        customtypes (dict or df): mapping of body ids to a cluster id (over-rules pre-exising types)
        postprocess (func): set with *_process function (each function allows users to set input and output to 0)
        (other paramters)
        sort_types (boolean): True to sort, False to collapse
        minconn (int): Only consider connections >= minconn
        roi_restriction (list): ROIs to restrict feature extraction (default: no restriction)
    Returns:
        dataframe: index: body ids; columns: different features 
    """
    
    if customtypes is not None and type(customtypes) == pd.DataFrame:
        customtypes =  dict(zip(customtypes["bodyid"], customtypes["type"]))

    # update custom types with provided features
    if prev_features is not None:
        hcluster = cluster.hierarchical_cluster(features_pro)
        cluster2neuron, neuron2cluster = hcluster.get_partitions(len(neuronlist)//npc)
        customtypes_new =  dict(zip(neuron2cluster["bodyid"], neuron2cluster["type"]))
        customtypes.update(customtypes_new)
   

    features_type = None
    lastdist = 9999999999

    # get and save all ccnnectivity features
    replay_data = None
    replay_data = features.compute_connection_similarity_features(npclient, dataset, neuronlist,
            use_saved_types=use_saved_types, custom_types=custom_types, postprocess=postprocess, 
            sort_types=sort_types, minconnn=minconn, roi_restriction=roi_restriction, dump_replay=True)

    # compute type features based on predicted clusters
    num_iterations = 0
    neuron2cluster = None
    while True:
        num_iterations += 1
        customtypes = {}
        if neuron2cluster is not None:
            customtypes =  dict(zip(neuron2cluster["bodyid"], neuron2cluster["type"]))

        # add new cell type groupings
        p1, p2, p3, p4, old_body2type = replay_data
        old_body2type.update(customtypes)
        replay_data_mod = (p1,p2,p3,p4,old_body2type)

        # compute type features based on predicted clusters
        replay_data = features.compute_connection_similarity_features(npclient, dataset, neuronlist,
                use_saved_types=use_saved_types, custom_types=custom_types, postprocess=postprocess, 
                sort_types=sort_types, minconnn=minconn, roi_restriction=roi_restriction, replay_data=replay_data_mod)
        # iteratively refine to specified limit or when no improvement found
        if num_iterations == iterations:
            break

        hcluster = cluster.hierarchical_cluster(features_type)
        cluster2neuron, neuron2cluster, dist, bpair  = hcluster.get_partitions(len(neuronlist)//npc, return_max=True)
        print(f"Computed type clusters: {dist}")

        # keep iterating until distance gets larger or doesn't change much
        diff = abs(dist - lastdist)
        if dist >= lastdist or diff < 0.0001:
            break
        lastdist = dist

    return features_type

def non_iterative_features(npclient, dataset, neuronlist, wgts=[0.4,0.4,0.2]):
    """Computes features based on connection distribution, roi projections, and roi overlap.

    Args:
        npclient (object): neuprint client object
        dataset (str): name of neuprint dataset
        neuronlist (list): list of body ids
        wgts (list): relative weight for connection distribution, projection, and roi overlap features respectively.
    
    Returns:
        dataframe: index: body ids; columns: different features 

    """

    dist_features = features.compute_connection_similarity_features(npclient, dataset, neuronlist, pattern_only=True)
    pro_features = features.extract_projection_features(npclient, dataset, neuronlist)
    overlap_features = features.extract_roioverlap_features(npclient, dataset, neuronlist)

    return features._combine_features([dist_features, pro_features, overlap_features], wgts)

