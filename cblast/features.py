"""Neuron feature generation.

Contains a set of functions for extracting features for a list of neurons.
"""

import json
import numpy as np
import pandas as pd

def save_features(features, filename):
    """Save features to disk.
    
    Args:
        features (dataframe): features for a set of bodies (body ids should be the index)
        filename (str): name of file
    """

    tmpfeatures = features.reset_index()
    tmpfeatures.to_feather(filename)

def load_features(filename):
    """Load features from disk.
    
    Args:
        filename (str): name of file
    Returns:
        dataframe: features for a set of bodies
    """

    features =  pd.read_feather(filename, columns=None, use_threads=True)
    features.set_index('index', inplace=True)
    return features

def sigmoid_process(start = 10, stop = 30, wgt_in = 0.5, wgt_out = 0.5):
    """Applies a sigmoid shifted by the start and stop range value.

    Note: disable inputs or outputs by setting wgt_in or wgt_out to 0 respectively.
    """
    
    def postprocess(features_in, features_out):
        #wgt_in = wgt_in**(1/2)
        #wgt_out = wgt_out**(1/2)
   
        # create sigmoid function
        shift = (stop-start)/2+start
        scale = (stop-start)/4
        func = np.vectorize(lambda x: 1 / (1 + np.exp(-((x-shift)/scale))))

        features_in = features_in.apply(func)
        features_out = features_out.apply(func)
        return _combine_features([features_in, features_out], [wgt_in, wgt_out])

    return postprocess

def piecewise_process(mid = 100, slope1=1.0, slope2=0.1, wgt_in = 0.5, wgt_out = 0.5):
    """Apply weights based on two piecewise linear functions.                           

    Note: disable inputs or outputs by setting wgt_in or wgt_out to 0 respectively.                                    
    """ 
    def postprocess(features_in, features_out):                                                                        
        func = np.vectorize(lambda x: x*slope1 if x < mid else mid*slope1 +(x-mid)*slope2 )   
        
        features_in = features_in.apply(func)
        features_out = features_out.apply(func)
        return _combine_features([features_in, features_out], [wgt_in, wgt_out])                                       

    return postprocess 

def clipped_process(start = 3, stop = 50, wgt_in = 0.5, wgt_out = 0.5):
    """Clips values at the top and bottom of the range to 0 and the stop value respectively.

    Note: disable inputs or outputs by setting wgt_in or wgt_out to 0 respectively.
    """
    def postprocess(features_in, features_out):
        #wgt_in = wgt_in**(1/2)
        #wgt_out = wgt_out**(1/2)
   
        # create clip function
        func = np.vectorize(lambda x: 0 if x < start else (stop if x > stop else x))

        features_in = features_in.apply(func)
        features_out = features_out.apply(func)
        return _combine_features([features_in, features_out], [wgt_in, wgt_out])

    return postprocess

def noop_process(wgt_in, wgt_out):
    """Simply uses input and output weights as is with no post-processing
    except for wgt_in and wgt_out.
    
    Note: disable inputs or outputs by setting wgt_in or wgt_out to 0 respectively.
    """

    def postprocess(features_in, features_out):
        #wgt_in = wgt_in**(1/2)
        #wgt_out = wgt_out**(1/2)
        
        return _combine_features([features_in, features_out], [wgt_in, wgt_out])

    return postprocess

def scaled_process(wgt_in = 1.0, wgt_out = 1.0, wgts=[0.8, 0.1, 0.1]):
    """Disable input or output by setting the weight to 0.

    The first weight corresponds to feature matrix that are normalized by the neurons
    input and output size.  The second weight corresponds to a matrix where each
    feature is treated independently and equally.  The final weight corresponds
    to the size of the neuron.

    Emphasizing the first feature matrix will give more weight to the biggest relative weights.
    Emphasizing the second will give more weight to the  smaller connections.  Emphasizing
    the final one will give more weight to the absolute neuron size.

    Note: default wgts for roioverlap_features and projection features is [0.25,0.6,0.15]
    
    Args:
        wgt_in (float): weight for input feature vector
        wgt_out (float): weight for output features vector
        wgts (list): weights for the three feature matrices produced by this function
    """

    def postprocess(features_in, features_out):
        # re-create size array, norm everything
        sz_in = pd.DataFrame(features_in.sum(axis=1), columns=["post"])
        sz_out = pd.DataFrame(features_out.sum(axis=1), columns=["pre"])

        # normalize the inputs and outputs
        features_in = features_in.div(features_in.sum(axis=1), axis=0)
        features_in = features_in.mul(1.0, fill_value=0)
        features_out = features_out.div(features_out.sum(axis=1), axis=0)
        features_out = features_out.mul(1.0, fill_value=0)

        # combine input and output into one array
        features = _combine_features([features_in, features_out], [wgt_in, wgt_out])
        
        # make size only features
        features_sz = _combine_features([sz_in, sz_out], [wgt_in, wgt_out])

        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import normalize
  
        features_arr = features.values

        # normalize each feature
        scaledfeatures = StandardScaler().fit_transform(features_arr)
        aux_features_arr = features_sz.values
        aux_scaledfeatures = StandardScaler().fit_transform(aux_features_arr)
    
        # ensure rows have the same magnitude -- this hopefully does not
        # impact the original vectors too much since each row should probably
        # be normalized in some way.
        features_norm = normalize(features_arr, axis=1, norm='l2')
        scaledfeatures_norm = normalize(scaledfeatures, axis=1, norm='l2')

        scaled_featnames=[]
        for name in features.columns.to_list():
            scaled_featnames.append(str(name)+"-s")

        features_normdf = pd.DataFrame(features_norm, index=features.index.values.tolist(), columns=features.columns.values.tolist())
        scaledfeatures_normdf = pd.DataFrame(scaledfeatures_norm, index=features.index.values.tolist(), columns=scaled_featnames)
        aux_scaledfeaturesdf = pd.DataFrame(aux_scaledfeatures, index=features.index.values.tolist(), columns=features_sz.columns.values.tolist())

        return _combine_features([features_normdf, scaledfeatures_normdf, aux_scaledfeaturesdf], wgts)

    return postprocess 

def extract_roioverlap_features(npclient, neuronlist,
        postprocess=scaled_process(0.5, 0.5, [0.66,0,0.33]),
        sym_excl = ["(L)", "(R)"], roilist=None, POLYADIC_HACK=5):
    """Extract simple ROI overlap features. 
    
    Note: Some tests show that the scaled_process works well when emphasizing the input/output normalization
    and absolute size feature matrices.  Scaling each feature independently may not make as much sense since
    the synapse counts for a neuron across ROIs are probably in the same scale.  Doing a simple piecewise_process
    also appears to work well by emphasizing the smaller connections a little more (e.g. mid=100, slope1=1, slope2=0.1).

    Args:
        npclient (object): neuprint client object
        neuronlist (list): list of body ids
        postprocess (func): set with *_process function (each function allows users to set input and output to 0)
        (default sets outputs to have 5x weight of inputs as a hack for Dropholia polyadic connections)
        sym_excl (list(str)): a list of substrings to exclude from the ROI list (specific to hemibrain for now)
        roilist (list): custom list of ROIs to use
        POLYADIC_HACK (int): temporary hack to account for outputs typically driving multiple inputs, set this
        to one if synapses are 1:1 (TODO: either base on the number of connections or reweight output and input
        to have roughly the same magnitude)

    Returns:
        dataframe: index: body ids; columns: different features 
    """

    # TODO: handle left/right symmetry

    # >= to use ROI feature
    PRE_IMPORTANCECUTOFF = 0
    POST_IMPORTANCECUTOFF = 0

    overlapquery=f"WITH {{}} AS TARGETS MATCH (n :Neuron) WHERE n.bodyId in TARGETS\
        RETURN n.bodyId AS bodyId, n.roiInfo AS roiInfo"
    
    # relevant rois
    inrois = set()
    outrois = set()

    # roi query if needed
    superrois = None
    if roilist is None:
        roiquery = f"MATCH (m :Meta) RETURN m.superLevelRois AS rois"
        roires = npclient.fetch_custom(roiquery)
        superrois = set(roires["rois"].iloc[0])
    else:
        superrois = set(roilist)

    # if left/right symmetry map ROIs to common name
    roi2roi = {}
    for roi in superrois:
        troi = roi
        for excl in sym_excl: 
            troi = troi.replace(excl, "")
        roi2roi[roi] = troi
    bodyinfo_in = {}
    bodyinfo_out = {}

    for iter1 in range(0, len(neuronlist), 500):
        currblist = neuronlist[iter1:(iter1+500)]
        print(f"fetch batch {iter1}")
        
        query = overlapquery.format(currblist)
        res = npclient.fetch_custom(query)

        for idx, row in res.iterrows():
            roiinfo = json.loads(row["roiInfo"])
            bodyinfo_in[row["bodyId"]] = {}
            bodyinfo_out[row["bodyId"]] = {}

            for roi, val in roiinfo.items():
                if roi in superrois:
                    roi = roi2roi[roi]
                    if val.get("pre", 0) > PRE_IMPORTANCECUTOFF:
                        inrois.add(roi)
                        if roi in bodyinfo_in[row["bodyId"]]:
                            bodyinfo_in[row["bodyId"]][roi] += val.get("pre", 0) 
                        else:
                            bodyinfo_in[row["bodyId"]][roi] = val.get("pre", 0) 
                    if val.get("post", 0) > POST_IMPORTANCECUTOFF:
                        outrois.add(roi)
                        if roi in bodyinfo_out[row["bodyId"]]:
                            bodyinfo_out[row["bodyId"]][roi] += (POLYADIC_HACK * val.get("post", 0))
                        else:
                            bodyinfo_out[row["bodyId"]][roi] = (POLYADIC_HACK * val.get("post", 0))

      # generate feature vector for the projectome and the pre and post sizes
    features_in = np.zeros((len(neuronlist), len(inrois)))
    features_out = np.zeros((len(neuronlist), len(outrois)))

    # populate feature arrays
    def load_features(features_arr, io_list, common_io):
        for iter1, bodyid in enumerate(neuronlist):
            featurevec = [0]*len(common_io)
            for idx, roi in enumerate(common_io):
                if bodyid in io_list:
                    if roi in io_list[bodyid]:
                        val = io_list[bodyid][roi]
                        featurevec[idx] = val
            features_arr[iter1] = featurevec

    load_features(features_in, bodyinfo_in, inrois)
    load_features(features_out, bodyinfo_out, outrois)

    # construct dataframes for the features
    featurenames = []
    for froi in inrois:
        featurenames.append(froi + "<=")
    features_in = pd.DataFrame(features_in, index=neuronlist, columns=featurenames) 
    
    featurenames = []
    for froi in outrois:
        featurenames.append(froi + "=>")
    features_out = pd.DataFrame(features_out, index=neuronlist, columns=featurenames) 

    return postprocess(features_in, features_out)


def extract_projection_features(npclient, neuronlist,
        postprocess=scaled_process(0.5, 0.5, [0.66,0,0.33]), sym_excl = ["(L)", "(R)"], roilist=None ):
    """Extract features from a list of neurons.

    This function generates features based on ROI connectivity pattern
    of the neurons and their partners.
    
    Note: Some tests show that the scaled_process works well when emphasizing the input/output normalization
    and absolute size feature matrices.  Scaling each feature independently may not make as much sense since
    the synapse counts for a neuron across ROIs are probably in the same scale.
    
    Args:
        npclient (object): neuprint client object
        neuronlist (list): list of body ids
        postprocess (func): set with *_process function (each function allows users to set input and output to 0)
        sym_excl (list(str)): a list of substrings to exclude from the ROI list (specific to hemibrain for now)
        roilist (list): custom list of ROIs to use
    Returns:
        dataframe: index: body ids; columns: different features 

    TODO:
        * allow users to specify ROI/sub-ROIs to be used (currently only
        top-level ROIs are used)
        * expose more parameters
        * allow users to specify 'symmetric' ROIs
        * allow users to define neuron groups that can be treated as
        a column for features (essentially treating certain neurons as an ROI)

    Note: Current queries limit partner connections to those that are traced or leaves.

    """
    # >= give significant connections that should be weighted in the connectome
    # when normalizing across secondary connections, only consider those above the threshold
    IMPORTANCECUTOFF = 3

    # query db X neurons at a time (restrict to traced neurons for the targets)
    # make a set for all unique ROI labels
    outputsquery=f"WITH {{}} AS TARGETS MATCH(n :Neuron)-[x :ConnectsTo]->(m :Neuron) WHERE n.bodyId in TARGETS AND\
    m.status=\"Traced\" AND x.weight >= {IMPORTANCECUTOFF} RETURN n.bodyId AS body1, x.roiInfo AS info, m.bodyId AS body2, m.roiInfo AS minfo"

    inputsquery=f"WITH {{}} AS TARGETS MATCH(n :Neuron)<-[x :ConnectsTo]-(m :Neuron) WHERE n.bodyId in TARGETS AND\
    m.status=\"Traced\" AND x.weight >= {IMPORTANCECUTOFF} RETURN n.bodyId AS body1, x.roiInfo AS info, m.bodyId AS body2, m.roiInfo AS minfo"

    # relevant rois
    inrois = set()
    outrois = set()

    secrois = set()

    # roi query if needed
    superrois = None
    if roilist is None:
        roiquery = f"MATCH (m :Meta) RETURN m.superLevelRois AS rois"
        roires = npclient.fetch_custom(roiquery)
        superrois = set(roires["rois"].iloc[0])
        superrois.add("None")
    else:
        superrois = set(roilist)


    # if left/right symmetry map ROIs to common name
    roi2roi = {}
    for roi in superrois:
        troi = roi
        for excl in sym_excl: 
            troi = troi.replace(excl, "")
        roi2roi[roi] = troi

    # biody inputs (body => roi => [unsorted weights])
    body2inputs = {}
    body2outputs = {}
    body2incount = {}
    body2outcount = {}

    # determine the inputs and outputs for each neuron and find their ROI
    # synapse distribution
    for iter1 in range(0, len(neuronlist), 50):
        currblist = neuronlist[iter1:iter1+50]
        queryin = inputsquery.format(currblist)
        queryout = outputsquery.format(currblist)

        print(f"fetch batch {iter1}")
        resin = npclient.fetch_custom(queryin)
        resout = npclient.fetch_custom(queryout)

        for index, row in resin.iterrows():
            b1 = row["body1"]
            b2 = row["body2"]
            if b1 == b2:
                continue # ignore autapses for now!
            roiinfo = json.loads(row["info"])
            minfo = json.loads(row["minfo"])
            secrois = secrois.union(superrois.intersection(set(minfo.keys())))

            # only examine neurons that have a certain number of synapses
            for key, val in roiinfo.items():
                # restrict to super ROIS
                if key in superrois:
                    # add roi and output for roi
                    inrois.add(key)
                    if b1 not in body2inputs:
                        body2inputs[b1] = {}
                        body2incount[b1] = 0
                    if key not in body2inputs[b1]:
                        body2inputs[b1][key] = []
                    if val.get("post", 0) >= IMPORTANCECUTOFF:
                        body2inputs[b1][key].append((val.get("post", 0), minfo))
                    body2incount[b1] += val.get("post", 0)

        for index, row in resout.iterrows():
            b1 = row["body1"]
            b2 = row["body2"]
            if b1 == b2:
                continue # ignore autapses for now!

            roiinfo = json.loads(row["info"])

            for key, val in roiinfo.items():
                if key in superrois:
                    # add roi and output for roi
                    outrois.add(key)
                    if b1 not in body2outputs:
                        body2outputs[b1] = {}
                        body2outcount[b1] = 0
                    if key not in body2outputs[b1]:
                        body2outputs[b1][key] = []
                    if val.get("post", 0) >= IMPORTANCECUTOFF:
                        body2outputs[b1][key].append((val.get("post", 0), json.loads(row["minfo"])))
                    body2outcount[b1] += val.get("post", 0)


    # generate feature vector for the projectome (separatee upstream and downstream)
    features_in = np.zeros((len(neuronlist), 2*len(secrois)*len(inrois)))
    features_out = np.zeros((len(neuronlist), 2*len(secrois)*len(outrois)))

    def load_features(features_arr, firstrois, body2io):
        # populate feature arrays.  Currently each feature row is normalized by the synapse input/output counts
        for iter1, bodyid in enumerate(neuronlist):
            featurevec = []

            for roi in firstrois:
                secroifeat = [0]*2*len(secrois)
                if bodyid in body2io:
                    if roi in body2io[bodyid]:
                        partners = body2io[bodyid][roi]
                        roitot = 0
                        roi2normincount = {}
                        roi2normoutcount = {}
                        for (count, roiinfo) in partners:
                            roitot += count

                        for (count, roiinfo) in partners:
                            pretot = 0
                            posttot = 0
                            for secroi in secrois:
                                if secroi in roiinfo:
                                    pretot += roiinfo[secroi].get("pre", 0)
                                    posttot += roiinfo[secroi].get("post", 0)
                            for secroi in secrois:
                                if secroi in roiinfo:
                                    if secroi not in roi2normincount:
                                        roi2normincount[secroi] = 0
                                    if roiinfo[secroi].get("pre", 0) > 0:
                                        roi2normincount[secroi] += ((roiinfo[secroi].get("pre", 0)/pretot)*(count/roitot))*(roitot)

                                    if secroi not in roi2normoutcount:
                                        roi2normoutcount[secroi] = 0
                                    if roiinfo[secroi].get("post", 0) > 0:
                                        roi2normoutcount[secroi] += ((roiinfo[secroi].get("post", 0)/posttot)*(count/roitot))*(roitot)

                        # load input features
                        secroifeat = []
                        for secroi in secrois:
                            if secroi in roi2normincount:
                                secroifeat.append(roi2normincount[secroi])
                            else:
                                secroifeat.append(0)

                        # load output features
                        for secroi in secrois:
                            if secroi in roi2normoutcount:
                                secroifeat.append(roi2normoutcount[secroi])
                            else:
                                secroifeat.append(0)
                featurevec.extend(secroifeat)
            features_arr[iter1] = featurevec
    load_features(features_in, inrois, body2inputs)
    load_features(features_out, outrois, body2outputs)

    def set_features(features_arr, firstrois, dilim):
        # construct dataframes for the features
        featurenames = []
        equivclasses = {}

        for froi in firstrois:
            froi = roi2roi[froi]
            
            for sroi in secrois:
                sroi = roi2roi[sroi]
                key = froi + dilim + "(" + sroi +"<=)"
                if key in equivclasses:
                    newkey = key + "-" + str(len(equivclasses[key]))
                    equivclasses[key].append(newkey)
                    key = newkey
                else:
                    equivclasses[key] = [key]
                featurenames.append(key)
            for sroi in secrois:
                sroi = roi2roi[sroi]
                key = froi + dilim + "(" + sroi +"=>)"
                if key in equivclasses:
                    newkey = key + "-" + str(len(equivclasses[key]))
                    equivclasses[key].append(newkey)
                    key = newkey
                else:
                    equivclasses[key] = [key]
                featurenames.append(key)
        features = pd.DataFrame(features_arr, index=neuronlist, columns=featurenames) 
        
        if len(equivclasses):
            equivlists = []
            for key, arr in equivclasses.items():
                equivlists.append(arr)
            
            features = _sort_equiv_features(features, equivlists)
        return features
    features_in = set_features(features_in, inrois, "<=")
    features_out = set_features(features_out, outrois, "=>")

    return postprocess(features_in, features_out)

def compute_connection_similarity_features(npclient, neuronlist,
        use_saved_types=True, customtypes={}, postprocess=scaled_process(0.5, 0.5, [0.9,0.0,0.1]),
        sort_types=True, pattern_only=False, minconn=3, roi_restriction=None,
        dump_replay=False, replay_data = None, morph_only=False, statuses=["Anchor"],
        hack_test=False):
    """Computes an pairwise adjacency matrix for the given set of neurons.

    This function looks at inputs and outputs for the set of neurons.  The connections
    are restricted to the roi_restriction if provided.  The caller can specify
    whether common inputs and outputs should be grouped based on type.  The returned
    adjacency matrix will sort the body ids based on clustering.

    Note: postprocessing seems to work best by balancing between a matrix that emphasizes
    stronger connections and one that treates all connections the same.  Scaling the connections
    per neurons is probably important due to reconstruction incompleteness and some neurons
    just having bigger connections.  Similarly, because connections to a given neuron might
    have different impact compared to connections to another neuron, some feature indepedent
    scaling is probably also useful.

    Args:
        npclient (object): neuprint client object
        neuronlist (list): list of body ids
        use_saved_types(boolean): use types stored in neuprint already to guide clustering
        customtypes (dict or df): mapping of body ids to a cluster id (over-rules pre-exising types)
        postprocess (func): set with *_process function (each function allows users to set input and output to 0)
        (other paramters)
        sort_types (boolean): True to sort, False to collapse
        pattern_only (boolean): If true, do not consider the specific id of the neuron or type, only the
        distribution of connection weights
        minconn (int): Only consider connections >= minconn
        roi_restriction (list): ROIs to restrict feature extraction (default: no restriction)
        dump_replay (boolean): dumps features after parsing neuprint
        replay_data (tuple): data to enable replay
        morph_only (boolean): EXPERIMENTAL FEATURE only uses the root morpho type for each prior type
        statuses (list of strings): only consider target neurons with the given statuses; default "Anchor" only
    Returns:
        dataframe: index: body ids; columns: different features 
    
    """

    if pattern_only:
        sort_types = True
        use_saved_types = False
        customtypes = {}

    # set of all common input and outputs
    commonin = set()
    commonout = set()

    # inputs and outputs for each body id
    # format: (name or id}: val
    inputs_list = {} 
    outputs_list = {} 

    body2type = {}
  
    features_in = None
    features_out = None
    import re

    if customtypes is not None and type(customtypes) == pd.DataFrame:
        customtypes =  dict(zip(customtypes["bodyid"], customtypes["type"]))

    if replay_data is not None:
        features_in, features_out, commonin, commonout, body2type = replay_data 
    else:
        for iter1 in range(0, len(neuronlist), 100):
            currlist = neuronlist[iter1:iter1+100]
            
            outputsquery=f"""\
                WITH {currlist} AS TARGETS 
                MATCH(n :Neuron)-[x :ConnectsTo]->(m :Neuron) 
                WHERE n.bodyId in TARGETS AND m.status IN {statuses} AND x.weight >= {minconn} 
                RETURN n.bodyId AS body1, x.roiInfo AS info, m.bodyId AS body2, m.type AS type, x.weight AS weight 
                ORDER BY weight DESC
                """

            inputsquery=f"""\
                WITH {currlist} AS TARGETS 
                MATCH(n :Neuron)<-[x :ConnectsTo]-(m :Neuron) 
                WHERE n.bodyId in TARGETS AND m.status IN {statuses} AND x.weight >= {minconn} 
                RETURN n.bodyId AS body1, x.roiInfo AS info, m.bodyId AS body2, m.type AS type, x.weight AS weight 
                ORDER BY weight DESC
                """
         
            def collect_features(query, io_list, common_io):
                res = npclient.fetch_custom(query)
                for idx, row in res.iterrows():
                    if row["body1"] == row["body2"]:
                        continue
                    feat_type = row["body2"]

                    row_type = row["type"]
                    if row_type is not None:
                        row_type = row_type.replace("put_", "")
                        row_type = row_type.replace("_pct", "")

                    if not sort_types:
                        if feat_type in customtypes:
                            feat_type = customtypes[feat_type]
                        elif use_saved_types and row_type is not None and row_type != "":
                            feat_type = row_type

                        if morph_only:
                            if len(re.findall(r"^.*_[a-z]$", feat_type)) > 0:
                                feat_type = feat_type[:-2]
                    else:
                        common_type = ""
                        if feat_type in customtypes:
                            common_type = customtypes[feat_type]
                        elif use_saved_types and row_type is not None and row_type != "":
                            common_type = row_type
                        elif pattern_only:
                            common_type = "ph"
                        
                        if morph_only:
                            if len(re.findall(r"^.*_[a-z]$", common_type)) > 0:
                                common_type = common_type[:-2]
                        if common_type != "":
                            body2type[feat_type] = common_type

                    roiinfo = json.loads(row["info"])

                    totconn = 0
                    # restrict connections to provided ROIs
                    if roi_restriction is not None:
                        for roi, val in roiinfo.items():
                            if roi not in roi_restriction:
                                continue
                            totconn += val.get("post", 0)
                    else:
                        totconn = row["weight"]

                    if totconn >= minconn:
                        if row["body1"] not in io_list:
                            io_list[row["body1"]] = {}

                        if hack_test:
                            # only take top 5 plus buffer
                            if len(io_list[row["body1"]]) >= 5:
                                count_list = []
                                for name, count in io_list[row["body1"]].items():
                                    count_list.append(count)
                                count_list.sort()
                                count_list.reverse()
                                if (count_list[4] - (count_list[4]**(1/2))) > totconn:
                                    continue
                            
                        if feat_type not in io_list[row["body1"]]:
                            io_list[row["body1"]][feat_type] = 0
                        io_list[row["body1"]][feat_type] += totconn
                        common_io.add(feat_type)
    
            collect_features(outputsquery, outputs_list, commonout)
            collect_features(inputsquery, inputs_list, commonin)

        features_in = np.zeros((len(neuronlist), len(commonin)))
        features_out = np.zeros((len(neuronlist), len(commonout)))
        
        def load_features(features_arr, io_list, common_io):
            for iter1, bodyid in enumerate(neuronlist):
                featurevec = [0]*len(common_io)

                # load input features
                if bodyid in io_list:
                    for idx, io in enumerate(common_io):
                        if io in io_list[bodyid]:
                            val = io_list[bodyid][io]
                            featurevec[idx] = val
                features_arr[iter1] = featurevec

        load_features(features_in, inputs_list, commonin)
        load_features(features_out, outputs_list, commonout)

        # return intermediate data if requested
        if dump_replay:
            return (features_in, features_out, commonin, commonout, body2type)

    def set_features(features_arr, common_io, dilin):
        featurenames = []
        equivclasses = {}

        # if body2type is loaded, sort type connectivity
        for io in common_io:
            key = str(io)+dilin
            if io in body2type:
                key = "c-" + str(body2type[io]) + "(" + str(io) + ")" + dilin
                if body2type[io] in equivclasses:
                    equivclasses[body2type[io]].append(key)
                else:
                    equivclasses[body2type[io]] = [key]
            featurenames.append(key)
        features = pd.DataFrame(features_arr, index=neuronlist, columns=featurenames) 

        # sort connections to neurons in the same class
        if len(equivclasses):
                equivlists = []
                for key, arr in equivclasses.items():
                    equivlists.append(arr)
                
                features = _sort_equiv_features(features, equivlists)
        return features

    features_in = set_features(features_in, commonin, "=>")
    features_out = set_features(features_out, commonout, "<=")

    return postprocess(features_in, features_out)

def find_max_differences(features, body1, body2, numfeatures = 10):
    """Find features that are most different for body1 and body2.

    Args:
        features (dataframe): features for a set of bodies
        body1 (int): id for body/neuron
        body2 (int): id for body/neuron
        numfeatures (int): number of features to return for largest agreements and differences
    Return:
        (dataframe): Show X largest values and X largest differences

    Note: should examine features before PCA or other domain reduction.
    """

    b1feat = features.loc[body1]
    b2feat = features.loc[body2]

    # find biggest differences and the largest combined value
    diff = abs(b1feat-b2feat).sort_values(ascending=False)                                                           
    comb = (b1feat+b2feat).sort_values(ascending=False)                                                              

    if len(diff) < numfeatures//2:                                                                                   
        numfeatures = len(diff)//2

    # restrict dataframe to the two relevant bodies                                                                          
    features_restr = features.loc[[body1, body2]]
    
    diff_feat = abs(b1feat-b2feat)
    diff_feat.name = "diff"
    features_restr = features_restr.append(diff_feat)

    # restrict featuress to most interesting
    restr = list(diff.index[0:numfeatures])
    restr.extend(list(comb.index[0:numfeatures]))
    features_restr = features_restr[restr]                                                                                                           
    
    return features_restr

def find_max_variance(features, numfeatures=40):
    """Find features with the most variance

    Args:
        features (dataframe): features for a set of bodies
        numfeatures (int): number of features to return with the most variance
    Return:
        (dataframe): Show X most variance

    Note: should examine features before PCA or other domain reduction.
    """
    min_val = features.min()
    max_val = features.max()
    
    diff = abs(max_val-min_val).sort_values(ascending=False)
    
    restr = list(diff.index[0:numfeatures])
    features_restr = features[restr]

    return features_restr


def reduce_features(features, num_components=400):
    """Reduce dimensionality of features.

    This function wraps PCA.

    Args:
        features (dataframe): matrix of features
        num_components (int): number of pca components
    Returns:
        dataframe: A matrix contain features that can be used for clustering.
    """

    from sklearn.decomposition import PCA
    pca_num = min(features.shape[1], num_components)
    pca_num = min(pca_num, features.shape[0])
    pca = PCA(n_components=pca_num)
    principalComponents = pca.fit_transform(features)
    print(f"PCA Explained Variance: {sum(pca.explained_variance_ratio_)}")
    
    return pd.DataFrame(principalComponents, index=features.index.values.tolist())


def nonlinear_reduce_features(features, num_components=2):
    """Non-linear feature reduction.
    """
    from sklearn.manifold import TSNE
    reducedfeatures = TSNE(n_components=num_components).fit_transform(features.values)
    return pd.DataFrame(reducedfeatures, index=features.index.values.tolist())



##### helper functions #########


def _combine_features(feature_list, wgt_list):
    """Combines features based on provided weights.

    Args:
        feature_list (list): list of dataframes contain features
        wgt_list: weight to associate with each set of features
    Returns:
        (dataframe): A combined set of features

    Note: all features should have the same number of rows.
    """
    nfeature_list = []
    featurenames = []

    for idx, wgt in enumerate(wgt_list):
        # ignore really small values
        if wgt > 0.0000001:
            nfeature_list.append(feature_list[idx].values*wgt)
            featurenames.extend(feature_list[idx].columns.values.tolist())

    if len(nfeature_list) == 0:
        raise RuntimeError("no features to combine")
    combofeatures = np.concatenate(tuple(nfeature_list), axis=1)
    return pd.DataFrame(combofeatures, index=feature_list[0].index.values.tolist(), columns=featurenames) 


def _process_chunk(df_chunk_pair):
    df_chunk, equivlists = df_chunk_pair
    for idx, row in df_chunk.iterrows():
        rowdict = row.to_dict()
        for group in equivlists:
            vals = []
            labels = []
            for item in group:
                vals.append(rowdict[item])
                labels.append(item)
            vals.sort(reverse=True)
            for idx2, label in enumerate(labels):
                rowdict[label] = vals[idx2]

        for idx2, cname in enumerate(row.index.tolist()):
            row[idx2] = rowdict[cname] 
    return df_chunk

def _sort_equiv_features(features, equivlists):
    """Sort related features from big to small.

    Note: this function uses multiple processes
    """

    # break dataframe into chunks
    CHUNK_SIZE = 100
    df_chunks = []
    for chunk_start in range(0, len(features), CHUNK_SIZE):
        df_chunks.append( (features.iloc[chunk_start:(chunk_start+CHUNK_SIZE)], equivlists) )


    # process chunks in parallel
    from multiprocessing import Pool
    with Pool(processes=None) as pool:
        sorted_chunks = pool.map(_process_chunk, df_chunks)
    
    return pd.concat(sorted_chunks)
