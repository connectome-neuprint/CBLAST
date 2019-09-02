"""Neuron feature generation.

Contains a set of functions for extracting features for a list of neurons.
"""

import json
import numpy as np
import pandas as pd

def extract_roioverlap_features(neuronlist, dataset, npclient, roilist=None, preprocess=True, wgts=[0.25,0.6,0.15]):
    """Extract simple ROI overlap features. 

    Args:
        neuronlist (list): list of body ids
        dataset (str): name of neuprint dataset
        npclient (object): neuprint client object
        roilist (list): custom list of ROIs to use
        preprocess (boolean): if true, features produced by the algorithm are combined in weighted manner
        wgts (list): a list of weights for each feature set (unscaled vs scaled vs size features)
    Returns:
        (dataframe, dataframe): A tuple containing roi features
        for each body id and pre and post size per body id (only one df returned if preprocess is true)

    """

    # >= to use ROI feature
    PRE_IMPORTANCECUTOFF = 10
    POST_IMPORTANCECUTOFF = 50

    overlapquery=f"WITH {neuronlist} AS TARGETS MATCH (n :`{dataset}-Neuron`) WHERE n.bodyId in TARGETS\
        RETURN n.bodyId AS bodyId, n.roiInfo AS roiInfo"

    # relevant rois
    inrois = set()
    outrois = set()

    # roi query if needed
    superrois = None
    if roilist is None:
        roiquery = f"MATCH (m :Meta) WHERE m.dataset=\"{dataset}\" RETURN m.superLevelRois AS rois"
        roires = npclient.fetch_custom(roiquery)
        superrois = set(roires["rois"].iloc[0])
    else:
        superrois = set(roilist)

    res = npclient.fetch_custom(overlapquery)

    bodyinfo_in = {}
    bodyinfo_out = {}
    for idx, row in res.iterrows():
        roiinfo = json.loads(row["roiInfo"])
        bodyinfo_in[row["bodyId"]] = {}
        bodyinfo_out[row["bodyId"]] = {}

        for roi, val in roiinfo.items():
            if roi in superrois:
                if val["pre"] > PRE_IMPORTANCECUTOFF:
                    inrois.add(roi)
                    bodyinfo_in[row["bodyId"]][roi] = val["pre"] 
                if val["post"] > POST_IMPORTANCECUTOFF:
                    outrois.add(roi)
                    bodyinfo_out[row["bodyId"]][roi] = val["post"] 
   
      # generate feature vector for the projectome and the pre and post sizes
    featurearray = np.zeros((len(neuronlist), len(inrois)+len(outrois)))
    featureszarray = np.zeros((len(neuronlist), 2))

    # populate feature arrays.  Currently each feature row is normalized by the synapse input/output counts
    for iter1, bodyid in enumerate(neuronlist):
        featurevec = [0]*(len(inrois)+len(outrois))
        insize = 0
        for roi in inrois:
            if bodyid in bodyinfo_in:
                if roi in bodyinfo_in[bodyid]:
                    val = bodyinfo_in[bodyid][roi]
                    insize += val
        
        for idx, roi in enumerate(inrois):
            if bodyid in bodyinfo_in:
                if roi in bodyinfo_in[bodyid]:
                    val = bodyinfo_in[bodyid][roi]
                    featurevec[idx] = val/insize

        outsize = 0
        for roi in outrois:
            if bodyid in bodyinfo_out:
                if roi in bodyinfo_out[bodyid]:
                    val = bodyinfo_out[bodyid][roi]
                    outsize += val
        
        for idx, roi in enumerate(outrois):
            if bodyid in bodyinfo_out:
                if roi in bodyinfo_out[bodyid]:
                    val = bodyinfo_out[bodyid][roi]
                    featurevec[len(inrois)+idx] = val/outsize

        featureszarray[iter1] = [insize, outsize]
        featurearray[iter1] = featurevec

    # construct dataframes for the features
    featurenames = []
    for froi in inrois:
        featurenames.append(froi + "<=")
    for froi in outrois:
        featurenames.append(froi + "=>")
    
    features = pd.DataFrame(featurearray, index=neuronlist, columns=featurenames) 
    features_sz = pd.DataFrame(featureszarray, index=neuronlist, columns=["post", "pre"]) 
   
    if preprocess:
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import normalize
  
        features_arr = features.values

        scaledfeatures = StandardScaler().fit_transform(features_arr)
        aux_features_arr = features_sz.values
        aux_scaledfeatures = StandardScaler().fit_transform(aux_features_arr)
    
        # ensure rows have the same magnitude -- this hopefully does not
        # impact the original vectors too much since each row should probably
        # be normalized in some way.
        features_norm = normalize(features_arr, axis=1, norm='l2')
        scaledfeatures_norm = normalize(scaledfeatures, axis=1, norm='l2')

        features_normdf = pd.DataFrame(features_norm, index=features.index.values.tolist(), columns=features.columns.values.tolist())
        scaledfeatures_normdf = pd.DataFrame(scaledfeatures_norm, index=features.index.values.tolist(), columns=features.columns.values.tolist())
        aux_scaledfeaturesdf = pd.DataFrame(aux_scaledfeatures, index=features.index.values.tolist(), columns=features_sz.columns.values.tolist())

        return combine_features([features_normdf, scaledfeatures_normdf, aux_scaledfeaturesdf], wgts)

    return features, features_sz



def extract_projection_features(neuronlist, dataset, npclient, roilist=None, preprocess=True, wgts=[0.25,0.6,0.15], leftright=False):
    """Extract features from a list of neurons.

    This function generates features based on ROI connectivity pattern
    of the neurons and their partners.

    Args:
        neuronlist (list): list of body ids
        dataset (str): name of neuprint dataset
        npclient (object): neuprint client object
        roilist (list): custom list of ROIs to use
        preprocess (boolean): if true, features produced by the algorithm are combined in weighted manner
        wgts (list): a list of weights for each feature set (unscaled vs scaled vs size features)
        leftright (boolean): true to treat left and right ROIs as symmetric
    Returns:
        (dataframe, dataframe): A tuple containing projection features
        for each body id and pre and post size per body id (only one df returned if preprocess is true)

    TODO:
        * allow users to specify ROI/sub-ROIs to be used (currently only
        top-level ROIs are used)
        * expose more parameters
        * allow users to specify 'symmetric' ROIs
        * allow users to define neuron groups that can be treated as
        a column for features (essentially treating certain neurons as an ROI)

    Note: Current queries limit partner connections to those that are traced or leaves.

    """

    # if left/right symmetry map ROIs to common name
    roi2roi = {}
    if roilist is not None and leftright:
        for roi in roilist:
            troi = roi.replace("(L)", "")
            troi = troi.replace(" (right)", "")
            troi = troi.replace(" (left)", "")
            roi2roi[roi] = troi


    # >= give significant connections that should be weighted in the connectome
    # when normalizing across secondary connections, only consider those above the threshold
    # ?? maybe should use percentil of connections rather than absolute weight?
    # ?? should include the number of neurons that make up this threshold -- need separate mechanism to normalize
    IMPORTANCECUTOFF = 2

    # query db X neurons at a time (restrict to traced neurons for the targets)
    # make a set for all unique ROI labels
    outputsquery=f"WITH {{}} AS TARGETS MATCH(x :`{dataset}-ConnectionSet`)-\
    [:From]->(n :`{dataset}-Neuron`) WHERE n.bodyId in TARGETS WITH collect(x) as\
    csets, n UNWIND csets as cset MATCH (cset)-[:To]->(m :`{dataset}-Neuron`) where\
    (m.status=~'.*raced' OR m.status=\"Leaves\") RETURN n.bodyId AS body1, cset.roiInfo AS info, m.bodyId AS body2, m.roiInfo AS minfo"

    inputsquery=f"WITH {{}} AS TARGETS MATCH(x :`{dataset}-ConnectionSet`)-\
    [:To]->(n :`{dataset}-Neuron`) WHERE n.bodyId in TARGETS WITH collect(x) as\
    csets, n UNWIND csets as cset MATCH (cset)-[:From]->(m :`{dataset}-Neuron`) where\
    (m.status=~'.*raced' OR m.status=\"Leaves\") RETURN n.bodyId AS body1, cset.roiInfo AS info, m.bodyId AS body2, m.roiInfo AS minfo"

    # relevant rois
    inrois = set()
    outrois = set()

    secrois = set()

    # roi query if needed
    superrois = None
    if roilist is None:
        roiquery = f"MATCH (m :Meta) WHERE m.dataset=\"{dataset}\" RETURN m.superLevelRois AS rois"
        roires = npclient.fetch_custom(roiquery)
        superrois = set(roires["rois"].iloc[0])
    else:
        superrois = set(roilist)

    # body inputs (body => roi => [unsorted weights])
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
                    if val["post"] >= IMPORTANCECUTOFF:
                        body2inputs[b1][key].append((val["post"], minfo))
                    body2incount[b1] += val["post"]

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
                    if val["post"] >= IMPORTANCECUTOFF:
                        body2outputs[b1][key].append((val["post"], json.loads(row["minfo"])))
                    body2outcount[b1] += val["post"]


    # generate feature vector for the projectome and the pre and post sizes
    featurearray = np.zeros((len(neuronlist), 2*len(secrois)*len(inrois)+2*len(secrois)*len(outrois)))
    featureszarray = np.zeros((len(neuronlist), 2))

    # populate feature arrays.  Currently each feature row is normalized by the synapse input/output counts
    for iter1, bodyid in enumerate(neuronlist):
        featurevec = []

        insize = 0
        for roi in inrois:
            secroifeat = [0]*2*len(secrois)
            if bodyid in body2inputs:
                if roi in body2inputs[bodyid]:
                    partners = body2inputs[bodyid][roi]
                    roitot = 0
                    globtot = body2incount[bodyid]
                    insize = globtot
                    roi2normincount = {}
                    roi2normoutcount = {}
                    for (count, roiinfo) in partners:
                        roitot += count

                    for (count, roiinfo) in partners:
                        pretot = 0
                        posttot = 0
                        for secroi in secrois:
                            if secroi in roiinfo:
                                pretot += roiinfo[secroi]["pre"]
                                posttot += roiinfo[secroi]["post"]
                        for secroi in secrois:
                            if secroi in roiinfo:
                                if secroi not in roi2normincount:
                                    roi2normincount[secroi] = 0
                                if roiinfo[secroi]["pre"] > 0:
                                    roi2normincount[secroi] += ((roiinfo[secroi]["pre"]/pretot)*(count/roitot))*(roitot/globtot)

                                if secroi not in roi2normoutcount:
                                    roi2normoutcount[secroi] = 0
                                if roiinfo[secroi]["post"] > 0:
                                    roi2normoutcount[secroi] += ((roiinfo[secroi]["post"]/posttot)*(count/roitot))*(roitot/globtot)


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

        outsize = 0
        for roi in outrois:
            secroifeat = [0]*2*len(secrois)
            if bodyid in body2outputs:
                if roi in body2outputs[bodyid]:
                    partners = body2outputs[bodyid][roi]
                    roitot = 0
                    globtot = body2outcount[bodyid]
                    outsize = globtot
                    roi2normincount = {}
                    roi2normoutcount = {}
                    for (count, roiinfo) in partners:
                        roitot += count

                    for (count, roiinfo) in partners:
                        pretot = 0
                        posttot = 0
                        for secroi in secrois:
                            if secroi in roiinfo:
                                pretot += roiinfo[secroi]["pre"]
                                posttot += roiinfo[secroi]["post"]
                        for secroi in secrois:
                            if secroi in roiinfo:
                                if secroi not in roi2normincount:
                                    roi2normincount[secroi] = 0
                                if roiinfo[secroi]["pre"] > 0:
                                    roi2normincount[secroi] += ((roiinfo[secroi]["pre"]/pretot)*(count/roitot))*(roitot/globtot)

                                if secroi not in roi2normoutcount:
                                    roi2normoutcount[secroi] = 0
                                if roiinfo[secroi]["post"] > 0:
                                    roi2normoutcount[secroi] += ((roiinfo[secroi]["post"]/posttot)*(count/roitot))*(roitot/globtot)

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
        featureszarray[iter1] = [insize, outsize]
        featurearray[iter1] = featurevec


    # construct dataframes for the features
    featurenames = []
    equivclasses = {}

    for froi in inrois:
        if len(roi2roi) > 0:
            froi = roi2roi[froi]

        for sroi in secrois:
            key = froi + "<=" + "(" + sroi +"<=)"
            if len(roi2roi) > 0:
                sroi = roi2roi[sroi]
                key = froi + "<=" + "(" + sroi +"<=)"
                if key in equivclasses:
                    newkey = key + "-" + str(len(equivclasses[key]))
                    equivclasses[key].append(newkey)
                    key = newkey
                else:
                    equivclasses[key] = [key]
            featurenames.append(key)
        for sroi in secrois:
            key = froi + "<=" + "(" + sroi + "=>)"
            if len(roi2roi) > 0:
                sroi = roi2roi[sroi]
                key = froi + "<=" + "(" + sroi +"=>)"
                if key in equivclasses:
                    newkey = key + "-" + str(len(equivclasses[key]))
                    equivclasses[key].append(newkey)
                    key = newkey
                else:
                    equivclasses[key] = [key]
            featurenames.append(key)
    for froi in outrois:
        if len(roi2roi) > 0:
            froi = roi2roi[froi]
        
        for sroi in secrois:
            key = froi + "=>" + "(" + sroi +"<=)"
            if len(roi2roi) > 0:
                sroi = roi2roi[sroi]
                key = froi + "=>" + "(" + sroi +"<=)"
                if key in equivclasses:
                    newkey = key + "-" + str(len(equivclasses[key]))
                    equivclasses[key].append(newkey)
                    key = newkey
                else:
                    equivclasses[key] = [key]
            featurenames.append(key)
        for sroi in secrois:
            key = froi + "=>" + "(" + sroi + "=>)"
            if len(roi2roi) > 0:
                sroi = roi2roi[sroi]
                key = froi + "=>" + "(" + sroi +"=>)"
                if key in equivclasses:
                    newkey = key + "-" + str(len(equivclasses[key]))
                    equivclasses[key].append(newkey)
                    key = newkey
                else:
                    equivclasses[key] = [key]
            featurenames.append(key)
    
    features = pd.DataFrame(featurearray, index=neuronlist, columns=featurenames) 
    
    if len(equivclasses):
        equivlists = []
        for key, arr in equivclasses.items():
            equivlists.append(arr)
        
        features = sort_equiv_features(features, equivlists)
    
    features_sz = pd.DataFrame(featureszarray, index=neuronlist, columns=["post", "pre"]) 
   
    if preprocess:
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import normalize
  
        features_arr = features.values

        scaledfeatures = StandardScaler().fit_transform(features_arr)
        aux_features_arr = features_sz.values
        aux_scaledfeatures = StandardScaler().fit_transform(aux_features_arr)
    
        # ensure rows have the same magnitude -- this hopefully does not
        # impact the original vectors too much since each row should probably
        # be normalized in some way.
        features_norm = normalize(features_arr, axis=1, norm='l2')
        scaledfeatures_norm = normalize(scaledfeatures, axis=1, norm='l2')

        features_normdf = pd.DataFrame(features_norm, index=features.index.values.tolist(), columns=features.columns.values.tolist())
        scaledfeatures_normdf = pd.DataFrame(scaledfeatures_norm, index=features.index.values.tolist(), columns=features.columns.values.tolist())
        aux_scaledfeaturesdf = pd.DataFrame(aux_scaledfeatures, index=features.index.values.tolist(), columns=features_sz.columns.values.tolist())

        return combine_features([features_normdf, scaledfeatures_normdf, aux_scaledfeaturesdf], wgts)

    return features, features_sz

def combine_features(feature_list, wgt_list):
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
        nfeature_list.append(feature_list[idx].values*wgt)
        featurenames.extend(feature_list[idx].columns.values.tolist())

    combofeatures = np.concatenate(tuple(nfeature_list), axis=1)
    return pd.DataFrame(combofeatures, index=feature_list[0].index.values.tolist(), columns=featurenames) 


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

def compute_connection_similarity_features(neuronlist, dataset, npclient, roi_restriction=None,
        minconn=3, collapse_type=True, normalize=True, preprocess=True, wgts=[0.8,0.1,0.1], customtypes={}):
    """Computes an pairwise adjacency matrix for the given set of neurons.

    This function looks at inputs and outputs for the set of neurons.  The connections
    are restricted to the roi_restriction if provided.  The caller can specify
    whether common inputs and outputs should be grouped based on type.  The returned
    adjacency matrix will sort the body ids based on clustering.

    Args:
        neuronlist (list): list of body ids
        dataset (str): name of neuprint dataset
        npclient (object): neuprint client object
        roi_restriction (list): ROIs to restrict feature extractio (default all)n
        minconn (int): Only consider connections >= minconn
        collapse_type (boolean): If true, clustering uses existing cell type info
        normalize (boolean): If true, each row is normalized to a unit vector
        preprocess (boolean): if true, features produced by the algorithm are combined in weighted manner
        wgts (list): a list of weights for each feature set (unscaled vs scaled vs size features)
    Returns:
        (dataframe): Distance matrix between provided neurosn
    

    Note: only connection to traced neurons are considered..

    """

   
    # set of all common input and outputs
    commonin = set()
    commonout = set()

    # inputs and outputs for each body id
    # format: (name or id}: val
    inputs_list = {} 
    outputs_list = {} 

    for iter1 in range(0, len(neuronlist), 100):
        currlist = neuronlist[iter1:iter1+100]
        outputsquery=f"WITH {currlist} AS TARGETS MATCH(x :`{dataset}-ConnectionSet`)-\
        [:From]->(n :`{dataset}-Neuron`) WHERE n.bodyId in TARGETS WITH collect(x) as\
        csets, n UNWIND csets as cset MATCH (cset)-[:To]->(m :`{dataset}-Neuron`) where\
        (m.status=~'.*raced' OR m.status=\"Leaves\") RETURN n.bodyId AS body1, cset.roiInfo AS info, m.bodyId AS body2, m.type AS type"

        inputsquery=f"WITH {currlist} AS TARGETS MATCH(x :`{dataset}-ConnectionSet`)-\
        [:To]->(n :`{dataset}-Neuron`) WHERE n.bodyId in TARGETS WITH collect(x) as\
        csets, n UNWIND csets as cset MATCH (cset)-[:From]->(m :`{dataset}-Neuron`) where\
        (m.status=~'.*raced' OR m.status=\"Leaves\") RETURN n.bodyId AS body1, cset.roiInfo AS info, m.bodyId AS body2, m.type AS type"
     

        outres = npclient.fetch_custom(outputsquery)
        currblist = neuronlist[iter1:iter1+50]
        queryin = inputsquery.format(currblist)
        queryout = outputsquery.format(currblist)

        print(f"fetch batch {iter1}")

        for idx, row in outres.iterrows():
            if row["body1"] == row["body2"]:
                continue
            feat_type = row["body2"]
            if feat_type in customtypes:
                feat_type = customtypes[feat_type]
            elif collapse_type and row["type"] is not None and row["type"] != "":
                feat_type = row["type"]
            
            roiinfo = json.loads(row["info"])

            totconn = 0
            for roi, val in roiinfo.items():
                if roi_restriction is not None:
                    if roi not in roi_restriction:
                        continue
                totconn += val["post"]

            if totconn >= minconn:
                if row["body1"] not in outputs_list:
                    outputs_list[row["body1"]] = {}
                if feat_type not in outputs_list[row["body1"]]:
                    outputs_list[row["body1"]][feat_type] = 0
                outputs_list[row["body1"]][feat_type] += totconn
                commonout.add(feat_type)

        inres = npclient.fetch_custom(inputsquery)
        for idx, row in inres.iterrows():
            if row["body1"] == row["body2"]:
                continue
            feat_type = row["body2"]
            if collapse_type and row["type"] is not None and row["type"] != "":
                feat_type = row["type"]
            roiinfo = json.loads(row["info"])

            totconn = 0
            for roi, val in roiinfo.items():
                if roi_restriction is not None:
                    if roi not in roi_restriction:
                        continue
                totconn += val["post"]

            if totconn >= minconn:
                if row["body1"] not in inputs_list:
                    inputs_list[row["body1"]] = {}
                if feat_type not in inputs_list[row["body1"]]:
                    inputs_list[row["body1"]][feat_type] = 0
                inputs_list[row["body1"]][feat_type] += totconn
                commonin.add(feat_type)


    # populate feature array
    features_arr = np.zeros((len(neuronlist), len(commonin) + len(commonout)))
    features_size_arr = np.zeros((len(neuronlist), 2))

    for iter1, bodyid in enumerate(neuronlist):
        featurevec = [0]*(len(commonin)+len(commonout))

        # load input features
        tot_in = 0
        if bodyid in inputs_list:
            for input in commonin:
                if input in inputs_list[bodyid]:
                    val = inputs_list[bodyid][input]
                    tot_in += val
            for idx, input in enumerate(commonin):
                if input in inputs_list[bodyid]:
                    val = inputs_list[bodyid][input]
                    if normalize:
                        val /= tot_in
                    featurevec[idx] = val

        # load output features
        tot_out = 0
        if bodyid in outputs_list:
            for output in commonout:
                if output in outputs_list[bodyid]:
                    val = outputs_list[bodyid][output]
                    tot_out += val
            for idx, output in enumerate(commonout):
                if output in outputs_list[bodyid]:
                    val = outputs_list[bodyid][output]
                    if normalize:
                        val /= tot_out
                    featurevec[len(commonin)+idx] = val
    
        features_arr[iter1] = featurevec
        features_size_arr[iter1] = [tot_in, tot_out]

    featurenames = []
    for input in commonin:
        featurenames.append(str(input)+"=>")
    for output in commonout:
        featurenames.append(str(output)+"<=")

    features = pd.DataFrame(features_arr, index=neuronlist, columns=featurenames) 
    features_sz = pd.DataFrame(features_size_arr, index=neuronlist, columns=["post", "pre"]) 
   
    if preprocess:
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import normalize
  
        features_arr = features.values

        scaledfeatures = StandardScaler().fit_transform(features_arr)
        aux_features_arr = features_sz.values
        aux_scaledfeatures = StandardScaler().fit_transform(aux_features_arr)
    
        # ensure rows have the same magnitude -- this hopefully does not
        # impact the original vectors too much since each row should probably
        # be normalized in some way.
        features_norm = normalize(features_arr, axis=1, norm='l2')
        scaledfeatures_norm = normalize(scaledfeatures, axis=1, norm='l2')

        scaled_featnames=[]
        for name in featurenames:
            scaled_featnames.append(str(name)+"-2")

        features_normdf = pd.DataFrame(features_norm, index=features.index.values.tolist(), columns=features.columns.values.tolist())
        scaledfeatures_normdf = pd.DataFrame(scaledfeatures_norm, index=features.index.values.tolist(), columns=scaled_featnames)
        aux_scaledfeaturesdf = pd.DataFrame(aux_scaledfeatures, index=features.index.values.tolist(), columns=features_sz.columns.values.tolist())

        return combine_features([features_normdf, scaledfeatures_normdf, aux_scaledfeaturesdf], wgts)

    return features, features_sz

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

def _sort_equiv_features(features, equivlists):
    """Sort related features from big to small.
    """
   
    for idx, row in res.iterrows():
        for group in equivlists:
            res = list(row[group])
            res.sort()
            res.reverse()
            row[group] = res
    return features







