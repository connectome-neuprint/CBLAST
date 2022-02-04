"""Visualization functions.

Contains a set of functions for extracting features for a list of neurons.
"""

from bokeh.plotting import figure, output_notebook, show, gridplot, row, reset_output, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20 as palette
import itertools
import pandas as pd
import hvplot.pandas
import plotly
import plotly.graph_objects as go
import numpy as np

def features_scatterplot2D(features, clusters=None, groundtruth=None, htmlfile=None, tsne=False):
    """Plot each neuron on a 2D scatter plot, which should reveal clusters.

    Args:
        features (dataframe): body id index, features...
        clusters (dataframe): body id index, "type" -- produced from get_partitions
        groundtruth (dataframe): body id index, "type"
        htmlfile (str): save data as html file instead of displaying in notebook
        tsne (boolean): use tsne instead of UMAP

    Note: features should be reduce using PCA first (ideally <500 features).
    Clusters and ground truth should have a column "type" and the same
    body id index.
    """
    from sklearn.manifold import TSNE
    
    import warnings
    warnings.filterwarnings("ignore")

    # set output destination
    reset_output()
    if htmlfile is None:
        output_notebook()
    else:
        output_file(htmlfile)

    xsize, ysize = features.shape
    dembed = dembed_v2 = features.values
    if ysize > 2: # only run tsne if a reduction is needed
        if tsne:
            # try two different TSNE thresholds
            dembed = TSNE(n_components=2, perplexity=30).fit_transform(features)
            dembed_v2 = TSNE(n_components=2, perplexity=10).fit_transform(features)
        else:
            import umap
            reducer = umap.UMAP()
            dembed = reducer.fit_transform(features)
            reducer2 = umap.UMAP()
            dembed_v2 = reducer2.fit_transform(features)


    #colors has a list of colors which can be used in plots
    colors = itertools.cycle(palette[20])

    if groundtruth is not None and clusters is not None:
        p = figure(title="Ground truth(v1)", plot_width=800, plot_height=800, tools="pan,wheel_zoom,reset,save", active_scroll="wheel_zoom")
        p2 = figure(title="Ground truth(v2)", plot_width=800, plot_height=800, tools="pan,wheel_zoom,reset,save", active_scroll="wheel_zoom")
        p3 = figure(title="Clusters (v1)", plot_width=800, plot_height=800, tools="pan,wheel_zoom,reset,save", active_scroll="wheel_zoom")
        p4 = figure(title="Clusters (v2)", plot_width=800, plot_height=800, tools="pan,wheel_zoom,reset,save", active_scroll="wheel_zoom")
    elif groundtruth is not None or clusters is not None:
        if groundtruth is None:
            p = figure(title="Clusters (v1)", plot_width=800, plot_height=800, tools="pan,wheel_zoom,reset,save", active_scroll="wheel_zoom")
            p2 = figure(title="Clusters (v2)", plot_width=800, plot_height=800, tools="pan,wheel_zoom,reset,save", active_scroll="wheel_zoom")
        else:
            p = figure(title="Ground truth (v1)", plot_width=800, plot_height=800, tools="pan,wheel_zoom,reset,save", active_scroll="wheel_zoom")
            p2 = figure(title="Ground truth (v2)", plot_width=800, plot_height=800, tools="pan,wheel_zoom,reset,save", active_scroll="wheel_zoom")
    else:
            p = figure(title="Features (v1)", plot_width=800, plot_height=800, tools="pan,wheel_zoom,reset,save", active_scroll="wheel_zoom")
            p2 = figure(title="Features (v2)", plot_width=800, plot_height=800, tools="pan,wheel_zoom,reset,save", active_scroll="wheel_zoom")

    # create bokeh data sources
    coordtable = pd.DataFrame(data={"bodyid": features.index.values, "x": dembed[:,0], "y": dembed[:,1]})
    coordtable2 = pd.DataFrame(data={"bodyid": features.index.values, "x": dembed_v2[:,0], "y": dembed_v2[:,1]})

    gtypes = []
    ctypes = []
    # load gt types
    if groundtruth is not None:
        gt = groundtruth[["bodyid", "type"]].copy()
        gt.rename(columns={"type": "typegt"}, inplace=True)
        gtypes = gt["typegt"].unique()
        coordtable = coordtable.merge(gt, on="bodyid", how="left")
        coordtable2 = coordtable2.merge(gt, on="bodyid", how="left")


    # load cluster types
    if clusters is not None:
        clusters = clusters[["bodyid", "type"]].copy()
        clusters.rename(columns={"type": "cblast"}, inplace=True)
        ctypes = clusters["cblast"].unique()
        coordtable = coordtable.merge(clusters, on="bodyid", how="left")
        coordtable2 = coordtable2.merge(clusters, on="bodyid", how="left")

    for celltype in gtypes:
        cval = next(colors)
        p.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable.loc[coordtable["typegt"] == celltype]))
        p2.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable2.loc[coordtable2["typegt"] == celltype]))

    for celltype in ctypes:
        cval = next(colors)
        if groundtruth is None:
            p.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable.loc[coordtable["cblast"] == celltype]))
            p2.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable2.loc[coordtable2["cblast"] == celltype]))
        else:
            p3.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable.loc[coordtable["cblast"] == celltype]))
            p4.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable2.loc[coordtable2["cblast"] == celltype]))

    if groundtruth is None and clusters is None:
        cval = next(colors)
        p.square("x", "y", size=10, color=cval, alpha=0.8, source=ColumnDataSource(coordtable))
        p2.square("x", "y", size=10, color=cval, alpha=0.8, source=ColumnDataSource(coordtable2))

    if groundtruth is not None and clusters is not None:
        p.add_tools( HoverTool(tooltips= [("body id","@bodyid"), ("type (gt)","@typegt"), ("type (cluster)","@cblast")]))
        p.legend.click_policy="mute"
        p2.add_tools( HoverTool(tooltips= [("body id","@bodyid"),("type (gt)","@typegt"), ("type (cluster)","@cblast")]))
        p2.legend.click_policy="mute"
        p3.add_tools( HoverTool(tooltips= [("body id","@bodyid"), ("type (gt)","@typegt"), ("type (cluster)","@cblast")]))
        p3.legend.click_policy="mute"
        p4.add_tools( HoverTool(tooltips= [("body id","@bodyid"),("type (gt)","@typegt"), ("type (cluster)","@cblast")]))
        p4.legend.click_policy="mute"
        # grid = gridplot([[p, p2], [p3, p4]])
        # grid = gridplot([[p], [p3]])
        grid = row(p, p3)
        show(grid)
    elif groundtruth is None and clusters is None:
        #grid = gridplot([[p, p2]])
        p.add_tools( HoverTool(tooltips= [("body id","@bodyid")]))
        grid = gridplot([[p]])
        show(grid)
    else:
        if groundtruth is not None:
            p.add_tools( HoverTool(tooltips= [("body id","@bodyid"), ("type (gt)","@typegt")]))
            p.legend.click_policy="mute"
            p2.add_tools( HoverTool(tooltips= [("body id","@bodyid"),("type (gt)","@typegt")]))
            p2.legend.click_policy="mute"
        else:
            p.add_tools( HoverTool(tooltips= [("body id","@bodyid"), ("type (cluster)","@cblast")]))
            p.legend.click_policy="mute"
            p2.add_tools( HoverTool(tooltips= [("body id","@bodyid"), ("type (cluster)","@cblast")]))
            p2.legend.click_policy="mute"

        #grid = gridplot([[p, p2]])
        grid = gridplot([[p]])
        show(grid)


def features_compareheatmap(features):
    """Show heatmap for the provided features (such as filtered from
    "features.find_max_differences")

    Arg:
        features (dframe): neuron features
    Returns:
        heatmap showing the features

    Note: should restrict to only a small number of features that have not been reduced 
    """

    featuresx = features.copy()

    from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar

    featuresx.index.name = "bodyid"
    featuresx.columns.name = "feature"

    featuresx.index = featuresx.index.map(str)
    bodyid = featuresx.index.values.tolist()
    feature = featuresx.columns.values.tolist()

    # reshape to 1D array or rates with a month and year for each row.
    df = pd.DataFrame(featuresx.stack(), columns=['score']).reset_index()
    return df.hvplot.heatmap(x="feature", y="bodyid", C="score", colorbar=True, width=900, height=400)


def connection_differences(npclient, neuronlist):
    """Show top inputs and outputs for set of neurons.

    Args:
        npclient (object): neuprint client object
        neuronlist (list): list of body ids
    Returns:
        (dataframe, dataframe): 

    """
    
    # fetch all connections from this cell type
    query = f"MATCH (n :Neuron)-[x :ConnectsTo]-(m) WHERE n.bodyId in {neuronlist} RETURN n.bodyId AS bodyId, n.instance AS instance, x.weight AS weight, m.bodyId AS bodyId2, m.type AS type2, (startNode(x) = n) as isOutput, n.status AS body1status, m.status AS body2status, m.cropped AS iscropped2, n.cropped AS iscropped1"
    connections = npclient.fetch_custom(query)

    minweight = 3 # ignore connections for canonical inputs or outputs that are below this
    celltype_lists_inputs = {}
    celltype_lists_outputs = {}
    for idx, row in connections.iterrows():
        bodyid = row["bodyId"]
        type_status = row["body1status"]
        type_status2 = row["body2status"] 
        is_output = row["isOutput"]
        is_cropped1 = row["iscropped1"]
        is_cropped2 = row["iscropped2"]

        # might as well ignore connection as well if not to traced
        if type_status2 != "Traced":
            continue

        conntype = row["type2"]
        hastype = True
        if conntype is None or conntype == "":
            conntype = str(row["bodyId2"])
            hastype = False

        # don't consider the edge for something that is leaves and has not type
        if not hastype and is_cropped2:
            continue
            
        # don't consider a weak edge
        if row["weight"] < minweight:
            continue
            
        if is_output:       
            if bodyid not in celltype_lists_outputs:
                celltype_lists_outputs[bodyid] = []

            # add body id in the middle to allow sorting
            celltype_lists_outputs[bodyid].append((row["weight"], row["bodyId2"], {"partner": conntype, "weight": row["weight"], "hastype": hastype, "important": False}))
        else:       
            if bodyid not in celltype_lists_inputs:
                celltype_lists_inputs[bodyid] = []

            # add body id in the middle to allow sorting
            celltype_lists_inputs[bodyid].append((row["weight"], row["bodyId2"], {"partner": conntype, "weight": row["weight"], "hastype": hastype, "important": False}))

    importance_cutoff = 0.25 # ignore connections after the top 25% (with some error margin)
    
    # sort each list and cut-off anything below 25% with some error bar  
    def sort_lists(cell_type_lists):  
        for bid, info in cell_type_lists.items():
            important_count = 0
            info.sort()
            info.reverse()
            total = 0
            for (weight, ignore, val) in info:
                total += weight

            count = 0
            threshold = 0
            for (weight, ignore, val) in info:
                # ignore connections below thresholds (but take at least 5 inputs and outputs)
                if weight < threshold and important_count > 5:
                    break
                important_count += 1
                val["important"] = True

                count += weight
                # disable marking similar strength neurons as important
                # since it doesn't matter for the feature table
                if threshold == 0 and count > (total*importance_cutoff):
                    threshold = weight # - weight**(1/2)
    sort_lists(celltype_lists_inputs)
    sort_lists(celltype_lists_outputs)

    def _generate_feature_table(celltype_lists):
        # add shared dict to a globally sorted list from each partner
        global_queue = []
        for neuron in neuronlist:
            if neuron in celltype_lists:
                for (weight, bid2, val) in celltype_lists[neuron]:
                    val["examined"] = False
                    if val["important"]:
                        global_queue.append((weight, bid2, neuron, val))
        global_queue.sort()
        global_queue.reverse()
        
        # provide rank from big to small and provide connection count
        # (count delineated by max in row but useful for bookkeeping)
        celltype_rank = []
        
        # from large to small make a list of common partners, each time another is added, iterate through
        # other common partners to find a match (though we need to look at the whole list beyond the cut-off)
        for (count, ignore, neuron, entry) in global_queue:
            if entry["examined"]:
                continue
            celltype_rank.append((entry["partner"], count, entry["hastype"]))
        
            # check for matches for each cell type instance in neuron working set
            for neuron in neuronlist:
                if neuron in celltype_lists:
                    for (weight, ignore, val) in celltype_lists[neuron]:
                        if not val["examined"] and val["partner"] == entry["partner"] and val["hastype"] == entry["hastype"]:
                            val["examined"] = True
                            break
                        
        # generate feature map for neuron
        features = np.zeros((len(neuronlist), len(celltype_rank)))
        
        iter1 = 0
        for neuron in neuronlist:
            connlist = []
            # it is possible that a neuron has no input or outputs
            if neuron in celltype_lists:
                connlist = celltype_lists[neuron]
            match_list = {}
            for (weight, ignore, val) in connlist:
                matchkey = (val["partner"], val["hastype"])
                if matchkey not in match_list:
                    match_list[matchkey] = []
                match_list[matchkey].append(weight)
            cfeatures = []
            for (ctype, count, hastype) in celltype_rank:
                rank_match = (ctype, hastype)
                if rank_match in match_list:
                    cfeatures.append(match_list[rank_match][0])
                    del match_list[rank_match][0]
                    if len(match_list[rank_match]) == 0:
                        del match_list[rank_match]
                else:
                    cfeatures.append(0)
            features[iter1] = cfeatures
            iter1 +=1
                
        ranked_names = []
        for (ctype, count, hastype) in celltype_rank:
            ranked_names.append(ctype)
            
        return pd.DataFrame(features, index=neuronlist, columns=ranked_names)
    features_inputs = _generate_feature_table(celltype_lists_inputs)
    features_outputs = _generate_feature_table(celltype_lists_outputs)
    
    # TODO add heatmap viz that shows +/- % from median

    #features_inputs = features_inputs[features_inputs.median().sort_values(ascending=False).index]
    #features_outputs = features_outputs[features_outputs.median().sort_values(ascending=False).index]


    imed = features_inputs.var()   
    omed = features_outputs.var()
    i_order = np.argsort(imed)[::-1]
    o_order = np.argsort(omed)[::-1]

    features_inputs = features_inputs.iloc[:, i_order]
    features_outputs = features_outputs.iloc[:, o_order]

    features_inputs.loc["median"] = features_inputs.median()
    features_outputs.loc["median"] = features_outputs.median()

    return features_inputs, features_outputs


def features_radarchart(features, clusters=None, htmlfile=None):
    """Show radar chart for the provided features.

    Arg:
        features (dframe): neuron features
        clusterinfo (dframe): 
    Returns:
        heatmap showing the features

    Note: should call on non-pca features but should restrict to most important features.
    """
    fig = go.Figure()
    
    if clusters is None:
        categories = list(features.columns)
        
        for index, row in features.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row.values,
                theta=categories,
                fill='toself',
                text=index,
                name=index
            ))
    else:
        clusters = clusters.set_index('bodyid')

        featuresx = pd.concat([features, clusters], axis=1)
        featuresx = featuresx.sort_values(by=['type'])
        
        categories = list(featuresx.columns)
        categories.remove('type')
        
        for index, row in featuresx.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row.values,
                theta=categories,
                fill='toself',
                text=index,
                name=row['type']
            ))
    
    fig.update_traces(
        hoverinfo='text')
    fig.update_layout(
        width = 900, 
        height = 800,
    polar=dict(
        radialaxis=dict(visible=True)
        ),
    showlegend=True
    )
    
    if htmlfile is not None:
        plotly.offline.plot(fig, filename=htmlfile)
    fig.show()
