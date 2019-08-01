"""Visualization functions.

Contains a set of functions for extracting features for a list of neurons.
"""

from bokeh.plotting import figure, output_notebook, show, gridplot, reset_output, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20 as palette
import itertools
import pandas as pd


def features_scatterplot2D(features, clusters=None, groundtruth=None, htmlfile=None, tsne=False):
    """Plot each neuron on a 2D scatter plot, which should reveal clusters.

    Args:
        features (dataframe): body id index, features...
        clusters (dataframe): body id index, "type"
        groundtruth (dataframe): body id index, "type"
        htmlfile (str): save data as html file instead of displaying in notebook
        tsne (boolean): use tsne instead of UMAP

    Note: features should be reduce using PCA first (ideally <500 features).
    Clusters and ground truth should have a column "type" and the same
    body id index.
    """
    from sklearn.manifold import TSNE

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
        clusters.rename(columns={"type": "typecluster"}, inplace=True)
        ctypes = clusters["typecluster"].unique()
        coordtable = coordtable.merge(clusters, on="bodyid", how="left")
        coordtable2 = coordtable2.merge(clusters, on="bodyid", how="left")

    for celltype in gtypes:
        cval = next(colors)
        p.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable.loc[coordtable["typegt"] == celltype]))
        p2.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable2.loc[coordtable2["typegt"] == celltype]))

    for celltype in ctypes:
        cval = next(colors)
        if groundtruth is None:
            p.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable.loc[coordtable["typecluster"] == celltype]))
            p2.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable2.loc[coordtable2["typecluster"] == celltype]))
        else:
            p3.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable.loc[coordtable["typecluster"] == celltype]))
            p4.square("x", "y", size=10, color=cval, alpha=0.8, legend=str(celltype), muted_color=cval, muted_alpha=0.1, source=ColumnDataSource(coordtable2.loc[coordtable2["typecluster"] == celltype]))

    if groundtruth is None and clusters is None:
        cval = next(colors)
        p.square("x", "y", size=10, color=cval, alpha=0.8, source=ColumnDataSource(coordtable))
        p2.square("x", "y", size=10, color=cval, alpha=0.8, source=ColumnDataSource(coordtable2))

    if groundtruth is not None and clusters is not None:
        p.add_tools( HoverTool(tooltips= [("body id","@bodyid"), ("type (gt)","@typegt"), ("type (cluster)","@typecluster")]))
        p.legend.click_policy="mute"
        p2.add_tools( HoverTool(tooltips= [("body id","@bodyid"),("type (gt)","@typegt"), ("type (cluster)","@typecluster")]))
        p2.legend.click_policy="mute"
        p3.add_tools( HoverTool(tooltips= [("body id","@bodyid"), ("type (gt)","@typegt"), ("type (cluster)","@typecluster")]))
        p3.legend.click_policy="mute"
        p4.add_tools( HoverTool(tooltips= [("body id","@bodyid"),("type (gt)","@typegt"), ("type (cluster)","@typecluster")]))
        p4.legend.click_policy="mute"
        #grid = gridplot([[p, p2], [p3, p4]])
        grid = gridplot([[p], [p3]])
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
            p.add_tools( HoverTool(tooltips= [("body id","@bodyid"), ("type (cluster)","@typecluster")]))
            p.legend.click_policy="mute"
            p2.add_tools( HoverTool(tooltips= [("body id","@bodyid"), ("type (cluster)","@typecluster")]))
            p2.legend.click_policy="mute"

        #grid = gridplot([[p, p2]])
        grid = gridplot([[p]])
        show(grid)
