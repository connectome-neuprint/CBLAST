# CBLAST
Cluster neurons based on synaptic connectivity to derive cell types.

This application has numerous strategies for extracting features for different
neurons that can be used for automatic clustering.  Thees tools can be run offline
or within Jupyer notebooks to faciliate interactive exploration.

# installation and running

Install [conda](https://docs.conda.io/en/latest/miniconda.html).

    % conda create -n cblast
    % conda install scipy scikit-learn umap-learn jupyter hvplot bokeh plotly neuprint-python pyarrow -n typecluster -c conda-forge -c flyem-forge
    % source activate cblast 
    % git clone https://github.com/connectome-neuprint/cblast.git
    % cd cblast; python setup.py install
    % export NEUPRINT_APPLICATION_CREDENTIALS=YOURTOKEN
    % jupyter notebook 

To use the library, import cblast.

# todo

* add example notebooks
* metrics for outlier analysis
