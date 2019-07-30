# typecluster
**status: experimental**
Cluster neurons based on synapse connections.

This application has numerous strategies for extracting features for different
neurons that can be used for automatic clustering.  These tools, for now, are intended
to run within a Jupyter notebook and to enable users to find new neuron clusters or
to evaluate previously create ones.  It is also hoped that interactive exploration
will aid the in creation of even better features.

# installation and running

Install conda.

    % conda create -n typecluster
    % conda install scipy scikit-learn jupyter hvplot bokeh neuprint-python -n typecluster -c conda-forge -c flyem-forge
    % source activate typecluster
    % python setup.py install
    % export NEUPRINT_APPLICATION_CREDENTIALS=YOURTOKEN
    % jupyter notebook 

To use the library, import typecluster.

# todo

* add more visualization widgets (especially to debug cluster features and to evalaute ground truth)
* create recipes for example workflows
* add example notebooks
* add functions for outlier analysis
* improve featues creation algorithms (e.g., iterative feature generation)
* create batch-based workflow to run with minimal supervision
