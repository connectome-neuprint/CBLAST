"""Utility functions.
"""

def extract_faninfanout(npclient, dataset, neuronlist, minconn=3):
    """Find all neurons in fanin or fanout of list of neurons.

    Note: return list includes neuronlist

    Args:
        npclient (object): neuprint client object
        dataset (str): name of neuprint dataset
        neuronlist (list): list of body ids
        minconn (int): Only consider connections >= minconn
    Returns:
        list: list of body ids
    """

    bodyset = set(neuronlist)

    for iter1 in range(0, len(neuronlist), 100):
        currlist = neuronlist[iter1:(iter1+100)]
        partnerquery = f"WITH {currlist} AS TARGETS MATCH (n :Neuron)-[x :ConnectsTo]-(m :Neuron) WHERE n.bodyId in TARGETS AND x.weight >= {minconn} AND m.status=\"Traced\" DISTINCT m AS m RETURN collect(m.bodyId) AS bodylist"

        res = npclient.fetch_custom(partnerquery, dataset=dataset)
        if len(res) == 1:
            bodyset = bodyset.union(set(res["bodylist"][0]))


    return list(bodyset)
    
