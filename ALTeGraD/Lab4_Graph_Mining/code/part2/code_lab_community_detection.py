"""
Graph Mining - ALTEGRAD - Dec 2021
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans

from scipy.sparse import csr_matrix, eye, diags
import networkx.algorithms.community as nx_comm



############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    A = csr_matrix(nx.adjacency_matrix(G))
    D_inv = diags([1/G.degree(node) for node in G.nodes()])
    L_rw = eye(G.number_of_nodes()) - D_inv @ A
    
    eigenvals, eigenvecs = eigs(L_rw, which='SR', k=k)
    eigenvecs = eigenvecs.real
    
    kmeans = KMeans(n_clusters=k).fit(eigenvecs)
    clustering = dict(zip(G.nodes(), kmeans.labels_))

    return clustering



############## Task 7

G = nx.readwrite.edgelist.read_edgelist('../datasets/CA-HepTh.txt', comments='#', delimiter='\t')
lcc_set = max(nx.connected_components(G), key=len)
lcc = G.subgraph(lcc_set)

lcc_clustering = spectral_clustering(lcc, 50)



############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    nb_clusters = max(clustering.values()) + 1
    clusters = [set() for _ in range(nb_clusters)]
    
    for node, cluster in clustering.items():
        clusters[cluster].add(node)

    return nx_comm.modularity(G, clusters)



############## Task 9

print(f"The modularity of spectral clustering performed on LCC is {modularity(lcc, lcc_clustering):.5f}.")

random_clustering = {}
for node in lcc.nodes():
    random_clustering[node] = randint(0, 49)

print(f"VS. modularity of random clustering performed on LCC is {modularity(lcc, random_clustering):.5f}.")