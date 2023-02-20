"""
Graph Mining - ALTEGRAD - Dec 2021
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

G = nx.readwrite.edgelist.read_edgelist('../datasets/CA-HepTh.txt', comments='#', delimiter='\t')

V = G.number_of_nodes()
E = G.number_of_edges()

print(f"The graph has {V} nodes and {E} edges.\n")



############## Task 2

print(f"The graph has {nx.number_connected_components(G)} connected components.")

lcc_set = max(nx.connected_components(G), key=len)
lcc = G.subgraph(lcc_set)
V_lcc = lcc.number_of_nodes()
E_lcc = lcc.number_of_edges()

print(f"The largest connected component has {V_lcc} nodes and {E_lcc} edges.")
print(f"It constitutes about {V_lcc/V:.2f} of nodes and {E_lcc/E:.2f} of edges.\n")



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

min_degree = np.min(degree_sequence)
max_degree = np.max(degree_sequence)
median_degree = np.median(degree_sequence)
mean_degree = np.mean(degree_sequence)

print(f"Statistics of degrees in the graph: min={min_degree}, max={max_degree}, median={median_degree}, mean={mean_degree:.2f}.\n")



############## Task 4

plt.plot(nx.degree_histogram(G))
plt.xlabel('degree'), plt.ylabel('frequency')
plt.grid(), plt.show()

plt.loglog(nx.degree_histogram(G))
plt.xlabel('degree'), plt.ylabel('frequency')
plt.grid(), plt.show();



############## Task 5

print(f"The global clustering coefficient of the graph is {nx.transitivity(G):.3f}.")
