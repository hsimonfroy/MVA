"""
Graph Mining - ALTEGRAD - Dec 2021
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


############## Task 10
# Generate simple dataset
def create_dataset():
    cycles = [nx.cycle_graph(n) for n in range(3,103)]
    paths = [nx.path_graph(n) for n in range(3,103)]
    Gs = cycles + paths
    y = [0 for _ in range(100)] + [1 for _ in range(100)]

    ##################
    # your code here #
    ##################

    return Gs, y


Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)

# Compute the shortest path kernel
def shortest_path_kernel(G_train, G_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(G_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(G_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(G_train), len(all_paths)))
    for i in range(len(G_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(G_test), len(all_paths)))
    for i in range(len(G_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 11
# Compute the graphlet kernel
def graphlet_kernel(G_train, G_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    nb_graphlets = len(graphlets)
    
    phi_train = np.zeros((len(G_train), nb_graphlets))
    
    ##################
    # your code here #
    ##################

    for i, G in enumerate(G_train):
        nodes = list(G.nodes())
        for _ in range(n_samples):
            s = np.random.choice(nodes, size=3, replace=False)
            subG = G.subgraph(s)
            for j in range(nb_graphlets):
                if nx.is_isomorphic(subG, graphlets[j]):
                    phi_train[i,j] += 1

    phi_test = np.zeros((len(G_test), nb_graphlets))
    
    ##################
    # your code here #
    ##################
    
    for i, G in enumerate(G_test):
        nodes = list(G.nodes())
        for _ in range(n_samples):
            s = np.random.choice(nodes, size=3, replace=False)
            subG = G.subgraph(s)
            for j in range(nb_graphlets):
                if nx.is_isomorphic(subG, graphlets[j]):
                    phi_train[i,j] += 1

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)



############## Task 12

##################
# your code here #
##################

K_train_gl, K_test_gl = graphlet_kernel(G_train, G_test, n_samples=200)



############## Task 13

##################
# your code here #
##################

# Initialize SVMs and train
clf_sp = SVC(kernel="precomputed")
clf_sp.fit(K_train_sp, y_train)

clf_gl = SVC(kernel="precomputed")
clf_gl.fit(K_train_gl, y_train)

# Predict
y_pred_sp = clf_sp.predict(K_test_sp)
acc_sp = accuracy_score(y_test, y_pred_sp)

y_pred_gl = clf_sp.predict(K_test_gl)
acc_gl = accuracy_score(y_test, y_pred_gl)

print(f"The classification accuracy of shortest path kernel model is {acc_sp}.")
print(f"VS. classification accuracy of graphlet kernel model is {acc_gl}.")

