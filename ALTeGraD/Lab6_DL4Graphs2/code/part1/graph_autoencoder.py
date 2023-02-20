"""
Deep Learning on Graphs - ALTEGRAD - Jan 2022
"""

import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score

from utils import normalize_adjacency, sparse_to_torch_sparse, loss_function
from models import GAE

import scipy.sparse as sp #

# Initialize device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
epochs = 200
n_hidden_1 = 16
n_hidden_2 = 32
learning_rate = 0.01
dropout_rate = 0.1

# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print(G.number_of_nodes())
print(G.number_of_edges())

n = G.number_of_nodes()

adj = nx.adjacency_matrix(G) # Obtains the adjacency matrix
adj_un = adj + sp.eye(adj.shape[0]) #
adj_un = sparse_to_torch_sparse(adj_un).to(device) #
adj = normalize_adjacency(adj) # Normalizes the adjacency matrix

features = np.random.randn(n, 10) # Generates node features

# Transforms the numpy matrices/vectors to torch tensors
features = torch.FloatTensor(features).to(device)
adj = sparse_to_torch_sparse(adj).to(device)

# Creates the model and specifies the optimizer
model = GAE(features.shape[1], n_hidden_1, n_hidden_2, dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Trains the model
for epoch in range(epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    z = model(features, adj)
    loss = loss_function(z, adj_un, device)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss.item()),
              'time: {:.4f}s'.format(time.time() - t))


# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

############## Task 4
    
##################
emb = z.detach().cpu().numpy()
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(emb)
##################

# Visualizes the nodes
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=y)
plt.title('PCA Visualization of the nodes')
plt.xlabel('1st dimension')
plt.ylabel('2nd dimension')
plt.show()

############## Task 5
    
##################
kmeans = KMeans(n_clusters=2)
y_pred = kmeans.fit_predict(emb)
print("Homogeneity score is:",homogeneity_score(y, y_pred))
##################
