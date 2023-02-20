"""
Graph-based Recommendations - ALTEGRAD - Jan 2022
"""

import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import SR_GNN
from utils import load_dataset, generate_batches

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Loads training and test data
sessions_train, sessions_test, y_train, y_test, max_item_id = load_dataset()

# Generates batches
adj_train, items_train, last_item_train, idx_train, targets_train = generate_batches(sessions_train, y_train, batch_size=256, device=device)
adj_test, items_test, last_item_test, idx_test, targets_test = generate_batches(sessions_test, y_test, batch_size=256, device=device)

# Hyperparameters
epochs = 30
hidden_dim = 64
dropout = 0.0
learning_rate = 0.001

model = SR_GNN(max_item_id+1, hidden_dim, dropout, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
loss_function = nn.CrossEntropyLoss()

# Trains and evaluates the SR-GNN model
start = time.time()
best_results = {'hit': {'val':0, 'epoch': 0}, 'mrr': {'val':0, 'epoch': 0}}
for epoch in range(epochs):
    t = time.time()
    model.train()
    train_loss = 0.0
    count = 0
    n_train_batches = len(adj_train)
    for i in range(n_train_batches):
        optimizer.zero_grad()
        scores = model(adj_train[i], items_train[i], last_item_train[i], idx_train[i])
        loss = loss_function(scores, targets_train[i]-1)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * scores.size(0)
        count += scores.size(0)
            
    print('Epoch: {:02d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(train_loss / count),
          'time: {:.4f}s'.format(time.time() - t))
    
    model.eval()
    hit, mrr = [], []
    n_test_batches = len(adj_test)
    for i in range(n_test_batches):
        scores = model(adj_test[i], items_test[i], last_item_test[i], idx_test[i])
        sub_scores = scores.topk(20)[1]
        sub_scores = sub_scores.detach().cpu().numpy()
        for score, target in zip(sub_scores, targets_test[i]):
            target = target.cpu().numpy()
            hit.append(np.isin(target-1, score))
            if len(np.where(score == target-1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    
    hit = np.mean(hit)*100
    if hit >= best_results['hit']['val']:
        best_results['hit']['val'] = hit
        best_results['hit']['epoch'] = epoch
   
    mrr = np.mean(mrr)*100
    if mrr >= best_results['mrr']['val']:
        best_results['mrr']['val'] = mrr
        best_results['mrr']['epoch'] = epoch
        
    print('Best Results on Test Set:')
    print('Recall@20: {:.4f}'.format(best_results['hit']['val']),
          'MMR@20: {:.4f}'.format(best_results['mrr']['val']))
    print('-------------------------------------------------------')
    print()