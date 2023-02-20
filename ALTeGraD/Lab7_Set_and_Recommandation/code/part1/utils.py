"""
Learning on Sets - ALTEGRAD - Jan 2022
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    X_train = np.zeros((n_train, max_train_card))
    for i in range(n_train):
        card  = np.random.randint(1,11)
        X_train[i, -card:] = np.random.randint(1,11, size=card)
    y_train = X_train.sum(axis=1)
    ##################

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    n_test = 200000
    min_test_card  = 5
    step_test_card = 5
    max_test_card = 100
    
    cards = range(min_test_card, max_test_card+1, step_test_card)
    nb_samp_per_card = n_test // len(cards)
    
    X_test, y_test = [], []
    for card in cards:
        X_test.append(np.random.randint(1,11,size=(nb_samp_per_card, card)))
        y_test.append(np.sum(X_test[-1], axis=1))
    ##################

    return X_test, y_test




