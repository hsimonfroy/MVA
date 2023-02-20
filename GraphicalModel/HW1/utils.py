import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
from matplotlib.patches import Ellipse
from scipy.linalg import eigh



def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def load_data(dataset_name, verbose=False):
    train_data = pd.read_csv('./data/train'+dataset_name, sep=" ", header=None)
    train_data = np.array(train_data)
    X_train = train_data[:,:2]
    y_train = train_data[:,2]

    test_data = pd.read_csv('./data/test'+dataset_name, sep=" ", header=None)
    test_data = np.array(test_data)
    X_test = test_data[:,:2]
    y_test = test_data[:,2]
    
    if verbose:
        print(f"dataset {dataset_name} loaded")
        print(f"feature dim: {X_train.shape[1:]}, train size: {X_train.shape[0]}, test size: {X_test.shape[0]}")
    return [X_train, y_train, X_test, y_test]



def plot_decision_function(classifier, dataset, title=None, ax=None, plot_legend=True):
    ax = ax or plt.gca()
    X_train, y_train, X_test, y_test = dataset

    colormap = 'bwr' 
    ax.scatter(X_train[:, 0], X_train[:, 1], s=20, c=y_train, alpha=0.9, cmap=colormap, zorder=3, label="train")
    ax.scatter(X_test[:, 0], X_test[:, 1], s=20, c=y_test, cmap=colormap, edgecolors="k", zorder=3, label="test")
    
    step = 500
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], step), np.linspace(ylim[0], ylim[1], step))
    z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    ax.contourf(xx, yy, z, alpha=0.7, cmap=colormap, zorder=2)
    ax.contour(xx, yy, z, [0.5], colors='k', zorder=2)

    ax.grid(), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_title(title)
    if plot_legend:
        ax.legend()



def plot_ellipse(mean, cov, ax, n_std=1., alpha=.2, facecolor=None):
    v, w = eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    ell = Ellipse(mean, v[0], v[1], 180.0 + angle, alpha=alpha, facecolor=facecolor, zorder=2)
    ax.add_patch(ell)