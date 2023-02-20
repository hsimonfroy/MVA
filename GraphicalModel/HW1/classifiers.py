import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import sigmoid

# ======= LINEAR DISCRIMINANT ANALYSIS CLASSIFIER =======
class LDA(BaseEstimator, ClassifierMixin):
    
    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        X0 = X[y == 0, :]
        X1 = X[y == 1, :]

        self.pi = np.mean(y)
        self.mu = np.array([np.mean(X0, axis=0), np.mean(X1, axis=0)])
        self.sigma = (X0 - self.mu[0]).T @ (X0 - self.mu[0]) + (X1 - self.mu[1]).T @ (X1 - self.mu[1])
        self.sigma /= self.n_samples

        inv_sigma = np.linalg.pinv(self.sigma)
        self.w = inv_sigma @ (self.mu[1] - self.mu[0])
        self.b = (self.mu[0].T@inv_sigma@self.mu[0] - self.mu[1].T@inv_sigma@self.mu[1]) / 2

        return self
    
    def decision_function(self, X):
        return sigmoid(X @ self.w + self.b + np.log( self.pi/(1-self.pi) ) )
    
    def predict_proba(self, X):
        probaC1 = self.decision_function(X)
        return np.array([1-probaC1, probaC1]).T
        
    def predict(self, X):
        return (self.decision_function(X) > 0.5).astype(float)
    
    

# ======= LOGISTIC REGRESSION CLASSIFIER =======
class LogisticRegression(BaseEstimator, ClassifierMixin):
    
    def fit(self, X, y, max_nb_iter = 100, eps = 1e-3):
        self.n_samples, self.n_features = X.shape        
        X = np.concatenate((np.ones((self.n_samples, 1)), X), axis=1)
        self.w = np.zeros(self.n_features + 1)
        
        nb_iter = 0
        criterion = np.inf
        while criterion > eps and nb_iter < max_nb_iter:
            
            # compute gradient and hessian
            sig_dot = sigmoid(X @ self.w)
            grad = X.T @ (sig_dot - y)
            hess = X.T * (sig_dot * (1-sig_dot)) @ X
            
            # compute newton step and criterion
            inv_hess_grad = np.linalg.solve(hess, grad)
            criterion = (grad @ inv_hess_grad) / 2

            # update
            self.w -= inv_hess_grad
            
        return self
    
    def decision_function(self, X):
        return sigmoid(X @ self.w[1:] + self.w[0])
    
    def predict_proba(self, X):
        probaC1 = self.decision_function(X)
        return np.array([1-probaC1, probaC1]).T
        
    def predict(self, X):
        return (self.decision_function(X) > 0.5).astype(float)
    
    
# ======= LINEAR REGRESSION CLASSIFIER =======  
class LinearRegression(BaseEstimator, ClassifierMixin):
    
    def fit(self, X, y, nb_max_iter = 100, eps = 1e-3):
        self.n_samples, self.n_features = X.shape        
        X = np.concatenate((np.ones((self.n_samples, 1)), X), axis=1)
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y
   
        return self
    
    def decision_function(self, X):
        return X @ self.w[1:] + self.w[0]
    
    def predict_proba(self, X):
        probaC1 = self.decision_function(X)
        return np.array([1-probaC1, probaC1]).T
        
    def predict(self, X):
        return (self.decision_function(X) > 0.5).astype(float)
    

    
    
    
    
