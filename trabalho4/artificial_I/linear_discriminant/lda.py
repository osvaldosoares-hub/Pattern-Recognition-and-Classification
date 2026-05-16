import numpy as np
from scipy.linalg import inv, det, eigh
from scipy.stats import multivariate_normal


class LinearDiscriminantAnalysis:
    
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.classes_ = None
        self.means_ = None
        self.cov_ = None
        self.priors_ = None
        self.weights_ = None
        self.intercepts_ = None
    
    def fit(self, X, y):
       
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        
        self.means_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[i, :] = X_c.mean(axis=0)
            self.priors_[i] = len(X_c) / n_samples
        
        
        self.cov_ = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            X_c_centered = X_c - self.means_[i, :]
            self.cov_ += X_c_centered.T @ X_c_centered
        
        self.cov_ /= (n_samples - n_classes)
        
        
        self.cov_ += np.eye(n_features) * 1e-6
        
        
        cov_inv = inv(self.cov_)
        
        
        self.weights_ = np.zeros((n_classes, n_features))
        self.intercepts_ = np.zeros(n_classes)
        
        for i, c in enumerate(self.classes_):
            self.weights_[i, :] = self.means_[i, :] @ cov_inv
            self.intercepts_[i] = (
                -0.5 * (self.means_[i, :] @ cov_inv @ self.means_[i, :].T) +
                np.log(self.priors_[i])
            )
    
    def decision_function(self, X):
        
        return X @ self.weights_.T + self.intercepts_
    
    def predict_proba(self, X):
        
        scores = self.decision_function(X)
        
        scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return scores_exp / scores_exp.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
