import numpy as np
from scipy.linalg import inv, det, eigh
from scipy.stats import multivariate_normal


class LinearDiscriminantAnalysis:
    """
    Análise de Discriminante Linear (LDA)
    
    Baseado em Bayesiano Gaussiano com matriz de covariância compartilhada entre classes.
    
    Regra de decisão:
        f(x) = argmax_i [ w_i^T * x + w_{i0} ]
    
    onde os pesos são calculados usando a covariância pooled (comum para todas as classes).
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.classes_ = None
        self.means_ = None
        self.cov_ = None
        self.priors_ = None
        self.weights_ = None
        self.intercepts_ = None
    
    def fit(self, X, y):
        """
        Treina o LDA estimando parâmetros.
        
        Parâmetros:
            X: array (n_samples, n_features)
            y: array (n_samples,) com labels das classes
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Estimar médias e priors para cada classe
        self.means_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[i, :] = X_c.mean(axis=0)
            self.priors_[i] = len(X_c) / n_samples
        
        # Estimar covariância pooled (compartilhada entre classes)
        self.cov_ = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            X_c_centered = X_c - self.means_[i, :]
            self.cov_ += X_c_centered.T @ X_c_centered
        
        self.cov_ /= (n_samples - n_classes)
        
        # Adicionar regularização para evitar singularidade
        self.cov_ += np.eye(n_features) * 1e-6
        
        # Calcular inversa da covariância
        cov_inv = inv(self.cov_)
        
        # Calcular pesos e interceptos para cada classe
        self.weights_ = np.zeros((n_classes, n_features))
        self.intercepts_ = np.zeros(n_classes)
        
        for i, c in enumerate(self.classes_):
            self.weights_[i, :] = self.means_[i, :] @ cov_inv
            self.intercepts_[i] = (
                -0.5 * (self.means_[i, :] @ cov_inv @ self.means_[i, :].T) +
                np.log(self.priors_[i])
            )
    
    def decision_function(self, X):
        """Calcula as funções discriminantes para cada classe."""
        return X @ self.weights_.T + self.intercepts_
    
    def predict_proba(self, X):
        """Retorna probabilidades a posteriori para cada classe."""
        scores = self.decision_function(X)
        # Normalizar usando softmax
        scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return scores_exp / scores_exp.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        """Prediz a classe para cada amostra."""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
