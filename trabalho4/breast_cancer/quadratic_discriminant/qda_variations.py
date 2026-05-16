import numpy as np
from scipy.linalg import inv, det


class DiscriminantAnalysisWithVariations:
    """
    Classificador baseado em Bayesiano Gaussiano com múltiplas estratégias
    de covariância para investigar diferentes situações.
    
    Permite testar:
    1. QDA Full - Σ_i para cada classe (covariância não compartilhada)
    2. LDA - Σ compartilhada para todas as classes (caso especial de QDA)
    3. QDA com regularização - Σ_i + λI
    4. QDA com covariância pooled ponderada
    5. Outras variações conforme necessário
    """
    
    def __init__(self, mode='qda_full', reg_covar=1e-6, reg_strength=0.5):
        """
        Parâmetros:
            mode: 'qda_full' | 'lda' | 'qda_reg' | 'qda_pooled'
            reg_covar: Regularização Ridge (λ)
            reg_strength: Força de regularização para modo pooled (0=LDA, 1=QDA)
        """
        self.mode = mode
        self.reg_covar = reg_covar
        self.reg_strength = reg_strength
        self.classes_ = None
        self.means_ = None
        self.covs_ = None
        self.priors_ = None
        self.cov_pooled_ = None
    
    def fit(self, X, y):
        """
        Treina o classificador com a estratégia de covariância especificada.
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Estimar priors
        self.priors_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            self.priors_[i] = sum(y == c) / n_samples
        
        # Estimar médias
        self.means_ = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes_):
            self.means_[i] = X[y == c].mean(axis=0)
        
        # Estratégia 1: QDA Full - covariância específica por classe
        if self.mode == 'qda_full':
            self.covs_ = np.zeros((n_classes, n_features, n_features))
            for i, c in enumerate(self.classes_):
                X_c = X[y == c]
                X_c_centered = X_c - self.means_[i]
                cov = (X_c_centered.T @ X_c_centered) / (len(X_c) - 1)
                cov += np.eye(n_features) * self.reg_covar
                self.covs_[i] = cov
        
        # Estratégia 2: LDA - covariância pooled (compartilhada)
        elif self.mode == 'lda':
            self.cov_pooled_ = np.zeros((n_features, n_features))
            for i, c in enumerate(self.classes_):
                X_c = X[y == c]
                X_c_centered = X_c - self.means_[i]
                self.cov_pooled_ += X_c_centered.T @ X_c_centered
            self.cov_pooled_ /= (n_samples - n_classes)
            self.cov_pooled_ += np.eye(n_features) * self.reg_covar
            
            # Copiar para compatibilidade
            self.covs_ = np.zeros((n_classes, n_features, n_features))
            for i in range(n_classes):
                self.covs_[i] = self.cov_pooled_.copy()
        
        # Estratégia 3: QDA com regularização adicional
        elif self.mode == 'qda_reg':
            self.covs_ = np.zeros((n_classes, n_features, n_features))
            for i, c in enumerate(self.classes_):
                X_c = X[y == c]
                X_c_centered = X_c - self.means_[i]
                cov = (X_c_centered.T @ X_c_centered) / (len(X_c) - 1)
                # Regularização mais agressiva
                cov += np.eye(n_features) * self.reg_covar * 10
                self.covs_[i] = cov
        
        # Estratégia 4: QDA com covariância pooled interpolada
        # Combina covariância por classe com pooled: Σ_i = α*Σ_i + (1-α)*Σ_pool
        elif self.mode == 'qda_pooled':
            # Calcular covariância pooled
            cov_pooled = np.zeros((n_features, n_features))
            for i, c in enumerate(self.classes_):
                X_c = X[y == c]
                X_c_centered = X_c - self.means_[i]
                cov_pooled += X_c_centered.T @ X_c_centered
            cov_pooled /= (n_samples - n_classes)
            
            # Calcular covariâncias por classe e interpolar
            self.covs_ = np.zeros((n_classes, n_features, n_features))
            for i, c in enumerate(self.classes_):
                X_c = X[y == c]
                X_c_centered = X_c - self.means_[i]
                cov_i = (X_c_centered.T @ X_c_centered) / (len(X_c) - 1)
                
                # Interpolação: λ=0 -> LDA (pooled), λ=1 -> QDA puro
                # self.reg_strength controla o quanto de cada
                cov = (self.reg_strength * cov_i + 
                       (1 - self.reg_strength) * cov_pooled)
                
                cov += np.eye(n_features) * self.reg_covar
                self.covs_[i] = cov
        
        else:
            raise ValueError(f"Modo desconhecido: {self.mode}")
    
    def _log_likelihood(self, X, class_idx):
        """
        Calcula log-verossimilhança Gaussiana para uma classe.
        
        log p(x|ω_i) = -½log|Σ_i| - ½(x-μ_i)^T Σ_i^(-1)(x-μ_i)
        """
        mean = self.means_[class_idx]
        cov = self.covs_[class_idx]
        n_features = mean.shape[0]
        
        try:
            cov_inv = inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
            logdet = np.log(np.linalg.det(cov + np.eye(n_features) * 1e-10))
        
        X_centered = X - mean
        mahal_dist = np.sum(X_centered @ cov_inv * X_centered, axis=1)
        
        log_likelihood = -0.5 * logdet - 0.5 * mahal_dist - 0.5 * n_features * np.log(2 * np.pi)
        
        return log_likelihood
    
    def decision_function(self, X):
        """Calcula as funções discriminantes para cada classe."""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        scores = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            scores[:, i] = self._log_likelihood(X, i) + np.log(self.priors_[i])
        
        return scores
    
    def predict_proba(self, X):
        """Retorna probabilidades a posteriori para cada classe."""
        scores = self.decision_function(X)
        scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return scores_exp / scores_exp.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        """Prediz a classe para cada amostra."""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


# Aliases para compatibilidade
class LinearDiscriminantAnalysis(DiscriminantAnalysisWithVariations):
    """LDA - Covariância compartilhada (Σ igual para todas as classes)."""
    def __init__(self, reg_covar=1e-6):
        super().__init__(mode='lda', reg_covar=reg_covar)


class QuadraticDiscriminantAnalysis(DiscriminantAnalysisWithVariations):
    """QDA Full - Covariância específica por classe (Σ_i)."""
    def __init__(self, reg_covar=1e-6):
        super().__init__(mode='qda_full', reg_covar=reg_covar)


class QuadraticDiscriminantAnalysisRegularized(DiscriminantAnalysisWithVariations):
    """QDA com regularização adicional."""
    def __init__(self, reg_covar=1e-5):
        super().__init__(mode='qda_reg', reg_covar=reg_covar)


class QuadraticDiscriminantAnalysisPooled(DiscriminantAnalysisWithVariations):
    """QDA interpolado com covariância pooled."""
    def __init__(self, reg_covar=1e-6, reg_strength=0.5):
        super().__init__(mode='qda_pooled', reg_covar=reg_covar, reg_strength=reg_strength)
