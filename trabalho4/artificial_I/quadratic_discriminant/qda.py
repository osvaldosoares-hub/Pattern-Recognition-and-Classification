
import numpy as np
from scipy.linalg import inv


class QuadraticDiscriminantAnalysis:
   

    VALID_MODES = ('qda_full', 'qda_diagonal', 'qda_spherical',
                   'qda_reg', 'qda_pooled')

    def __init__(self, mode='qda_diagonal', reg_covar=1e-6, reg_strength=0.5):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode='{mode}' inválido. Escolha entre: {self.VALID_MODES}")
        self.mode         = mode
        self.reg_covar    = reg_covar
        self.reg_strength = reg_strength

        
        self.classes_     = None
        self.priors_      = None  
        self.means_       = None   
        self.covs_        = None   
        self.cov_pooled_  = None   

    def fit(self, X, y):
        
        self.classes_         = np.unique(y)
        n_samples, n_features = X.shape
        K                     = len(self.classes_)

        self.priors_ = np.array([np.sum(y == c) / n_samples
                                 for c in self.classes_])
        self.means_  = np.array([X[y == c].mean(axis=0)
                                 for c in self.classes_])

        
        S_pool = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            Xc = X[y == c] - self.means_[i]
            S_pool += Xc.T @ Xc
        S_pool /= (n_samples - K)
        self.cov_pooled_ = S_pool + np.eye(n_features) * self.reg_covar

        
        self.covs_ = np.zeros((K, n_features, n_features))
        for i, c in enumerate(self.classes_):
            Xc   = X[y == c] - self.means_[i]
            nc   = len(X[y == c])
            
            S_i  = (Xc.T @ Xc) / max(nc - 1, 1)

            

            if self.mode == 'lda':
              
                self.covs_[i] = self.cov_pooled_

            elif self.mode == 'qda_full':
               
                self.covs_[i] = S_i + np.eye(n_features) * self.reg_covar

            elif self.mode == 'qda_diagonal':
               
                S_diag = np.diag(np.diag(S_i))
                self.covs_[i] = S_diag + np.eye(n_features) * self.reg_covar

            elif self.mode == 'qda_spherical':
                
                sigma2 = np.diag(S_i).mean()
                self.covs_[i] = (sigma2 + self.reg_covar) * np.eye(n_features)

            elif self.mode == 'qda_reg':
               
                self.covs_[i] = S_i + np.eye(n_features) * self.reg_covar * 10

            elif self.mode == 'qda_pooled':
                
                alpha         = self.reg_strength
                S_interp      = alpha * S_i + (1 - alpha) * S_pool
                self.covs_[i] = S_interp + np.eye(n_features) * self.reg_covar

        return self

    
    def _log_likelihood(self, X, class_idx):
       
        mean = self.means_[class_idx]
        cov  = self.covs_[class_idx]
        p    = mean.shape[0]

        try:
            cov_inv          = inv(cov)
            _, logdet        = np.linalg.slogdet(cov)
        except np.linalg.LinAlgError:
            
            cov_inv = np.linalg.pinv(cov)
            logdet  = np.log(np.linalg.det(cov + np.eye(p) * 1e-10) + 1e-300)

        X_c      = X - mean
       
        mah      = np.sum(X_c @ cov_inv * X_c, axis=1)

        return -0.5 * logdet - 0.5 * mah

    def decision_function(self, X):
       
        K      = len(self.classes_)
        scores = np.zeros((X.shape[0], K))
        for i in range(K):
            scores[:, i] = (self._log_likelihood(X, i)
                            + np.log(self.priors_[i] + 1e-300))
        return scores

    def predict_proba(self, X):
        
        scores     = self.decision_function(X)
        
        scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return scores_exp / scores_exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        
        return np.mean(self.predict(X) == y)

    
    def __repr__(self):
        return (f"DiscriminantAnalysis(mode='{self.mode}', "
                f"reg_covar={self.reg_covar}, "
                f"reg_strength={self.reg_strength})")




# Classes de conveniência para modos específicos
class QDAFull(QuadraticDiscriminantAnalysis):
    """QDA com matrizes completas (Caso 1)."""
    def __init__(self, reg_covar=1e-6):
        super().__init__(mode='qda_full', reg_covar=reg_covar)


class LDAPooled(QuadraticDiscriminantAnalysis):
    """LDA com covariância pooled (Caso 2)."""
    def __init__(self, reg_covar=1e-6):
        super().__init__(mode='lda', reg_covar=reg_covar)


class QDADiagonal(QuadraticDiscriminantAnalysis):
    """QDA com matrizes diagonais (Caso 3)."""
    def __init__(self, reg_covar=1e-6):
        super().__init__(mode='qda_diagonal', reg_covar=reg_covar)


class QDASpherical(QuadraticDiscriminantAnalysis):
    """QDA com matrizes isotrópicas (Caso 4)."""
    def __init__(self, reg_covar=1e-6):
        super().__init__(mode='qda_spherical', reg_covar=reg_covar)


class QDARegularized(QuadraticDiscriminantAnalysis):
    """QDA com regularização adicional."""
    def __init__(self, reg_covar=1e-5):
        super().__init__(mode='qda_reg', reg_covar=reg_covar)


class QDAPooled(QuadraticDiscriminantAnalysis):
    """QDA com covariâncias combinadas (pooled)."""
    def __init__(self, reg_covar=1e-6, reg_strength=0.5):
        super().__init__(mode='qda_pooled',
                         reg_covar=reg_covar,
                         reg_strength=reg_strength)


# Para compatibilidade com o nome esperado
LinearDiscriminantAnalysis = LDAPooled


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               random_state=42, stratify=y)

    print(f"{'Modo':<20} {'Acurácia':>10}")
    print("-" * 32)
    
    # Testar diferentes modos
    classificadores = {
        'qda_full': QDAFull(),
        'lda': LDAPooled(),
        'qda_diagonal': QDADiagonal(),
        'qda_spherical': QDASpherical(),
        'qda_reg': QDARegularized(),
        'qda_pooled': QDAPooled(reg_strength=0.5)
    }
    
    for nome, clf in classificadores.items():
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        print(f"{nome:<20} {acc*100:>9.2f}%")