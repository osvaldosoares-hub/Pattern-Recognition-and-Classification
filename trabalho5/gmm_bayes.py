import numpy as np
from sklearn.cluster import KMeans
from scipy.special import logsumexp


class GaussianMixtureEM:


    VALID_COV_TYPES = ('full', 'tied', 'diag', 'spherical')

    def __init__(
        self,
        n_components=3,
        covariance_type='full',
        reg_covar=1e-6,
        random_state=42,
        max_iter=200,
        tol=1e-4,
        init_params='kmeans'
    ):
        if covariance_type not in self.VALID_COV_TYPES:
            raise ValueError(f"covariance_type deve ser um de {self.VALID_COV_TYPES}")

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.init_params = init_params

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -np.inf

    @staticmethod
    def _safe_log(x):
        return np.log(np.clip(x, 1e-300, None))

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        if self.init_params == 'kmeans' and n_samples >= self.n_components:
            kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            means = kmeans.cluster_centers_
            counts = np.bincount(labels, minlength=self.n_components).astype(float)
            weights = counts / np.maximum(n_samples, 1)
            weights = np.clip(weights, 1e-12, None)
            weights /= weights.sum()
        else:
            idx = rng.choice(n_samples, self.n_components, replace=n_samples < self.n_components)
            means = X[idx].copy()
            weights = np.ones(self.n_components) / self.n_components

        # covariâncias iniciais
        emp_cov = np.cov(X.T) if n_samples > 1 else np.eye(n_features)
        if emp_cov.ndim == 0:
            emp_cov = np.array([[float(emp_cov)]])
        emp_cov = emp_cov + self.reg_covar * np.eye(n_features)

        if self.covariance_type == 'full':
            covariances = np.array([emp_cov.copy() for _ in range(self.n_components)])
        elif self.covariance_type == 'diag':
            d = np.diag(emp_cov).copy()
            d = np.maximum(d, self.reg_covar)
            covariances = np.array([d.copy() for _ in range(self.n_components)])
        elif self.covariance_type == 'spherical':
            v = float(np.mean(np.diag(emp_cov)))
            v = max(v, self.reg_covar)
            covariances = np.full(self.n_components, v, dtype=float)
        else:  # tied
            covariances = emp_cov.copy()

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances

    def _estimate_log_gaussian_prob(self, X):
        n_samples, n_features = X.shape
        K = self.n_components
        log_prob = np.empty((n_samples, K))

        if self.covariance_type == 'full':
            for k in range(K):
                cov = self.covariances_[k] + self.reg_covar * np.eye(n_features)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    cov = cov + 1e-6 * np.eye(n_features)
                    _, logdet = np.linalg.slogdet(cov)
                inv_cov = np.linalg.inv(cov)
                diff = X - self.means_[k]
                maha = np.sum((diff @ inv_cov) * diff, axis=1)
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + logdet + maha)

        elif self.covariance_type == 'diag':
            for k in range(K):
                var = np.maximum(self.covariances_[k], self.reg_covar)
                diff = X - self.means_[k]
                maha = np.sum((diff ** 2) / var, axis=1)
                logdet = np.sum(np.log(var))
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + logdet + maha)

        elif self.covariance_type == 'spherical':
            for k in range(K):
                var = max(float(self.covariances_[k]), self.reg_covar)
                diff = X - self.means_[k]
                maha = np.sum(diff ** 2, axis=1) / var
                logdet = n_features * np.log(var)
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + logdet + maha)

        else:  # tied
            cov = self.covariances_ + self.reg_covar * np.eye(n_features)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                cov = cov + 1e-6 * np.eye(n_features)
                _, logdet = np.linalg.slogdet(cov)
            inv_cov = np.linalg.inv(cov)
            for k in range(K):
                diff = X - self.means_[k]
                maha = np.sum((diff @ inv_cov) * diff, axis=1)
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + logdet + maha)

        return log_prob

    def _e_step(self, X):
        # Passo E: responsabilidades
        weighted_log_prob = self._estimate_log_gaussian_prob(X) + self._safe_log(self.weights_)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        resp = np.exp(log_resp)
        return resp, np.mean(log_prob_norm)

    def _m_step(self, X, resp):
        # Passo M: atualizar pesos, médias e covariâncias
        n_samples, n_features = X.shape
        K = self.n_components

        Nk = resp.sum(axis=0) + 1e-12
        self.weights_ = Nk / np.maximum(n_samples, 1)
        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]

        if self.covariance_type == 'full':
            covs = np.zeros((K, n_features, n_features))
            for k in range(K):
                diff = X - self.means_[k]
                cov = (resp[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
                cov += self.reg_covar * np.eye(n_features)
                covs[k] = cov
            self.covariances_ = covs

        elif self.covariance_type == 'diag':
            covs = np.zeros((K, n_features))
            for k in range(K):
                diff = X - self.means_[k]
                cov_diag = (resp[:, k][:, np.newaxis] * (diff ** 2)).sum(axis=0) / Nk[k]
                covs[k] = np.maximum(cov_diag, self.reg_covar)
            self.covariances_ = covs

        elif self.covariance_type == 'spherical':
            covs = np.zeros(K)
            for k in range(K):
                diff = X - self.means_[k]
                val = (resp[:, k] * np.sum(diff ** 2, axis=1)).sum() / (Nk[k] * n_features)
                covs[k] = max(float(val), self.reg_covar)
            self.covariances_ = covs

        else:  # tied
            cov = np.zeros((n_features, n_features))
            for k in range(K):
                diff = X - self.means_[k]
                cov += (resp[:, k][:, np.newaxis] * diff).T @ diff
            cov /= np.maximum(n_samples, 1)
            cov += self.reg_covar * np.eye(n_features)
            self.covariances_ = cov

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        if n_samples < 1:
            raise ValueError("X deve conter ao menos 1 amostra.")

        self._initialize_parameters(X)
        prev_lower_bound = -np.inf

        for n_iter in range(1, self.max_iter + 1):
            resp, lower_bound = self._e_step(X)
            self._m_step(X, resp)

            change = lower_bound - prev_lower_bound
            prev_lower_bound = lower_bound

            if abs(change) < self.tol:
                self.converged_ = True
                self.n_iter_ = n_iter
                self.lower_bound_ = lower_bound
                break
        else:
            self.converged_ = False
            self.n_iter_ = self.max_iter
            self.lower_bound_ = prev_lower_bound

        return self

    def score_samples(self, X):
        X = np.asarray(X, dtype=float)
        weighted_log_prob = self._estimate_log_gaussian_prob(X) + self._safe_log(self.weights_)
        return logsumexp(weighted_log_prob, axis=1)


class GMMBayesClassifier:
    VALID_COV_TYPES = ('full', 'tied', 'diag', 'spherical')

    def __init__(self, n_components=3, covariance_type='full',
                 reg_covar=1e-6, random_state=42):
        if covariance_type not in self.VALID_COV_TYPES:
            raise ValueError(f"covariance_type deve ser um de {self.VALID_COV_TYPES}")

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.classes_ = None
        self.priors_ = None
        self.gmms_ = {}
        self.n_features_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        n_samples = len(y)

        
        self.priors_ = {}
        for c in self.classes_:
            self.priors_[c] = np.sum(y == c) / n_samples

        
        for c in self.classes_:
            Xc = X[y == c]
            nc = len(Xc)

            
            if isinstance(self.n_components, dict):
                n_comp = self.n_components.get(c, 3)
            else:
                n_comp = self.n_components

           
            n_comp = min(n_comp, nc)
            if n_comp < 1:
                n_comp = 1

            
            gmm = GaussianMixtureEM(
                n_components=n_comp,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                max_iter=200,
                init_params='kmeans'
            )

            try:
                gmm.fit(Xc)
            except Exception:
                # Se falhar, tentar com menos componentes
                gmm = GaussianMixtureEM(
                    n_components=1,
                    covariance_type=self.covariance_type,
                    reg_covar=self.reg_covar,
                    random_state=self.random_state,
                    max_iter=200,
                    init_params='kmeans'
                )
                gmm.fit(Xc)

            self.gmms_[c] = gmm

        return self

    def _log_likelihood(self, X, class_idx):
      
        c = self.classes_[class_idx]
        gmm = self.gmms_[c]
        log_lik = gmm.score_samples(X)
        return log_lik

    def decision_function(self, X):
      
        K = len(self.classes_)
        scores = np.zeros((X.shape[0], K))

        for i in range(K):
          
            log_lik = self._log_likelihood(X, i)

          
            log_prior = np.log(self.priors_[self.classes_[i]] + 1e-300)

          
            scores[:, i] = log_lik + log_prior

        return scores

    def predict_proba(self, X):
       
        scores = self.decision_function(X)

      
        scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return scores_exp / scores_exp.sum(axis=1, keepdims=True)

    def predict(self, X):
       
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
      
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return (f"GMMBayesClassifier("
                f"n_components={self.n_components}, "
                f"covariance_type='{self.covariance_type}')")



class GMMBayesFull(GMMBayesClassifier):
   
    def __init__(self, n_components=3, reg_covar=1e-6, random_state=42):
        super().__init__(
            n_components=n_components,
            covariance_type='full',
            reg_covar=reg_covar,
            random_state=random_state
        )


class GMMBayesDiagonal(GMMBayesClassifier):
   
    def __init__(self, n_components=3, reg_covar=1e-6, random_state=42):
        super().__init__(
            n_components=n_components,
            covariance_type='diag',
            reg_covar=reg_covar,
            random_state=random_state
        )


class GMMBayesSpherical(GMMBayesClassifier):
    
    def __init__(self, n_components=3, reg_covar=1e-6, random_state=42):
        super().__init__(
            n_components=n_components,
            covariance_type='spherical',
            reg_covar=reg_covar,
            random_state=random_state
        )


class GMMBayesTied(GMMBayesClassifier):
    
    def __init__(self, n_components=3, reg_covar=1e-6, random_state=42):
        super().__init__(
            n_components=n_components,
            covariance_type='tied',
            reg_covar=reg_covar,
            random_state=random_state
        )


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Teste do GMM-Bayes Classifier (EM explícito)")
    print("-" * 52)

   
    classifiers = {
        'GMM-Bayes Full': GMMBayesFull(n_components=3),
        'GMM-Bayes Diagonal': GMMBayesDiagonal(n_components=3),
        'GMM-Bayes Spherical': GMMBayesSpherical(n_components=3),
        'GMM-Bayes Tied': GMMBayesTied(n_components=3),
    }

    print(f"{'Classificador':<25} {'Acurácia':>10}")
    print("-" * 40)

    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        print(f"{name:<25} {acc*100:>9.2f}%")
