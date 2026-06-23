import numpy as np
from scipy.special import logsumexp


class ParzenBayesClassifier:
   

    def __init__(self, bandwidth=None, bandwidth_scale=0.5):
       
        self.bandwidth = bandwidth
        self.bandwidth_scale = bandwidth_scale
        self.classes_ = None
        self.priors_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.bandwidths_ = None  # bandwidth por classe
        self.n_features_ = None

    def _estimate_bandwidth(self, Xc):
        """Estima bandwidth pela regra: h = scale * sigma * n^(-1/(d+4))"""
        n, d = Xc.shape
        sigma = np.mean(np.std(Xc, axis=0))
        if sigma < 1e-12:
            sigma = 1.0
        h = self.bandwidth_scale * sigma * n ** (-1.0 / (d + 4))
        return max(h, 1e-12)

    def fit(self, X, y):
        """
        Armazena os dados de treinamento para usar na estimação Parzen.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        n_samples = len(y)

        # Priors
        self.priors_ = {}
        for c in self.classes_:
            self.priors_[c] = np.sum(y == c) / n_samples

        # Bandwidth por classe
        self.bandwidths_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            if self.bandwidth is not None:
                self.bandwidths_[c] = self.bandwidth
            else:
                self.bandwidths_[c] = self._estimate_bandwidth(Xc)

        self.X_train_ = X
        self.y_train_ = y
        return self

    def _log_parzen_density(self, X, Xc, h):
        """
        Calcula log-density Parzen para pontos X dado conjunto de treino Xc
        com kernel Gaussiano e bandwidth h.

        log p(X|c) = -log(n_c) - (d/2)*log(2pi) - d*log(h)
                     + logsumexp( -0.5 * ||x - x_i||^2 / h^2 )
        """
        n_train = Xc.shape[0]
        d = Xc.shape[1]
        log_const = -np.log(n_train) - 0.5 * d * np.log(2 * np.pi) - d * np.log(h)

       
        X_norm_sq = np.sum(X ** 2, axis=1)  # (n_test,)
        Xc_norm_sq = np.sum(Xc ** 2, axis=1)  # (n_train,)

        # dist^2 = X_norm_sq[:, None] + Xc_norm_sq[None, :] - 2 * X @ Xc.T
        dist2 = X_norm_sq[:, np.newaxis] + Xc_norm_sq[np.newaxis, :] - 2.0 * np.dot(X, Xc.T)
        dist2 = np.clip(dist2, 0.0, None)  # evitar pequenos negativos

        log_kernel = -0.5 * dist2 / (h ** 2)  # (n_test, n_train)
        log_density = log_const + logsumexp(log_kernel, axis=1)

        return log_density

    def decision_function(self, X):
        """
        Retorna scores logarítmicos: log( p(x|c) * P(c) ) para cada classe.
        """
        X = np.asarray(X, dtype=float)
        K = len(self.classes_)
        scores = np.zeros((X.shape[0], K))

        for i, c in enumerate(self.classes_):
            Xc = self.X_train_[self.y_train_ == c]
            h = self.bandwidths_[c]
            log_lik = self._log_parzen_density(X, Xc, h)
            log_prior = np.log(max(self.priors_[c], 1e-300))
            scores[:, i] = log_lik + log_prior

        return scores

    def predict_proba(self, X):
        """Retorna probabilidades posteriores."""
        scores = self.decision_function(X)
        scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return scores_exp / scores_exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Retorna predição de classe."""
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return (f"ParzenBayesClassifier(bandwidth={self.bandwidth}, "
                f"bandwidth_scale={self.bandwidth_scale})")


# Alias mais conveniente
ParzenBayes = ParzenBayesClassifier