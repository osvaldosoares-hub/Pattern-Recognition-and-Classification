"""
Grid Search com K-Fold Cross Validation para ajustar o bandwidth do Parzen Bayes.

FLUXO SEM VAZAMENTO:
1. Separa TREINO e TESTE (holdout) -> teste fica congelado
2. No TREINO, faz K-Fold CV para escolher o melhor bandwidth
3. Retreina o ParzenBayes com o melhor bandwidth no TREINO completo
4. Avalia no TESTE (que nunca foi usado no CV)

Por que não vaza?
- O conjunto de teste NUNCA participa da escolha do hiperparâmetro
- O K-Fold CV divide apenas o TREINO em K subconjuntos
- A cada fold: treina em K-1 folds, valida no fold restante
- A média dos K folds estima a performance generalizável
- O melhor bandwidth é escolhido com base APENAS nos dados de treino
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from parzen_bayes import ParzenBayesClassifier


def grid_search_cv(X_train, y_train, bandwidth_scales, n_folds=5, random_state=42):
  
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_scores = {}
    for scale in bandwidth_scales:
        fold_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr_fold = X_train[train_idx]
            y_tr_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]

            clf = ParzenBayesClassifier(bandwidth=None, bandwidth_scale=scale)
            clf.fit(X_tr_fold, y_tr_fold)
            y_pred = clf.predict(X_val_fold)
            acc = accuracy_score(y_val_fold, y_pred)
            fold_scores.append(acc)

        cv_scores[scale] = fold_scores

    # Escolhe o melhor scale pela média da CV
    mean_scores = {s: np.mean(v) for s, v in cv_scores.items()}
    best_scale = max(mean_scores, key=mean_scores.get)
    best_score = mean_scores[best_scale]

    return best_scale, cv_scores, best_score


def grid_search_cv_bandwidth(X_train, y_train, bandwidth_values, n_folds=5, random_state=42):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_scores = {}
    for h in bandwidth_values:
        fold_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr_fold = X_train[train_idx]
            y_tr_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]

            clf = ParzenBayesClassifier(bandwidth=h, bandwidth_scale=1.0)
            clf.fit(X_tr_fold, y_tr_fold)
            y_pred = clf.predict(X_val_fold)
            acc = accuracy_score(y_val_fold, y_pred)
            fold_scores.append(acc)

        cv_scores[h] = fold_scores

    mean_scores = {s: np.mean(v) for s, v in cv_scores.items()}
    best_h = max(mean_scores, key=mean_scores.get)
    best_score = mean_scores[best_h]

    return best_h, cv_scores, best_score


# Grid padrão de bandwidth_scale para testar
DEFAULT_BANDWIDTH_SCALES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

# Grid de bandwidth fixo (alternativa)
DEFAULT_BANDWIDTH_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]