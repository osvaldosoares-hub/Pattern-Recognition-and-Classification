"""Classificador Naive Bayes, KNN e DMC - Flor Iris
Dataset carregado do scikit-learn (150 amostras, 4 atributos, 3 classes).
Superfície de decisão e gaussianas: petal length (f2) x petal width (f3).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ─────────────────────────── CONFIGURAÇÕES ──────────────────────────────
N_REAL    = 20
TEST_SIZE = 0.2
KNN_K     = 3
F1, F2    = 2, 3          # petal length, petal width (mais discriminativos)

FEAT_NAMES  = ['sepal length', 'sepal width', 'petal length', 'petal width']
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

# ─────────────────────────── CLASSIFICADORES ────────────────────────────

class NaiveBayes:
    """Naive Bayes Gaussiano – atributos independentes entre si."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n = len(y)
        self.priors_, self.means_, self.stds_ = {}, {}, {}
        for c in self.classes_:
            mask = (y == c)
            self.priors_[c] = mask.sum() / n
            self.means_[c]  = X[mask].mean(axis=0)
            self.stds_[c]   = X[mask].std(axis=0) + 1e-9

    def _log_likelihood(self, x, c):
        mu, s = self.means_[c], self.stds_[c]
        return np.sum(-0.5 * np.log(2 * np.pi * s**2) - 0.5 * ((x - mu) / s)**2)

    def predict_proba(self, X):
        """Probabilidades a posteriori (log-sum-exp)."""
        result = []
        for x in X:
            lp = {c: np.log(self.priors_[c]) + self._log_likelihood(x, c)
                  for c in self.classes_}
            m  = max(lp.values())
            ls = m + np.log(sum(np.exp(v - m) for v in lp.values()))
            result.append({c: float(np.exp(lp[c] - ls)) for c in self.classes_})
        return result

    def predict(self, X):
        return np.array([max(p, key=p.get) for p in self.predict_proba(X)])


class KNN:
    """K Vizinhos Mais Próximos."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_, self.y_ = X.copy(), y.copy()

    def predict(self, X):
        preds = []
        for x in X:
            d    = np.sqrt(((self.X_ - x)**2).sum(axis=1))
            lbls = self.y_[np.argsort(d)[:self.k]]
            vals, cnt = np.unique(lbls, return_counts=True)
            preds.append(vals[np.argmax(cnt)])
        return np.array(preds)


class DMC:
    """Classificador de Distância Mínima ao Centróide."""

    def fit(self, X, y):
        self.classes_   = np.unique(y)
        self.centroids_ = {c: X[y == c].mean(axis=0) for c in self.classes_}

    def predict(self, X):
        preds = []
        for x in X:
            d = {c: np.sqrt(((x - self.centroids_[c])**2).sum()) for c in self.classes_}
            preds.append(min(d, key=d.get))
        return np.array(preds)


# ─────────────────────────── UTILITÁRIOS ────────────────────────────────

def stratified_split(X, y, test_size, rng):
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = max(1, int(len(idx) * test_size))
        te.extend(idx[:n]); tr.extend(idx[n:])
    return np.array(tr), np.array(te)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm  = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def normalize(X_tr, X_te):
    mu, s = X_tr.mean(0), X_tr.std(0) + 1e-9
    return (X_tr - mu) / s, (X_te - mu) / s


# ─────────────────────────── PLOTS ──────────────────────────────────────

def plot_cm(cm, names, title, ax):
    ax.imshow(cm, cmap='Blues')
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Predito', fontsize=8); ax.set_ylabel('Real', fontsize=8)
    th = cm.max() / 2
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > th else 'black', fontsize=11)


def plot_decision_surface(clf_cls, clf_kw, X, y, f1, f2,
                           tr_idx, te_idx, cnames, fnames, ax, title):
    classes = np.unique(y)
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(classes)))
    X2 = X[:, [f1, f2]]
    X_tr_n, X_te_n = normalize(X2[tr_idx], X2[te_idx])[:2]

    clf = clf_cls(**clf_kw)
    clf.fit(X_tr_n, y[tr_idx])

    pad = 0.6
    xmn, xmx = X_tr_n[:, 0].min()-pad, X_tr_n[:, 0].max()+pad
    ymn, ymx = X_tr_n[:, 1].min()-pad, X_tr_n[:, 1].max()+pad
    xx, yy = np.meshgrid(np.linspace(xmn, xmx, 300), np.linspace(ymn, ymx, 300))
    Z   = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    cm_ = {c: i for i, c in enumerate(classes)}
    Z   = np.array([cm_[z] for z in Z]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, levels=np.arange(-0.5, len(classes)), cmap='tab10')
    for i, c in enumerate(classes):
        ax.scatter(X_tr_n[y[tr_idx]==c, 0], X_tr_n[y[tr_idx]==c, 1],
                   color=colors[i], marker='^', s=25, alpha=0.5)
        ax.scatter(X_te_n[y[te_idx]==c, 0], X_te_n[y[te_idx]==c, 1],
                   color=colors[i], marker='o', s=45, edgecolors='k',
                   linewidths=0.5, label=cnames[i])
    ax.set_xlabel(fnames[f1], fontsize=8); ax.set_ylabel(fnames[f2], fontsize=8)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7)


def plot_gaussians(X, y, tr_idx, te_idx, f1, f2, fnames, cnames, ax, title):
    classes = np.unique(y)
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(classes)))
    X2 = X[:, [f1, f2]]
    X_tr_n, X_te_n = normalize(X2[tr_idx], X2[te_idx])[:2]

    all_ = np.vstack([X_tr_n, X_te_n])
    pad  = 0.5
    xi = np.linspace(all_[:, 0].min()-pad, all_[:, 0].max()+pad, 200)
    yi = np.linspace(all_[:, 1].min()-pad, all_[:, 1].max()+pad, 200)
    xx, yy = np.meshgrid(xi, yi)

    for i, c in enumerate(classes):
        mtr = (y[tr_idx] == c); mte = (y[te_idx] == c)
        mu = X_tr_n[mtr].mean(0); s = X_tr_n[mtr].std(0) + 1e-9
        Z  = (np.exp(-0.5*((xx-mu[0])/s[0])**2) *
               np.exp(-0.5*((yy-mu[1])/s[1])**2)) / (2*np.pi*s[0]*s[1])
        ax.contour(xx, yy, Z, levels=4, colors=[colors[i]], linewidths=1.5, alpha=0.8)
        ax.scatter(X_tr_n[mtr, 0], X_tr_n[mtr, 1],
                   color=colors[i], marker='^', s=20, alpha=0.4)
        ax.scatter(X_te_n[mte, 0], X_te_n[mte, 1],
                   color=colors[i], marker='o', s=45, edgecolors='k',
                   linewidths=0.5, label=cnames[i])
    ax.set_xlabel(fnames[f1], fontsize=9); ax.set_ylabel(fnames[f2], fontsize=9)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8)


# ─────────────────────────── MAIN ───────────────────────────────────────

def main():
    print("=" * 60)
    print("IRIS  —  Naive Bayes, KNN, DMC")
    print("=" * 60)

    iris    = load_iris()
    X, y    = iris.data, iris.target
    classes = np.unique(y)
    print(f"Amostras: {len(X)}  |  Atributos: {X.shape[1]}  |  Classes: {len(classes)}")
    print(f"Distribuicao: { {CLASS_NAMES[c]: int((y==c).sum()) for c in classes} }\n")

    nb_accs, knn_accs, dmc_accs = [], [], []
    records = []

    for i in range(N_REAL):
        rng = np.random.default_rng(i)
        tr, te = stratified_split(X, y, TEST_SIZE, rng)
        Xtr, Xte = normalize(X[tr], X[te])[:2]

        nb  = NaiveBayes();  nb.fit(Xtr, y[tr])
        knn = KNN(KNN_K);    knn.fit(Xtr, y[tr])
        dmc = DMC();          dmc.fit(Xtr, y[tr])

        nb_p  = nb.predict(Xte)
        knn_p = knn.predict(Xte)
        dmc_p = dmc.predict(Xte)

        na = accuracy(y[te], nb_p)
        ka = accuracy(y[te], knn_p)
        da = accuracy(y[te], dmc_p)

        nb_accs.append(na); knn_accs.append(ka); dmc_accs.append(da)
        records.append(dict(tr=tr, te=te,
                            nb_p=nb_p, knn_p=knn_p, dmc_p=dmc_p,
                            na=na, ka=ka, da=da))

    print(f"{'Classificador':<22} {'Acuracia Media':>15}  {'Desvio Padrao':>14}")
    print("-" * 54)
    for nm, ac in [('Naive Bayes', nb_accs), (f'KNN (k={KNN_K})', knn_accs), ('DMC', dmc_accs)]:
        print(f"{nm:<22} {np.mean(ac)*100:>14.2f}%  {np.std(ac)*100:>13.2f}%")
    print()

    med = int(np.argmin([abs(r['na'] - np.mean(nb_accs)) for r in records]))
    r   = records[med]
    print(f"Realizacao escolhida para confusao: #{med+1}  "
          f"(NB={r['na']*100:.1f}%  KNN={r['ka']*100:.1f}%  DMC={r['da']*100:.1f}%)")
    print("Criterio: realizacao com acuracia NB mais proxima da media — mais representativa.\n")

    cm_nb  = confusion_matrix(y[r['te']], r['nb_p'],  classes)
    cm_knn = confusion_matrix(y[r['te']], r['knn_p'], classes)
    cm_dmc = confusion_matrix(y[r['te']], r['dmc_p'], classes)

    # ── Figura 1: Matrizes de Confusão ───────────────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig1.suptitle(f'Matrizes de Confusao — Iris  (Realizacao #{med+1})', fontsize=12)
    plot_cm(cm_nb,  CLASS_NAMES, f'Naive Bayes\nAcc={r["na"]*100:.1f}%',  axes[0])
    plot_cm(cm_knn, CLASS_NAMES, f'KNN (k={KNN_K})\nAcc={r["ka"]*100:.1f}%', axes[1])
    plot_cm(cm_dmc, CLASS_NAMES, f'DMC\nAcc={r["da"]*100:.1f}%',          axes[2])
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("Salvo: confusion_matrices.png")

    # ── Figura 2: Superfícies de Decisão ─────────────────────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(f'Superficies de Decisao — Iris\n'
                  f'Atributos: {FEAT_NAMES[F1]} x {FEAT_NAMES[F2]}', fontsize=12)
    for ax, (cls_, kw, ttl) in zip(axes, [
            (NaiveBayes, {},           'Naive Bayes'),
            (KNN,        {'k': KNN_K}, f'KNN (k={KNN_K})'),
            (DMC,        {},           'DMC')]):
        plot_decision_surface(cls_, kw, X, y, F1, F2,
                              r['tr'], r['te'], CLASS_NAMES, FEAT_NAMES, ax, ttl)
    plt.tight_layout()
    plt.savefig('decision_surfaces.png', dpi=150, bbox_inches='tight')
    print("Salvo: decision_surfaces.png")

    # ── Figura 3: Gaussianas sobre os dados ───────────────────────────────
    fig3, ax = plt.subplots(figsize=(7, 6))
    plot_gaussians(X, y, r['tr'], r['te'], F1, F2, FEAT_NAMES, CLASS_NAMES, ax,
                   f'Gaussianas NB por Classe — Iris\n'
                   f'{FEAT_NAMES[F1]} x {FEAT_NAMES[F2]}')
    plt.tight_layout()
    plt.savefig('gaussians.png', dpi=150, bbox_inches='tight')
    print("Salvo: gaussians.png")


if __name__ == '__main__':
    main()
