"""
Comparação de Resultados: DMC e KNN
Trabalho 2  vs  Trabalho 3

Trabalho 2: Bayesiano Gaussiano, KNN (k=3), DMC
            Split 80/20 | Sem normalização | 20 realizações

Trabalho 3: Naive Bayes, KNN (k=5), DMC
            Split 70/30 | COM normalização z-score | 20 realizações
"""

import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICADORES
# ──────────────────────────────────────────────────────────────────────────────

class KNN:
    def __init__(self, k=3):
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
    def fit(self, X, y):
        self.classes_   = np.unique(y)
        self.centroids_ = {c: X[y == c].mean(axis=0) for c in self.classes_}

    def predict(self, X):
        preds = []
        for x in X:
            d = {c: np.sqrt(((x - self.centroids_[c])**2).sum()) for c in self.classes_}
            preds.append(min(d, key=d.get))
        return np.array(preds)


# ──────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ──────────────────────────────────────────────────────────────────────────────

def stratified_split_t2(X, y, frac_teste=0.2, seed=None):
    """Divisão estratificada – protocolo Trabalho 2 (seed via np.random.default_rng)."""
    rng = np.random.default_rng(seed)
    idx_tr, idx_te = [], []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_te = max(1, int(len(idx_c) * frac_teste))
        idx_te.extend(idx_c[:n_te])
        idx_tr.extend(idx_c[n_te:])
    return X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]


def stratified_split_t3(X, y, test_size, rng):
    """Divisão estratificada – protocolo Trabalho 3 (rng externo)."""
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = max(1, int(len(idx) * test_size))
        te.extend(idx[:n]); tr.extend(idx[n:])
    return np.array(tr), np.array(te)


def normalize(X_tr, X_te):
    mu, s = X_tr.mean(0), X_tr.std(0) + 1e-9
    return (X_tr - mu) / s, (X_te - mu) / s


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# ──────────────────────────────────────────────────────────────────────────────
# AVALIAÇÃO
# ──────────────────────────────────────────────────────────────────────────────

def eval_trabalho2(X, y, n_real=20, k_knn=3):
    """
    Protocolo Trabalho 2: 80/20 estratificado, sem normalização, KNN k=3.
    """
    knn_accs, dmc_accs = [], []
    for seed in range(n_real):
        Xtr, Xte, ytr, yte = stratified_split_t2(X, y, 0.2, seed)
        knn = KNN(k_knn); knn.fit(Xtr, ytr)
        dmc = DMC();       dmc.fit(Xtr, ytr)
        knn_accs.append(accuracy(yte, knn.predict(Xte)))
        dmc_accs.append(accuracy(yte, dmc.predict(Xte)))
    return np.mean(knn_accs), np.std(knn_accs), np.mean(dmc_accs), np.std(dmc_accs)


def eval_trabalho3(X, y, n_real=20, k_knn=3):
    """
    Protocolo Trabalho 3: 80/20 estratificado, COM normalização z-score, KNN k=3.
    Seeds iguais ao Trabalho 2 (0..19).
    """
    knn_accs, dmc_accs = [], []
    for i in range(n_real):
        rng = np.random.default_rng(i)
        tr, te = stratified_split_t3(X, y, 0.2, rng)
        Xtr_n, Xte_n = normalize(X[tr], X[te])
        knn = KNN(k_knn); knn.fit(Xtr_n, y[tr])
        dmc = DMC();       dmc.fit(Xtr_n, y[tr])
        knn_accs.append(accuracy(y[te], knn.predict(Xte_n)))
        dmc_accs.append(accuracy(y[te], dmc.predict(Xte_n)))
    return np.mean(knn_accs), np.std(knn_accs), np.mean(dmc_accs), np.std(dmc_accs)


# ──────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DOS DATASETS
# ──────────────────────────────────────────────────────────────────────────────

def load_iris():
    from sklearn.datasets import load_iris as sk_iris
    d = sk_iris()
    return d.data, d.target

def load_breast_cancer_t2():
    path = os.path.join('trabalho2', 'Breast Cancer',
                        'breast+cancer+wisconsin+diagnostic', 'wdbc.data')
    import pandas as pd
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, 2:].values.astype(float)
    y = (df.iloc[:, 1] == 'M').astype(int).values
    return X, y

def load_breast_cancer_t3():
    path = os.path.join('trabalho3', 'breast_cancer', 'wdbc.data')
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            rows.append(parts)
    arr = np.array(rows)
    X = arr[:, 2:].astype(float)
    y = (arr[:, 1] == 'M').astype(int)
    return X, y

def load_dermatology_t2():
    import pandas as pd
    cols = [
        'erythema','scaling','definite_borders','itching','koebner_phenomenon',
        'polygonal_papules','follicular_papules','oral_mucosal_involvement',
        'knee_elbow_involvement','scalp_involvement','family_history',
        'melanin_incontinence','eosinophils_infiltrate','PNL_infiltrate',
        'fibrosis_papillary_dermis','exocytosis','acanthosis','hyperkeratosis',
        'parakeratosis','clubbing_rete_ridges','elongation_rete_ridges',
        'thinning_suprapapillary','spongiform_pustule','munro_microabcess',
        'focal_hypergranulosis','disappearance_granular_layer',
        'vacuolisation_basal_layer','spongiosis','saw_tooth_retes',
        'follicular_horn_plug','perifollicular_parakeratosis',
        'inflammatory_mononuclear','band_like_infiltrate','age','class'
    ]
    path = os.path.join('trabalho2', 'dermatology', 'dermatology', 'dermatology.data')
    df = pd.read_csv(path, header=None, names=cols, na_values='?')
    df['age'].fillna(df['age'].median(), inplace=True)
    X = df.drop('class', axis=1).values.astype(float)
    y = df['class'].values
    return X, y

def load_dermatology_t3():
    path = os.path.join('trabalho3', 'dermatology', 'dermatology.data')
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) < 35: continue
            rows.append(parts[:35])
    arr = np.array(rows)
    y_raw = arr[:, 34].astype(int)
    X_raw = arr[:, :34]
    X = np.zeros(X_raw.shape, dtype=float)
    for j in range(X_raw.shape[1]):
        col   = X_raw[:, j]
        valid = np.array([v != '?' for v in col])
        vals  = np.array([float(v) if v != '?' else np.nan for v in col])
        vals[~valid] = np.nanmean(vals)
        X[:, j] = vals
    y = y_raw - 1
    return X, y

def load_vertebral_t2():
    from scipy.io import arff
    import pandas as pd
    path = os.path.join('trabalho2', 'Vertebral', 'vertebral+column', 'column_3C_weka.arff')
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values
    return X, y

def load_vertebral_t3():
    path = os.path.join('trabalho3', 'vertebral', 'column_3C.dat')
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('@'): continue
            parts = line.split()
            if len(parts) == 7:
                data.append(parts)
    arr   = np.array(data)
    X     = arr[:, :6].astype(float)
    y_raw = arr[:, 6]
    lmap  = {'Hernia':'Hernia','Spondylolisthesis':'Spondylolisthesis','Normal':'Normal',
             'DH':'Hernia','SL':'Spondylolisthesis','NO':'Normal'}
    y = np.array([lmap.get(v, v) for v in y_raw])
    return X, y

def load_artificial_t2(seed=42, n_per_class=40):
    rng = np.random.default_rng(seed)
    params = [
        ([1.5, 7.0], [[0.40, 0.00], [0.00, 0.40]]),
        ([6.5, 7.0], [[0.50, 0.10], [0.10, 0.30]]),
        ([4.0, 3.5], [[0.40, 0.00], [0.00, 0.40]]),
    ]
    Xs, ys = [], []
    for label, (mean, cov) in enumerate(params):
        Xs.append(rng.multivariate_normal(mean, cov, n_per_class))
        ys.append(np.full(n_per_class, label))
    return np.vstack(Xs), np.concatenate(ys)

def load_artificial_t3(seed=42):
    path = os.path.join('trabalho3', 'artificial_I', 'artificial_I.csv')
    if os.path.exists(path):
        arr = np.loadtxt(path, delimiter=',', skiprows=1)
        return arr[:, :2], arr[:, 2].astype(int)
    # regenerar se não existir
    rng = np.random.default_rng(seed)
    X0  = rng.multivariate_normal([-2, -2], [[1.0, 0.3],  [0.3,  1.0]], 100)
    X1  = rng.multivariate_normal([ 2, -2], [[1.0,-0.3],  [-0.3, 1.0]], 100)
    X2  = rng.multivariate_normal([ 0,  2], [[1.0, 0.0],  [0.0,  1.0]], 100)
    X   = np.vstack([X0, X1, X2])
    y   = np.array([0]*100 + [1]*100 + [2]*100)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def fmt(mean, std):
    return f"{mean*100:6.2f}% ± {std*100:5.2f}%"

def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE)

    datasets = {
        "Iris":          (load_iris,           load_iris),
        "Breast Cancer": (load_breast_cancer_t2, load_breast_cancer_t3),
        "Dermatology":   (load_dermatology_t2,   load_dermatology_t3),
        "Vertebral":     (load_vertebral_t2,     load_vertebral_t3),
        "Artificial":    (load_artificial_t2,    load_artificial_t3),
    }

    header = (
        f"\n{'─'*100}\n"
        f"{'COMPARAÇÃO: DMC e KNN — Trabalho 2 vs Trabalho 3':^100}\n"
        f"{'─'*100}\n"
        f"  Trabalho 2 : KNN k=3 | Split 80/20 | Sem normalização | 20 realizações\n"
        f"  Trabalho 3 : KNN k=3 | Split 80/20 | Com normalização z-score | 20 realizações\n"
        f"{'─'*100}"
    )
    print(header)

    col = 16
    sep = "─" * 100
    print(f"\n{'Dataset':<16} {'Classificador':<12} "
          f"{'Trab2 (Acurácia ± DP)':^26}  {'Trab3 (Acurácia ± DP)':^26}  {'Δ (T3-T2)':>10}")
    print(sep)

    results = {}

    for name, (load_t2, load_t3) in datasets.items():
        try:
            X2, y2 = load_t2()
            X3, y3 = load_t3()
        except Exception as e:
            print(f"{name:<16}  ERRO ao carregar: {e}")
            continue

        k2_m, k2_s, d2_m, d2_s = eval_trabalho2(X2, y2)
        k3_m, k3_s, d3_m, d3_s = eval_trabalho3(X3, y3)

        results[name] = dict(knn_t2=(k2_m,k2_s), knn_t3=(k3_m,k3_s),
                             dmc_t2=(d2_m,d2_s), dmc_t3=(d3_m,d3_s))

        delta_knn = (k3_m - k2_m) * 100
        delta_dmc = (d3_m - d2_m) * 100
        sign_k = "▲" if delta_knn > 0 else ("▼" if delta_knn < 0 else "═")
        sign_d = "▲" if delta_dmc > 0 else ("▼" if delta_dmc < 0 else "═")

        print(f"\n{name:<16} {'KNN':<12} {fmt(k2_m, k2_s):^26}  {fmt(k3_m, k3_s):^26}  "
              f"{sign_k} {delta_knn:+.2f}%")
        print(f"{'':16} {'DMC':<12} {fmt(d2_m, d2_s):^26}  {fmt(d3_m, d3_s):^26}  "
              f"{sign_d} {delta_dmc:+.2f}%")

    print(f"\n{sep}")
    print("▲ = Trab3 melhor  |  ▼ = Trab2 melhor  |  ═ = empate\n")

    # ── Resumo global ──────────────────────────────────────────────────────
    if results:
        print(f"\n{'─'*60}")
        print(f"{'RESUMO GERAL — Média sobre os datasets':^60}")
        print(f"{'─'*60}")
        knn_t2_all = [v['knn_t2'][0] for v in results.values()]
        knn_t3_all = [v['knn_t3'][0] for v in results.values()]
        dmc_t2_all = [v['dmc_t2'][0] for v in results.values()]
        dmc_t3_all = [v['dmc_t3'][0] for v in results.values()]

        print(f"{'':20} {'Trabalho 2':>15}  {'Trabalho 3':>15}")
        print(f"  {'KNN (média global)':<20} {np.mean(knn_t2_all)*100:>14.2f}%  {np.mean(knn_t3_all)*100:>14.2f}%")
        print(f"  {'DMC (média global)':<20} {np.mean(dmc_t2_all)*100:>14.2f}%  {np.mean(dmc_t3_all)*100:>14.2f}%")
        print()

        # melhor classificador por dataset
        print(f"{'─'*60}")
        print(f"{'MELHOR CLASSIFICADOR POR DATASET':^60}")
        print(f"{'─'*60}")
        print(f"{'Dataset':<16} {'Trab2':>12}  {'Trab3':>12}")
        for name, v in results.items():
            best_t2 = "KNN" if v['knn_t2'][0] >= v['dmc_t2'][0] else "DMC"
            best_t3 = "KNN" if v['knn_t3'][0] >= v['dmc_t3'][0] else "DMC"
            t2_acc  = max(v['knn_t2'][0], v['dmc_t2'][0]) * 100
            t3_acc  = max(v['knn_t3'][0], v['dmc_t3'][0]) * 100
            print(f"{name:<16} {best_t2:>5} {t2_acc:>6.2f}%  {best_t3:>5} {t3_acc:>6.2f}%")
        print()


if __name__ == "__main__":
    main()
