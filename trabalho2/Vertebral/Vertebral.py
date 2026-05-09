"""
Classificador Bayesiano Gaussiano Multivariado
Comparação com KNN e DMC - Dataset Coluna Vertebral

- Vetor de média e matriz de covariância específicos por classe
- Probabilidades a posteriori calculadas diretamente (sem simplificações)
- 20 realizações com acurácia e desvio padrão
- Matriz de confusão e visualização para a realização mediana
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.io import arff
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)

# =============================================================
# 1. CARREGAMENTO DOS DADOS
# =============================================================

def load_arff(filepath):
    """Carrega arquivo ARFF e retorna X (features), y (labels), classes."""
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values
    classes = np.unique(y)
    return X, y, classes, df.columns[:-1].tolist()


# =============================================================
# 2. CLASSIFICADOR BAYESIANO GAUSSIANO MULTIVARIADO
# =============================================================

class GaussianBayesClassifier:
    """
    Classificador Bayesiano Gaussiano Multivariado completo.

    Para cada classe c:
      - mu_c  : vetor de médias (d,)
      - Sigma_c: matriz de covariância (d x d) — específica por classe
      - pi_c  : probabilidade a priori P(c)

    Regra de decisão (log-escala para estabilidade numérica):
      log P(c | x) ∝ log P(x | c) + log P(c)
    onde
      log P(x | c) = -0.5 * [(x-mu)^T Sigma^{-1} (x-mu)
                              + log|Sigma| + d*log(2π)]
    """

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_   = {}
        self.covs_    = {}
        self.priors_  = {}
        n = len(y)
        for c in self.classes_:
            Xc = X[y == c]
            self.means_[c]  = np.mean(Xc, axis=0)
            # ddof=1 para covariância não-viesada
            self.covs_[c]   = np.cov(Xc, rowvar=False)
            self.priors_[c] = len(Xc) / n
        return self

    def _log_posterior(self, x, c):
        """
        Calcula log P(c|x) ∝ log P(x|c) + log P(c).
        Usa a fórmula completa da gaussiana multivariada sem simplificações.
        """
        mu  = self.means_[c]
        cov = self.covs_[c]
        d   = len(mu)
        # Regularização numérica mínima para garantir inversibilidade
        cov_reg = cov + np.eye(d) * 1e-9

        sign, logdet = np.linalg.slogdet(cov_reg)
        if sign <= 0:
            return -np.inf

        cov_inv = np.linalg.inv(cov_reg)
        diff    = x - mu
        # Forma quadrática
        maha2   = diff @ cov_inv @ diff

        log_likelihood = -0.5 * (maha2 + logdet + d * np.log(2 * np.pi))
        return log_likelihood + np.log(self.priors_[c])

    def predict_proba(self, X):
        """Retorna probabilidades a posteriori normalizadas para cada amostra."""
        result = []
        for x in X:
            log_posts = np.array([self._log_posterior(x, c) for c in self.classes_])
            # Subtrai o máximo para estabilidade numérica antes de exp
            log_posts -= np.max(log_posts)
            probs = np.exp(log_posts)
            probs /= probs.sum()
            result.append(probs)
        return np.array(result)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx   = np.argmax(proba, axis=1)
        return np.array([self.classes_[i] for i in idx])


# =============================================================
# 3. KNN
# =============================================================

class KNNClassifier:
    """K-Nearest Neighbors implementado do zero."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            dists   = np.linalg.norm(self.X_train - x, axis=1)
            k_idx   = np.argsort(dists)[: self.k]
            k_labels = self.y_train[k_idx]
            unique, counts = np.unique(k_labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        return np.array(predictions)


# =============================================================
# 4. DMC — Distância Mínima ao Centroide
# =============================================================

class DMCClassifier:
    """Classifica pela distância euclidiana mínima ao centroide de cada classe."""

    def fit(self, X, y):
        self.classes_   = np.unique(y)
        self.centroids_ = {c: np.mean(X[y == c], axis=0) for c in self.classes_}
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            dists = {c: np.linalg.norm(x - self.centroids_[c]) for c in self.classes_}
            predictions.append(min(dists, key=dists.get))
        return np.array(predictions)


# =============================================================
# 5. EXPERIMENTOS — 20 REALIZAÇÕES
# =============================================================

def run_experiments(X, y, classes, dataset_name, n_realizations=20,
                    test_size=0.3, base_seed=42, k_knn=5):
    """
    Executa n_realizations experimentos com splits aleatórios diferentes.
    Retorna resultados e a realização mais próxima da mediana (GBC).
    """
    gbc_accs, knn_accs, dmc_accs = [], [], []
    realizations = []

    for i in range(n_realizations):
        seed = base_seed + i
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        # --- GBC ---
        gbc = GaussianBayesClassifier().fit(X_tr, y_tr)
        y_pred_gbc = gbc.predict(X_te)
        acc_gbc = np.mean(y_pred_gbc == y_te)
        gbc_accs.append(acc_gbc)

        # --- KNN ---
        knn = KNNClassifier(k=k_knn).fit(X_tr, y_tr)
        y_pred_knn = knn.predict(X_te)
        knn_accs.append(np.mean(y_pred_knn == y_te))

        # --- DMC ---
        dmc = DMCClassifier().fit(X_tr, y_tr)
        y_pred_dmc = dmc.predict(X_te)
        dmc_accs.append(np.mean(y_pred_dmc == y_te))

        realizations.append({
            'seed': seed, 'idx': i + 1,
            'X_tr': X_tr, 'X_te': X_te,
            'y_tr': y_tr, 'y_te': y_te,
            'y_pred_gbc': y_pred_gbc,
            'gbc': gbc, 'acc_gbc': acc_gbc,
        })

    # Realização mais próxima da mediana (representativa da tendência central)
    median_acc = np.median(gbc_accs)
    median_idx = int(np.argmin(np.abs(np.array(gbc_accs) - median_acc)))
    best = realizations[median_idx]

    # Impressão dos resultados
    print(f"\n{'='*62}")
    print(f"  Dataset: {dataset_name}  |  {n_realizations} realizações  |  KNN k={k_knn}")
    print(f"{'='*62}")
    print(f"{'Classificador':<22} {'Acurácia Média':>16} {'Desvio Padrão':>15}")
    print(f"{'-'*53}")
    print(f"{'GBC (Bayesiano)':<22} {np.mean(gbc_accs):>16.4f} {np.std(gbc_accs):>15.4f}")
    print(f"{'KNN (k='+str(k_knn)+')':<22} {np.mean(knn_accs):>16.4f} {np.std(knn_accs):>15.4f}")
    print(f"{'DMC':<22} {np.mean(dmc_accs):>16.4f} {np.std(dmc_accs):>15.4f}")
    print(f"\nRealização escolhida para análise: #{best['idx']}  "
          f"(GBC acc = {best['acc_gbc']:.4f}, mediana = {median_acc:.4f})")

    return gbc_accs, knn_accs, dmc_accs, best


# =============================================================
# 6. MATRIZ DE CONFUSÃO
# =============================================================

def plot_confusion_matrix(y_true, y_pred, classes, title, ax):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel='Rótulo Verdadeiro', xlabel='Rótulo Predito',
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylim(len(classes) - 0.5, -0.5)


# =============================================================
# 7. ELIPSE DE COVARIÂNCIA (GAUSSIANA 2D)
# =============================================================

def plot_covariance_ellipse(ax, mean_2d, cov_2d, color, n_std=2.0, alpha=0.15, lw=1.5):
    """Elipse de covariância 2D via decomposição em autovalores."""
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    w = 2 * n_std * np.sqrt(max(eigvals[0], 0))
    h = 2 * n_std * np.sqrt(max(eigvals[1], 0))
    ell = Ellipse(xy=mean_2d, width=w, height=h, angle=angle,
                  facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw)
    ax.add_patch(ell)


# =============================================================
# 8. VISUALIZAÇÃO: GAUSSIANAS + TREINO/TESTE (PCA 2D)
# =============================================================

def plot_gaussians_pca(gbc, X_tr, y_tr, X_te, y_te, classes, title, ax):
    """
    Projeta dados em 2D via PCA e plota:
    - Pontos de treino (○) e teste (△)
    - Elipses de covariância 1σ e 2σ por classe
    - Centroide (★) de cada classe
    """
    pca = PCA(n_components=2)
    pca.fit(X_tr)

    Xtr2 = pca.transform(X_tr)
    Xte2 = pca.transform(X_te)

    colors = plt.cm.tab10(np.linspace(0, 0.7, len(classes)))
    color_map = {c: colors[i] for i, c in enumerate(classes)}
    W = pca.components_.T  # (d, 2)

    for c, col in color_map.items():
        mask_tr = y_tr == c
        mask_te = y_te == c
        ax.scatter(Xtr2[mask_tr, 0], Xtr2[mask_tr, 1],
                   c=[col], marker='o', s=30, alpha=0.5,
                   label=f'Treino — {c}', edgecolors='none')
        ax.scatter(Xte2[mask_te, 0], Xte2[mask_te, 1],
                   c=[col], marker='^', s=65, alpha=0.9,
                   label=f'Teste — {c}', edgecolors='black', linewidths=0.7)

        # Parâmetros da gaussiana projetados no espaço PCA
        mu_proj  = pca.transform(gbc.means_[c].reshape(1, -1))[0]
        cov_proj = W.T @ gbc.covs_[c] @ W  # Σ_2D = W^T Σ W

        plot_covariance_ellipse(ax, mu_proj, cov_proj, col, n_std=1.0, alpha=0.25, lw=2.0)
        plot_covariance_ellipse(ax, mu_proj, cov_proj, col, n_std=2.0, alpha=0.12, lw=1.5)
        ax.scatter(*mu_proj, c=[col], marker='*', s=250,
                   edgecolors='black', linewidths=0.8, zorder=6)

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}% variância)')
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}% variância)')
    ax.set_title(title)
    ax.legend(fontsize=8, loc='best', framealpha=0.85)
    ax.grid(True, alpha=0.25)


# =============================================================
# 9. BOXPLOT COMPARATIVO
# =============================================================

def plot_comparison(gbc_accs, knn_accs, dmc_accs, dataset_name, ax, k_knn):
    data   = [gbc_accs, knn_accs, dmc_accs]
    labels = ['GBC', f'KNN\n(k={k_knn})', 'DMC']
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Acurácia')
    ax.set_title(f'Comparação — {dataset_name}')
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.4)

    # Anotar média ± std
    for i, (acc_list, lbl) in enumerate(zip(data, labels), 1):
        ax.text(i, np.mean(acc_list) + 0.01,
                f'{np.mean(acc_list):.3f}±{np.std(acc_list):.3f}',
                ha='center', va='bottom', fontsize=7, color='navy')


# =============================================================
# 10. MAIN
# =============================================================

if __name__ == '__main__':

    base_path = r'vertebral+column'
    datasets = {
        '2 Classes (Normal/Anormal)': f'{base_path}/column_2C_weka.arff',
        '3 Classes (Hernia/Spondylolisthesis/Normal)': f'{base_path}/column_3C_weka.arff',
    }

    N_REALIZATIONS = 20
    TEST_SIZE      = 0.2
    BASE_SEED      = 42
    K_KNN          = 5

    all_results = {}

    for ds_name, ds_path in datasets.items():
        X, y, classes, feat_names = load_arff(ds_path)
        print(f"\nCarregado: {ds_path}  →  {X.shape[0]} amostras, "
              f"{X.shape[1]} features, classes: {list(classes)}")

        gbc_accs, knn_accs, dmc_accs, best = run_experiments(
            X, y, classes, ds_name,
            n_realizations=N_REALIZATIONS,
            test_size=TEST_SIZE,
            base_seed=BASE_SEED,
            k_knn=K_KNN
        )
        all_results[ds_name] = {
            'gbc': gbc_accs, 'knn': knn_accs, 'dmc': dmc_accs,
            'best': best, 'X': X, 'y': y, 'classes': classes,
        }

    # ─── Figura 1: Boxplots comparativos ───────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (ds_name, res) in zip(axes1, all_results.items()):
        plot_comparison(res['gbc'], res['knn'], res['dmc'], ds_name, ax, K_KNN)
    fig1.suptitle('Comparação de Classificadores — Coluna Vertebral\n'
                  f'({N_REALIZATIONS} realizações, {int(TEST_SIZE*100)}% teste)',
                  fontsize=13)
    fig1.tight_layout()
    fig1.savefig('comparacao_classificadores.png', dpi=150, bbox_inches='tight')
    print('\nFigura salva: comparacao_classificadores.png')

    # ─── Figuras por dataset: Confusão + Gaussianas ────────────
    for ds_name, res in all_results.items():
        best   = res['best']
        ds_tag = '2C' if '2 Classes' in ds_name else '3C'
        n_cls  = len(res['classes'])

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        fig2.suptitle(
            f'Realização #{best["idx"]} — {ds_name}\n'
            f'(GBC acurácia = {best["acc_gbc"]:.4f} ≈ mediana das {N_REALIZATIONS} realizações)',
            fontsize=11
        )

        # Matriz de confusão (GBC)
        plot_confusion_matrix(
            best['y_te'], best['y_pred_gbc'], res['classes'],
            f'Matriz de Confusão GBC\n(Realização #{best["idx"]})',
            axes2[0]
        )

        # Gaussianas projetadas em PCA 2D
        plot_gaussians_pca(
            best['gbc'],
            best['X_tr'], best['y_tr'],
            best['X_te'], best['y_te'],
            res['classes'],
            f'Gaussianas por Classe (PCA 2D)\nTreino (○) e Teste (△)',
            axes2[1]
        )

        fig2.tight_layout()
        fname = f'confusao_gaussianas_{ds_tag}.png'
        fig2.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'Figura salva: {fname}')

    # ─── Figura 3: Acurácia por realização ─────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (ds_name, res) in zip(axes3, all_results.items()):
        real_nums = range(1, N_REALIZATIONS + 1)
        ax.plot(real_nums, res['gbc'], 'o-', label='GBC',  color='#4C72B0', linewidth=1.5)
        ax.plot(real_nums, res['knn'], 's-', label=f'KNN(k={K_KNN})', color='#DD8452', linewidth=1.5)
        ax.plot(real_nums, res['dmc'], '^-', label='DMC',  color='#55A868', linewidth=1.5)
        best_idx = res['best']['idx']
        ax.axvline(best_idx, color='red', linestyle='--', linewidth=1.2,
                   label=f'Realiz. #{best_idx} (mediana)')
        ax.set_xlabel('Realização')
        ax.set_ylabel('Acurácia')
        ax.set_title(ds_name)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(list(real_nums))
    fig3.suptitle('Acurácia por Realização', fontsize=13)
    fig3.tight_layout()
    fig3.savefig('acuracia_por_realizacao.png', dpi=150, bbox_inches='tight')
    print('Figura salva: acuracia_por_realizacao.png')

    plt.show()
    print('\nConcluído.')
