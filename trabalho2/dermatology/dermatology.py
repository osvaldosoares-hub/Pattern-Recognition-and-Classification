
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


col_names = [
    'erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon',
    'polygonal_papules', 'follicular_papules', 'oral_mucosal_involvement',
    'knee_elbow_involvement', 'scalp_involvement', 'family_history',
    'melanin_incontinence', 'eosinophils_infiltrate', 'PNL_infiltrate',
    'fibrosis_papillary_dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis',
    'parakeratosis', 'clubbing_rete_ridges', 'elongation_rete_ridges',
    'thinning_suprapapillary', 'spongiform_pustule', 'munro_microabcess',
    'focal_hypergranulosis', 'disappearance_granular_layer',
    'vacuolisation_basal_layer', 'spongiosis', 'saw_tooth_retes',
    'follicular_horn_plug', 'perifollicular_parakeratosis',
    'inflammatory_mononuclear', 'band_like_infiltrate', 'age', 'class'
]

df = pd.read_csv('dermatology/dermatology.data', header=None,
                 names=col_names, na_values='?')
# Preencher valores ausentes com a mediana (apenas coluna 'age')
df['age'].fillna(df['age'].median(), inplace=True)

X = df.drop('class', axis=1).values.astype(float)
y = df['class'].values

CLASS_NAMES = {
    1: 'Psoriasis',
    2: 'Seboreic Dermatitis',
    3: 'Lichen Planus',
    4: 'Pityriasis Rosea',
    5: 'Cronic Dermatitis',
    6: 'Pityriasis Rubra Pilaris'
}
CLASSES = sorted(np.unique(y))
COLORS  = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']


class GaussianBayesClassifier:
   

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n = len(y)
        self.priors_ = {}
        self.means_ = {}
        self.covs_ = {}

        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / n
            self.means_[c] = np.mean(Xc, axis=0)
            cov = np.cov(Xc, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-9
            self.covs_[c] = cov
        return self

    def _log_gaussian(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        diff = X - mean
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return np.full(len(X), -np.inf)
        inv_cov = np.linalg.inv(cov)
        mahal = np.einsum('ni,ij,nj->n', diff, inv_cov, diff)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)

    def predict_log_posterior(self, X: np.ndarray) -> np.ndarray:
        log_posts = np.column_stack([
            self._log_gaussian(X, self.means_[c], self.covs_[c]) + np.log(self.priors_[c])
            for c in self.classes_
        ])
        return log_posts

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_posts = self.predict_log_posterior(X)
        log_posts -= log_posts.max(axis=1, keepdims=True)
        probs = np.exp(log_posts)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_posts = self.predict_log_posterior(X)
        return self.classes_[np.argmax(log_posts, axis=1)]


class KNNClassifier:
    """K-Nearest Neighbors."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        # Distâncias euclidianas: broadcast (n_test, n_train)
        diff  = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        k_idx = np.argsort(dists, axis=1)[:, :self.k]
        k_labels = self.y_train[k_idx]           # (n_test, k)
        # Voto majoritário
        preds = np.array([
            np.bincount(row, minlength=np.max(self.y_train) + 1).argmax()
            for row in k_labels
        ])
        return preds


class DMCClassifier:
    """Distância ao Centroide de Mínima Distância (DMC)."""

    def fit(self, X, y):
        self.classes_   = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        # Distâncias euclidianas a cada centroide
        diff  = X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        return self.classes_[np.argmin(dists, axis=1)]


N_REALIZATIONS = 20
TEST_SIZE      = 0.2

acc_bayes, acc_knn, acc_dmc = [], [], []
realizations = []

print("Executando realizações...")
for seed in range(N_REALIZATIONS):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    # Gaussian Bayes
    bayes = GaussianBayesClassifier().fit(X_tr, y_tr)
    yp_b  = bayes.predict(X_te)
    a_b   = accuracy_score(y_te, yp_b)
    acc_bayes.append(a_b)

    # KNN
    knn  = KNNClassifier(k=5).fit(X_tr, y_tr)
    yp_k = knn.predict(X_te)
    a_k  = accuracy_score(y_te, yp_k)
    acc_knn.append(a_k)

    # DMC
    dmc  = DMCClassifier().fit(X_tr, y_tr)
    yp_d = dmc.predict(X_te)
    a_d  = accuracy_score(y_te, yp_d)
    acc_dmc.append(a_d)

    realizations.append({
        'seed': seed,
        'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te,
        'yp_b': yp_b, 'yp_k': yp_k, 'yp_d': yp_d,
        'a_b': a_b,   'a_k': a_k,   'a_d': a_d,
        'bayes': bayes
    })
    print(f"  [{seed:02d}] Bayes={a_b:.4f}  KNN={a_k:.4f}  DMC={a_d:.4f}")



print("\n" + "=" * 62)
print("RESULTADOS – 20 REALIZAÇÕES  (80% treino / 20% teste, estratificado)")
print("=" * 62)
print(f"{'Classificador':<25} {'Acurácia Média':>16} {'Desvio Padrão':>14}")
print("-" * 62)
print(f"{'Gaussiano Bayesiano':<25} {np.mean(acc_bayes):>16.4f} {np.std(acc_bayes):>14.4f}")
print(f"{'KNN (k=5)':<25} {np.mean(acc_knn):>16.4f} {np.std(acc_knn):>14.4f}")
print(f"{'DMC':<25} {np.mean(acc_dmc):>16.4f} {np.std(acc_dmc):>14.4f}")
print("=" * 62)

# Escolha da realização: mais próxima da acurácia média do classificador Bayes
mean_b = np.mean(acc_bayes)
chosen_idx = int(np.argmin(np.abs(np.array(acc_bayes) - mean_b)))
chosen = realizations[chosen_idx]

print(f"\nRealização escolhida: semente {chosen_idx}")
print(f"  Acurácia Bayes  = {chosen['a_b']:.4f}  (média = {mean_b:.4f})")
print(f"  Acurácia KNN    = {chosen['a_k']:.4f}")
print(f"  Acurácia DMC    = {chosen['a_d']:.4f}")
print("  Justificativa: realização cuja acurácia do classificador Bayesiano")
print("  é mais próxima da acurácia média das 20 realizações (mais representativa).")


label_names = [CLASS_NAMES[c] for c in CLASSES]
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle(
    f'Matrizes de Confusão – Realização {chosen_idx}  '
    f'(acurácia Bayes={chosen["a_b"]:.4f}, mais próxima da média)',
    fontsize=13, fontweight='bold'
)

for ax, yp, clf_name in zip(
    axes,
    [chosen['yp_b'], chosen['yp_k'], chosen['yp_d']],
    ['Gaussiano Bayesiano', 'KNN (k=5)', 'DMC']
):
    cm = confusion_matrix(chosen['y_te'], yp, labels=CLASSES)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax,
                linewidths=0.5, linecolor='gray')
    acc = accuracy_score(chosen['y_te'], yp)
    ax.set_title(f'{clf_name}\nAcurácia = {acc:.4f}', fontsize=11)
    ax.set_xlabel('Classe predita', fontsize=10)
    ax.set_ylabel('Classe real', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(N_REALIZATIONS)
ax.plot(x, acc_bayes, 'o-', label=f'Gaussiano Bayesiano  (μ={np.mean(acc_bayes):.4f}, σ={np.std(acc_bayes):.4f})',
        color='royalblue', linewidth=1.5, markersize=6)
ax.plot(x, acc_knn,   's-', label=f'KNN (k=5)             (μ={np.mean(acc_knn):.4f}, σ={np.std(acc_knn):.4f})',
        color='darkorange', linewidth=1.5, markersize=6)
ax.plot(x, acc_dmc,   '^-', label=f'DMC                   (μ={np.mean(acc_dmc):.4f}, σ={np.std(acc_dmc):.4f})',
        color='green', linewidth=1.5, markersize=6)
ax.axhline(np.mean(acc_bayes), color='royalblue',  linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(np.mean(acc_knn),   color='darkorange', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(np.mean(acc_dmc),   color='green',      linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(chosen_idx, color='gray', linestyle=':', linewidth=1.5, label=f'Realização escolhida ({chosen_idx})')
ax.set_xlabel('Realização (semente)')
ax.set_ylabel('Acurácia')
ax.set_title('Acurácia por Realização – 20 Realizações')
ax.set_xticks(x)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


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


# PCA na realização escolhida
pca = PCA(n_components=2)
pca.fit(chosen['X_tr'])
X_tr_2d = pca.transform(chosen['X_tr'])
X_te_2d = pca.transform(chosen['X_te'])

# Projeção da covariância: Σ_2d = W^T Σ W   (W = loadings, shape d×2)
W = pca.components_.T    # (34, 2)

colors_tab = plt.cm.tab10(np.linspace(0, 0.7, len(CLASSES)))

fig, ax = plt.subplots(figsize=(13, 9))

legend_handles = []
for i, c in enumerate(CLASSES):
    col = colors_tab[i]

    # --- Pontos de treino ---
    mask_tr = chosen['y_tr'] == c
    ax.scatter(X_tr_2d[mask_tr, 0], X_tr_2d[mask_tr, 1],
               c=[col], marker='o', s=30, alpha=0.5, edgecolors='none')

    # --- Pontos de teste ---
    mask_te = chosen['y_te'] == c
    ax.scatter(X_te_2d[mask_te, 0], X_te_2d[mask_te, 1],
               c=[col], marker='^', s=65, alpha=0.9,
               edgecolors='black', linewidths=0.7)

    # --- Gaussiana (média + elipses 1σ e 2σ) ---
    mean_2d = W.T @ chosen['bayes'].means_[c]          # (2,)
    cov_2d  = W.T @ chosen['bayes'].covs_[c] @ W       # (2, 2)
    plot_covariance_ellipse(ax, mean_2d, cov_2d, col, n_std=1.0, alpha=0.25, lw=2.0)
    plot_covariance_ellipse(ax, mean_2d, cov_2d, col, n_std=2.0, alpha=0.12, lw=1.5)
    ax.scatter(*mean_2d, c=[col], marker='*', s=250,
               edgecolors='black', linewidths=0.8, zorder=6)

    patch = mpatches.Patch(color=col, label=CLASS_NAMES[c])
    legend_handles.append(patch)

# Legenda de símbolos
legend_handles += [
    plt.Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=7, alpha=0.7, label='Treino (○)'),
    plt.Line2D([0], [0], marker='^', color='gray', linestyle='None',
               markersize=8, markeredgecolor='black', label='Teste (△)'),
    plt.Line2D([0], [0], marker='*', color='gray', linestyle='None',
               markersize=12, markeredgecolor='black', label='Centroide (★)'),
    mpatches.Patch(facecolor='gray', alpha=0.2, label='Elipse 1σ / 2σ'),
]

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100
ax.set_xlabel(f'CP1 ({var1:.1f}% variância)', fontsize=12)
ax.set_ylabel(f'CP2 ({var2:.1f}% variância)', fontsize=12)
ax.set_title(
    f'Gaussianas por Classe – Projeção PCA 2D  (Realização {chosen_idx})\n'
    f'Variância acumulada: {var1+var2:.1f}%   |   '
    f'Acurácia Bayes = {chosen["a_b"]:.4f}',
    fontsize=12
)
ax.legend(handles=legend_handles, loc='upper right',
          fontsize=8, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('gaussian_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFiguras salvas:")
print("  confusion_matrices.png")
print("  accuracy_comparison.png")
print("  gaussian_plot.png")
