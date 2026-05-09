
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.stats import multivariate_normal
from collections import Counter


GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


def generate_artificial_I(n_per_class: int = 40, seed: int = None):
    """Gera o dataset Artificial I com 3 classes gaussianas bidimensionais."""
    rng = np.random.default_rng(seed)

    # Parâmetros das distribuições de cada classe
    params = [
        # (média,             covariância)
        ([1.5, 7.0], [[0.40, 0.00], [0.00, 0.40]]),   # Classe 0 – círculos
        ([6.5, 7.0], [[0.50, 0.10], [0.10, 0.30]]),   # Classe 1 – triângulos
        ([4.0, 3.5], [[0.40, 0.00], [0.00, 0.40]]),   # Classe 2 – estrelas
    ]

    Xs, ys = [], []
    for label, (mean, cov) in enumerate(params):
        Xs.append(rng.multivariate_normal(mean, cov, n_per_class))
        ys.append(np.full(n_per_class, label))

    return np.vstack(Xs), np.concatenate(ys)



def train_test_split(X, y, test_size: float = 0.20, seed: int = None):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_size)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, classes):
    K = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[cls_idx[t], cls_idx[p]] += 1
    return cm



class GaussianBayesClassifier:
    """
    Classificador Bayesiano Gaussiano Multivariado.

    Cada classe possui seu próprio vetor de média mu_c e matriz de covariância
    Sigma_c. A probabilidade a posteriori é calculada diretamente via Bayes:

        P(c | x) = p(x | c) * P(c) / p(x)

    onde p(x | c) é a densidade gaussiana multivariada:

        p(x | c) = (2*pi)^(-d/2) * |Sigma_c|^(-1/2)
                   * exp(-0.5 * (x-mu_c)^T * Sigma_c^-1 * (x-mu_c))

    e p(x) = sum_c p(x|c)*P(c)  (evidência — usada para normalização).
    Nenhuma simplificação (LDA/QDA, log-soma, classes equiprováveis, etc.) é feita.
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n = len(y)
        self.priors_: dict = {}
        self.means_:  dict = {}
        self.covs_:   dict = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / n                    # prior empírico
            self.means_[c]  = np.mean(Xc, axis=0)            # mu_c
            self.covs_[c]   = np.cov(Xc, rowvar=False)       # Sigma_c (ddof=1)
        return self

    # ------------------------------------------------------------------
    def _likelihood(self, x: np.ndarray, c) -> float:
        """Densidade gaussiana multivariada p(x | c)."""
        return float(multivariate_normal.pdf(
            x,
            mean=self.means_[c],
            cov=self.covs_[c],
            allow_singular=True,
        ))

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna a matriz de probabilidades a posteriori P(c|x) de forma (n, K).
        O cálculo é feito diretamente: numerador = p(x|c)*P(c),
        denominador = soma dos numeradores sobre todas as classes (evidência).
        """
        n, K = len(X), len(self.classes_)
        posteriors = np.zeros((n, K))

        for i, x in enumerate(X):
            for j, c in enumerate(self.classes_):
                posteriors[i, j] = self._likelihood(x, c) * self.priors_[c]

        # Normalização pela evidência p(x) para obter P(c|x) verdadeiro
        evidence = posteriors.sum(axis=1, keepdims=True)
        evidence = np.where(evidence == 0, 1e-300, evidence)   # estabilidade numérica
        return posteriors / evidence

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ----------------------------------------------------------------------

class KNNClassifier:
    """K-Nearest Neighbours com distância Euclidiana."""

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X, y):
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            dists = np.linalg.norm(self.X_train_ - x, axis=1)
            knn_labels = self.y_train_[np.argsort(dists)[: self.k]]
            preds.append(Counter(knn_labels).most_common(1)[0][0])
        return np.array(preds)


# ----------------------------------------------------------------------

class DMCClassifier:
    """Distância Mínima ao Centróide (Minimum Distance Classifier)."""

    def fit(self, X, y):
        self.classes_   = np.unique(y)
        self.centroids_ = {c: X[y == c].mean(axis=0) for c in self.classes_}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.array(
            [[np.linalg.norm(x - self.centroids_[c]) for c in self.classes_]
             for x in X]
        )
        return self.classes_[np.argmin(dists, axis=1)]


# =============================================================================
# 4. Experimento — 20 Realizações (hold-out 80 / 20)
# =============================================================================

N_REALIZATIONS = 20
N_PER_CLASS    = 40
CLASS_NAMES    = ['Círculos', 'Triângulos', 'Estrelas']
CLASSES        = np.array([0, 1, 2])

results: dict = {'Bayes': [], 'KNN': [], 'DMC': []}

# Armazena todas as realizações para selecionar a representativa após o loop
all_realizations: list = []

print("=" * 55)
print(f"{'Realiz.':>8}  {'Bayes':>8}  {'KNN':>8}  {'DMC':>8}")
print("-" * 55)

for i in range(N_REALIZATIONS):
    seed = i * 11 + 17   # seeds distintas e reproduzíveis

    X, y = generate_artificial_I(n_per_class=N_PER_CLASS, seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, seed=seed)

    classifiers = {
        'Bayes': GaussianBayesClassifier(),
        'KNN':   KNNClassifier(k=5),
        'DMC':   DMCClassifier(),
    }

    acc_row: dict = {}
    preds_row: dict = {}
    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        acc_row[name]   = accuracy(y_te, pred)
        preds_row[name] = pred
        results[name].append(acc_row[name])

    print(f"{i+1:>8}  {acc_row['Bayes']:>8.4f}  {acc_row['KNN']:>8.4f}  {acc_row['DMC']:>8.4f}")

    all_realizations.append({
        'acc':         acc_row['Bayes'],
        'seed':        seed,
        'realization': i + 1,
        'X_tr':        X_tr, 'X_te': X_te,
        'y_tr':        y_tr, 'y_te': y_te,
        'preds':       preds_row,
        'clf':         GaussianBayesClassifier().fit(X_tr, y_tr),
    })

print("=" * 55)
print(f"\n{'Classificador':<12}  {'Acurácia Média':>16}  {'Desvio Padrão':>14}")
print("-" * 46)
for name in ['Bayes', 'KNN', 'DMC']:
    m = np.mean(results[name])
    s = np.std(results[name])
    print(f"{name:<12}  {m:>16.4f}  {s:>14.4f}")

# Critério de escolha: realização cuja acurácia do Bayes é mais próxima da mediana
bayes_accs = np.array(results['Bayes'])
mediana    = np.median(bayes_accs)
med_idx    = int(np.argmin(np.abs(bayes_accs - mediana)))
best       = all_realizations[med_idx]

print(
    f"\n>>> Realização escolhida para análise: #{best['realization']} "
    f"(seed={best['seed']}, acc_Bayes={best['acc']:.4f})\n"
    f"    Justificativa: selecionada por apresentar a acurácia do\n"
    f"    classificador Bayesiano mais próxima da mediana das {N_REALIZATIONS}\n"
    f"    realizações ({mediana:.4f}) — caso mais representativo do comportamento\n"
    f"    típico do classificador."
)


# =============================================================================
# 5. Matriz de Confusão — Realização Escolhida
# =============================================================================

cm_bayes = confusion_matrix(best['y_te'], best['preds']['Bayes'], CLASSES)
cm_knn   = confusion_matrix(best['y_te'], best['preds']['KNN'],   CLASSES)
cm_dmc   = confusion_matrix(best['y_te'], best['preds']['DMC'],   CLASSES)

def plot_cm(ax, cm, title):
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha='right', fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel('Predito'); ax.set_ylabel('Real')
    ax.set_title(title, fontsize=9)
    for r in range(3):
        for c in range(3):
            ax.text(c, r, cm[r, c], ha='center', va='center', fontsize=12,
                    color='white' if cm[r, c] > cm.max() / 2 else 'black')
    return im

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
plot_cm(axes[0], cm_bayes, f'Bayes (acc={best["acc"]:.4f})')
plot_cm(axes[1], cm_knn,   f'KNN  (acc={accuracy(best["y_te"], best["preds"]["KNN"]):.4f})')
plot_cm(axes[2], cm_dmc,   f'DMC  (acc={accuracy(best["y_te"], best["preds"]["DMC"]):.4f})')
fig.suptitle(f'Matrizes de Confusão — Realização #{best["realization"]}', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# 6. Superfície de Decisão + Gaussianas + Dados Treino/Teste
# =============================================================================

MARKERS = ['o', '^', '*']
COLORS  = ['royalblue', 'darkorange', 'seagreen']
CMAP_BG = ListedColormap(['#AED6F1', '#FAD7A0', '#A9DFBF'])

clf_plot = best['clf']
X_tr = best['X_tr'];  y_tr = best['y_tr']
X_te = best['X_te'];  y_te = best['y_te']

# --- Grade para superfície de decisão ---
margin = 1.2
x1_min = min(X_tr[:, 0].min(), X_te[:, 0].min()) - margin
x1_max = max(X_tr[:, 0].max(), X_te[:, 0].max()) + margin
x2_min = min(X_tr[:, 1].min(), X_te[:, 1].min()) - margin
x2_max = max(X_tr[:, 1].max(), X_te[:, 1].max()) + margin

res = 300
xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, res),
    np.linspace(x2_min, x2_max, res),
)
grid = np.c_[xx1.ravel(), xx2.ravel()]
Z = clf_plot.predict(grid).reshape(xx1.shape)

fig, ax = plt.subplots(figsize=(8, 6))

# Fundo com regiões de decisão
ax.contourf(xx1, xx2, Z, alpha=0.30, cmap=CMAP_BG, levels=[-0.5, 0.5, 1.5, 2.5])
ax.contour( xx1, xx2, Z, colors='k', linewidths=0.8, alpha=0.55, levels=[0.5, 1.5])

# Contornos das gaussianas por classe
for c, color in zip(clf_plot.classes_, COLORS):
    mu  = clf_plot.means_[c]
    cov = clf_plot.covs_[c]
    rv  = multivariate_normal(mean=mu, cov=cov)
    sx  = np.sqrt(cov[0, 0]); sy = np.sqrt(cov[1, 1])
    gx  = np.linspace(mu[0] - 4 * sx, mu[0] + 4 * sx, 200)
    gy  = np.linspace(mu[1] - 4 * sy, mu[1] + 4 * sy, 200)
    Gx, Gy = np.meshgrid(gx, gy)
    Gpos = np.dstack((Gx, Gy))
    Gz   = rv.pdf(Gpos)
    ax.contour(Gx, Gy, Gz, levels=5, colors=color, linewidths=1.4, alpha=0.85)

# Pontos de treino
for c, mk, col in zip(CLASSES, MARKERS, COLORS):
    idx = y_tr == c
    ax.scatter(X_tr[idx, 0], X_tr[idx, 1],
               marker=mk, c=col, edgecolors='k', linewidths=0.6,
               s=70,  label=f'Treino – {CLASS_NAMES[c]}', zorder=3)

# Pontos de teste (borda vermelha espessa)
for c, mk, col in zip(CLASSES, MARKERS, COLORS):
    idx = y_te == c
    ax.scatter(X_te[idx, 0], X_te[idx, 1],
               marker=mk, c=col, edgecolors='crimson', linewidths=1.8,
               s=90, label=f'Teste  – {CLASS_NAMES[c]}', zorder=4)

ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title(
    f'Superfície de Decisão — Bayes Gaussiano\n'
    f'Realização #{best["realization"]}  |  '
    f'Par de atributos: ($x_1$, $x_2$)  |  '
    f'Contornos coloridos = gaussianas de cada classe\n'
    f'Borda preta = treino  •  Borda vermelha = teste',
    fontsize=9,
)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=3, fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.savefig('decision_surface.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# 7. Gráfico comparativo das acurácias (barras + box-plot)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# --- Barras por realização ---
ax = axes[0]
x = np.arange(N_REALIZATIONS)
w = 0.26
ax.bar(x - w, results['Bayes'], w, label='Bayes', color='steelblue',    alpha=0.85)
ax.bar(x,     results['KNN'],   w, label='KNN',   color='darkorange',   alpha=0.85)
ax.bar(x + w, results['DMC'],   w, label='DMC',   color='seagreen',     alpha=0.85)
ax.axhline(np.mean(results['Bayes']), color='steelblue',  ls='--', lw=1.2)
ax.axhline(np.mean(results['KNN']),   color='darkorange', ls='--', lw=1.2)
ax.axhline(np.mean(results['DMC']),   color='seagreen',   ls='--', lw=1.2)
ax.set_xlabel('Realização'); ax.set_ylabel('Acurácia')
ax.set_title('Acurácia por Realização — Artificial I')
ax.set_xticks(x); ax.set_xticklabels(x + 1, fontsize=7)
ax.set_ylim([0.0, 1.08]); ax.legend()

# --- Box-plot ---
ax = axes[1]
bp = ax.boxplot(
    [results['Bayes'], results['KNN'], results['DMC']],
    labels=['Bayes', 'KNN', 'DMC'],
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2),
)
for patch, color in zip(bp['boxes'], ['steelblue', 'darkorange', 'seagreen']):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_ylabel('Acurácia')
ax.set_title('Distribuição da Acurácia\n(20 realizações, hold-out 70/30)')
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFiguras salvas: confusion_matrices.png | decision_surface.png | accuracy_comparison.png")
