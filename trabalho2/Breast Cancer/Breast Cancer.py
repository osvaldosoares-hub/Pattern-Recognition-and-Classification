#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classificador Bayesiano Gaussiano Multivariado
Dataset: Breast Cancer Wisconsin (WDBC)
Comparação com KNN e DMC

- Vetores de média e matrizes de covariância específicos por classe
- Probabilidades a posteriori calculadas diretamente (sem simplificações)
- 20 realizações com 80/20 treino/teste estratificado
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

# ─── Reprodutibilidade ────────────────────────────────────────────────────────
np.random.seed(0)

# ─── Carregamento dos Dados ───────────────────────────────────────────────────
data_path = os.path.join('breast+cancer+wisconsin+diagnostic', 'wdbc.data')
df = pd.read_csv(data_path, header=None)

X_raw = df.iloc[:, 2:].values.astype(float)
y_raw = (df.iloc[:, 1] == 'M').astype(int).values   # M=1 (Maligno), B=0 (Benigno)

print("=" * 65)
print("Dataset: Breast Cancer Wisconsin Diagnostic (WDBC)")
print("=" * 65)
print(f"  Total de amostras : {X_raw.shape[0]}")
print(f"  Total de features : {X_raw.shape[1]}")
print(f"  Classe 0 (Benigno): {np.sum(y_raw == 0)} amostras")
print(f"  Classe 1 (Maligno): {np.sum(y_raw == 1)} amostras")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICADORES
# ═══════════════════════════════════════════════════════════════════════════════

class GaussianBayesClassifier:
    """
    Classificador Bayesiano Gaussiano Multivariado (sem simplificações).

    Para cada classe k:
        P(C_k | x) = P(x | C_k) * P(C_k) / P(x)

    onde P(x | C_k) é a densidade gaussiana multivariada com parâmetros
    específicos da classe k (μ_k, Σ_k).

    A decisão é: argmax_k [ log P(x | C_k) + log P(C_k) ]
    As probabilidades a posteriori são normalizadas numericamente para
    fornecer valores reais em [0, 1].
    """

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n = len(y)
        self.priors_ = {}
        self.means_  = {}
        self.covs_   = {}

        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / n
            self.means_[c]  = np.mean(Xc, axis=0)
            # Covariância completa por classe + regularização numérica
            cov = np.cov(Xc, rowvar=False)
            cov += np.eye(X.shape[1]) * 1e-9
            self.covs_[c] = cov
        return self

    def _log_likelihood(self, X, c):
        """
        Log-densidade gaussiana multivariada:
        log p(x | C_k) = -0.5 * [d*ln(2π) + ln|Σ_k| + (x-μ_k)^T Σ_k^{-1} (x-μ_k)]
        """
        mean = self.means_[c]
        cov  = self.covs_[c]
        d    = X.shape[1]

        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return np.full(len(X), -np.inf)

        inv_cov  = np.linalg.inv(cov)
        diff     = X - mean                                          # (N, d)
        mahal    = np.einsum('ni,ij,nj->n', diff, inv_cov, diff)    # Mahalanobis²

        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + mahal)

    def predict_proba(self, X):
        """Probabilidades a posteriori normalizadas P(C_k | x) para cada amostra."""
        log_posts = np.column_stack([
            self._log_likelihood(X, c) + np.log(self.priors_[c])
            for c in self.classes_
        ])
        # Estabilização numérica antes de aplicar exp
        log_posts -= log_posts.max(axis=1, keepdims=True)
        probs = np.exp(log_posts)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        log_posts = np.column_stack([
            self._log_likelihood(X, c) + np.log(self.priors_[c])
            for c in self.classes_
        ])
        return self.classes_[np.argmax(log_posts, axis=1)]


class DMCClassifier:
    """Classificador de Mínima Distância ao Centróide (DMC)."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_ = {c: np.mean(X[y == c], axis=0) for c in self.classes_}
        return self

    def predict(self, X):
        dists = np.stack(
            [np.linalg.norm(X - self.means_[c], axis=1) for c in self.classes_],
            axis=1
        )
        return self.classes_[np.argmin(dists, axis=1)]


class KNNClassifier:
    """Classificador K-Vizinhos Mais Próximos (KNN) — implementação própria."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        # Distância euclidiana vetorizada: (N_test, N_train)
        dists  = np.sqrt(((X[:, None, :] - self.X_train_[None, :, :]) ** 2).sum(axis=2))
        k_idx  = np.argsort(dists, axis=1)[:, :self.k]
        votes  = self.y_train_[k_idx]           # (N_test, k)
        return np.array([np.bincount(v, minlength=2).argmax() for v in votes])


# ═══════════════════════════════════════════════════════════════════════════════
#  20 REALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════
N_REAL = 20
K_KNN  = 5
results  = {'GBC': [], 'KNN': [], 'DMC': []}
all_runs = []

print(f"\nExecutando {N_REAL} realizações (80% treino / 20% teste, estratificado)...")

for seed in range(N_REAL):
    Xtr_r, Xte_r, ytr, yte = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=seed, stratify=y_raw
    )
    # Normalização: fit apenas no treino, aplica em treino e teste
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_r)
    Xte = scaler.transform(Xte_r)

    # ── GBC ──────────────────────────────────────────────────────────────────
    gbc = GaussianBayesClassifier().fit(Xtr, ytr)
    ypred_gbc = gbc.predict(Xte)
    acc_gbc   = np.mean(ypred_gbc == yte)

    # ── KNN ──────────────────────────────────────────────────────────────────
    knn = KNNClassifier(k=K_KNN).fit(Xtr, ytr)
    ypred_knn = knn.predict(Xte)
    acc_knn   = np.mean(ypred_knn == yte)

    # ── DMC ──────────────────────────────────────────────────────────────────
    dmc = DMCClassifier().fit(Xtr, ytr)
    ypred_dmc = dmc.predict(Xte)
    acc_dmc   = np.mean(ypred_dmc == yte)

    results['GBC'].append(acc_gbc)
    results['KNN'].append(acc_knn)
    results['DMC'].append(acc_dmc)
    all_runs.append((Xtr, Xte, ytr, yte, ypred_gbc, ypred_knn, ypred_dmc, gbc, scaler, seed))

    print(f"  Seed {seed:02d} → GBC={acc_gbc*100:.2f}%  KNN={acc_knn*100:.2f}%  DMC={acc_dmc*100:.2f}%")


# ─── Resumo ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESUMO — 20 Realizações")
print("=" * 65)
print(f"{'Classificador':<12} {'Acurácia Média':>16}  {'Desvio Padrão':>14}")
print("-" * 46)
for name in ['GBC', 'KNN', 'DMC']:
    accs = results[name]
    print(f"{name:<12} {np.mean(accs)*100:>15.4f}%  {np.std(accs)*100:>13.4f}%")
print("=" * 65)

# ─── Realização escolhida ─────────────────────────────────────────────────────
# Critério: seed cuja acurácia GBC é a mais próxima da média das 20 realizações.
# Essa escolha representa o desempenho "típico" do classificador, evitando
# tanto os melhores quanto os piores casos e tornando a análise representativa.
gbc_accs   = np.array(results['GBC'])
chosen_idx = int(np.argmin(np.abs(gbc_accs - np.mean(gbc_accs))))
Xtr, Xte, ytr, yte, ypred_gbc, ypred_knn, ypred_dmc, gbc_model, scaler_ch, seed_ch = all_runs[chosen_idx]

print(f"\nRealização escolhida para análise detalhada: #{chosen_idx} (seed={seed_ch})")
print(f"  GBC Acurácia: {gbc_accs[chosen_idx]*100:.4f}%")
print(f"  Média geral : {np.mean(gbc_accs)*100:.4f}%")
print("  Justificativa: realização cuja acurácia GBC se encontra mais")
print("  próxima da média das 20 realizações → caso mais representativo")
print("  do comportamento típico do classificador.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNÇÕES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════════

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


def confusion_matrix_np(y_true, y_pred, n_classes=2):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_conf_matrix(ax, y_true, y_pred, title):
    cm     = confusion_matrix_np(y_true, y_pred)
    acc    = np.diag(cm).sum() / cm.sum()
    thresh = cm.max() / 2.0

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Benigno', 'Maligno'], fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Benigno', 'Maligno'], fontsize=10)
    ax.set_xlabel(f'Predito — Acurácia: {acc*100:.2f}%', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return cm


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 1 — Comparação de Acurácias (20 realizações)
# ═══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle('Comparação de Classificadores — Breast Cancer Wisconsin\n'
              '20 Realizações (80/20 treino/teste estratificado)',
              fontsize=13, fontweight='bold')

clf_names  = ['GBC', 'KNN', 'DMC']
clf_labels = [f'GBC\n(Bayes Gaussiano)', f'KNN\n(k={K_KNN})', 'DMC\n(Mín. Distância)']
box_colors = ['#4CAF50', '#2196F3', '#FF9800']

# — Boxplot —
ax = axes1[0]
data_box = [np.array(results[n]) * 100 for n in clf_names]
bp = ax.boxplot(data_box, labels=clf_labels, patch_artist=True, notch=False,
                medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_ylabel('Acurácia (%)', fontsize=12)
ax.set_title('Boxplot das Acurácias', fontsize=11)
ax.set_ylim([82, 102])
ax.grid(True, alpha=0.3, axis='y')
for i, (n, accs) in enumerate(zip(clf_names, data_box), 1):
    ax.text(i, np.mean(accs) + 0.4, f'μ={np.mean(accs):.2f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333')

# — Barras —
ax = axes1[1]
means = [np.mean(results[n]) * 100 for n in clf_names]
stds  = [np.std(results[n])  * 100 for n in clf_names]
x     = np.arange(len(clf_names))
bars  = ax.bar(x, means, yerr=stds, capsize=10, color=box_colors,
               alpha=0.8, edgecolor='black', width=0.5,
               error_kw=dict(elinewidth=2, capthick=2))
ax.set_xticks(x)
ax.set_xticklabels(clf_labels, fontsize=10)
ax.set_ylabel('Acurácia Média (%)', fontsize=12)
ax.set_title('Acurácia Média ± Desvio Padrão', fontsize=11)
ax.set_ylim([82, 108])
ax.grid(True, alpha=0.3, axis='y')
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.5,
            f'{m:.2f}%\n±{s:.2f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('fig1_comparacao_classificadores.png', dpi=150, bbox_inches='tight')
print("Figura 1 salva: fig1_comparacao_classificadores.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 2 — Matrizes de Confusão (realização escolhida)
# ═══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle(
    f'Matrizes de Confusão — Realização #{chosen_idx} (seed={seed_ch})\n'
    f'Acurácia GBC desta realização: {gbc_accs[chosen_idx]*100:.2f}%  '
    f'(Média 20 realizações: {np.mean(gbc_accs)*100:.2f}%)',
    fontsize=11, fontweight='bold'
)

plot_conf_matrix(axes2[0], yte, ypred_gbc, f'GBC — Bayes Gaussiano')
plot_conf_matrix(axes2[1], yte, ypred_knn, f'KNN (k={K_KNN})')
plot_conf_matrix(axes2[2], yte, ypred_dmc, 'DMC — Mínima Distância')

plt.tight_layout()
plt.savefig('fig2_matrizes_confusao.png', dpi=150, bbox_inches='tight')
print("Figura 2 salva: fig2_matrizes_confusao.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 3 — Gaussianas sobrepostas aos dados (PCA 2D)
# ═══════════════════════════════════════════════════════════════════════════════
# Redução PCA → 2 componentes para visualização das gaussianas
pca    = PCA(n_components=2, random_state=42)
Xtr_2d = pca.fit_transform(Xtr)   # fit apenas no treino
Xte_2d = pca.transform(Xte)

var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
W = pca.components_.T  # (d, 2)

c_colors = {0: '#2196F3', 1: '#F44336'}   # azul / vermelho
c_labels = {0: 'Benigno (B)', 1: 'Maligno (M)'}

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle(
    f'Distribuições Gaussianas por Classe — Espaço PCA 2D\n'
    f'Realização #{chosen_idx} | PC1: {var_pc1:.1f}%  PC2: {var_pc2:.1f}% da variância total',
    fontsize=12, fontweight='bold'
)

panel_titles = ['Conjunto de Treinamento', 'Treinamento + Teste']

for ax_i, ax in enumerate(axes3):
    # ── Elipses de covariância 1σ e 2σ (gaussianas projetadas) ───────────────
    for c in [0, 1]:
        mu_proj  = pca.transform(gbc_model.means_[c].reshape(1, -1))[0]
        cov_proj = W.T @ gbc_model.covs_[c] @ W  # Σ_2D = W^T Σ W
        plot_covariance_ellipse(ax, mu_proj, cov_proj, c_colors[c], n_std=1.0, alpha=0.25, lw=2.0)
        plot_covariance_ellipse(ax, mu_proj, cov_proj, c_colors[c], n_std=2.0, alpha=0.12, lw=1.5)

    # ── Dados de treinamento ──────────────────────────────────────────────────
    for c in [0, 1]:
        mask = ytr == c
        ax.scatter(Xtr_2d[mask, 0], Xtr_2d[mask, 1],
                   c=c_colors[c], marker='o', s=30, alpha=0.5,
                   label=f'Treino — {c_labels[c]}', edgecolors='none')

    # ── Dados de teste (apenas no segundo painel) ─────────────────────────────
    if ax_i == 1:
        for c in [0, 1]:
            mask = yte == c
            ax.scatter(Xte_2d[mask, 0], Xte_2d[mask, 1],
                       c=c_colors[c], marker='^', s=65, alpha=0.9,
                       label=f'Teste — {c_labels[c]}',
                       edgecolors='black', linewidths=0.7)

    # ── Centróides ────────────────────────────────────────────────────────────
    for c in [0, 1]:
        mu_proj = pca.transform(gbc_model.means_[c].reshape(1, -1))[0]
        ax.scatter(*mu_proj, c=c_colors[c], marker='*', s=250,
                   edgecolors='black', linewidths=0.8, zorder=6,
                   label=f'Centróide — {c_labels[c]}' if ax_i == 0 else '_')

    ax.set_xlabel(f'PC1 ({var_pc1:.1f}% variância)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_pc2:.1f}% variância)', fontsize=11)
    ax.set_title(panel_titles[ax_i], fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('fig3_gaussianas_pca.png', dpi=150, bbox_inches='tight')
print("Figura 3 salva: fig3_gaussianas_pca.png")

plt.show()

# ─── Tabela final resumida ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TABELA RESUMO FINAL")
print("=" * 65)
print(f"{'Classificador':<20} {'Média (%)':>12}  {'Desvio (%)':>12}  {'Min (%)':>9}  {'Max (%)':>9}")
print("-" * 65)
for name in ['GBC', 'KNN', 'DMC']:
    accs = np.array(results[name]) * 100
    print(f"{name:<20} {np.mean(accs):>11.4f}  {np.std(accs):>12.4f}  "
          f"{np.min(accs):>8.4f}  {np.max(accs):>8.4f}")
print("=" * 65)
