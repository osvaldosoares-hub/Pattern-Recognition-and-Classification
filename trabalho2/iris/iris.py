
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from itertools import combinations



class BayesianoGaussiano:
   

    def fit(self, X: np.ndarray, y: np.ndarray):
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


class KNN:
    """K-Nearest Neighbors com distância Euclidiana."""

    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            dists   = np.linalg.norm(self.X_train_ - x, axis=1)
            idx_knn = np.argsort(dists)[: self.k]
            classes, counts = np.unique(self.y_train_[idx_knn], return_counts=True)
            preds.append(classes[counts.argmax()])
        return np.array(preds)


class DMC:
    """Classificador por Distância Mínima ao Centróide (Nearest Centroid)."""

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_   = np.unique(y)
        self.centroids_ = {c: X[y == c].mean(axis=0) for c in self.classes_}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            dists = {c: np.linalg.norm(x - mu)
                     for c, mu in self.centroids_.items()}
            preds.append(min(dists, key=dists.get))
        return np.array(preds)


def split_estratificado(X, y, frac_teste=0.2, seed=None):
    """Divisão treino/teste estratificada."""
    rng     = np.random.default_rng(seed)
    idx_tr, idx_te = [], []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_te  = max(1, int(len(idx_c) * frac_teste))
        idx_te.extend(idx_c[:n_te])
        idx_tr.extend(idx_c[n_te:])
    return (X[idx_tr], y[idx_tr],
            X[idx_te], y[idx_te])


def avaliar(X, y, n_real=20, frac_teste=0.2, k_knn=3):
   
    resultados = {"Bayesiano": [], "KNN": [], "DMC": []}
    splits     = {"Bayesiano": None, "KNN": None, "DMC": None}
    best_acc   = {"Bayesiano": -1,   "KNN": -1,   "DMC": -1}

    for seed in range(n_real):
        Xtr, ytr, Xte, yte = split_estratificado(X, y, frac_teste, seed)

        modelos = {
            "Bayesiano": BayesianoGaussiano(),
            "KNN":       KNN(k=k_knn),
            "DMC":       DMC(),
        }
        for nome, modelo in modelos.items():
            modelo.fit(Xtr, ytr)
            pred = modelo.predict(Xte)
            acc  = (pred == yte).mean()
            resultados[nome].append(acc)
            if acc > best_acc[nome]:
                best_acc[nome] = acc
                splits[nome]   = (Xtr, ytr, Xte, yte, pred, seed)

    return resultados, splits

def matriz_confusao(y_true, y_pred, classes):
    n = len(classes)
    mc = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        mc[t, p] += 1
    return mc


def plot_matriz_confusao(mc, classes, titulo="Matriz de Confusão", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mc, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    thresh = mc.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, mc[i, j],
                    ha="center", va="center",
                    color="white" if mc[i, j] > thresh else "black",
                    fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax)


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


def plot_gaussianas(X, y, Xtr, ytr, Xte, yte, pred_te,
                    feat_i, feat_j, nomes_feat, nomes_classes,
                    titulo="Gaussianas por Classe"):
    
    classes   = np.unique(ytr)
    colors    = plt.cm.tab10(np.linspace(0, 0.7, len(classes)))
    color_map = {c: colors[i] for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(9, 7))

    # Treina Bayesiano nas 2 features selecionadas
    modelo_2d = BayesianoGaussiano()
    Xtr_2d    = Xtr[:, [feat_i, feat_j]]
    modelo_2d.fit(Xtr_2d, ytr)

    # Elipses e centroide por classe
    for c in classes:
        col    = color_map[c]
        mu_2d  = modelo_2d.media_[c]
        cov_2d = modelo_2d.cov_[c] + np.eye(2) * 1e-9
        plot_covariance_ellipse(ax, mu_2d, cov_2d, col, n_std=1.0, alpha=0.25, lw=2.0)
        plot_covariance_ellipse(ax, mu_2d, cov_2d, col, n_std=2.0, alpha=0.12, lw=1.5)
        ax.scatter(*mu_2d, c=[col], marker='*', s=250,
                   edgecolors='black', linewidths=0.8, zorder=6)

    # Pontos de treino
    for c in classes:
        col  = color_map[c]
        name = nomes_classes[c]
        mask = ytr == c
        ax.scatter(Xtr[mask, feat_i], Xtr[mask, feat_j],
                   c=[col], marker='o', s=30, alpha=0.5,
                   label=f"Treino — {name}", edgecolors='none')

    # Pontos de teste (△ = acerto, ✕ = erro)
    acerto_full = pred_te == yte
    for c in classes:
        col  = color_map[c]
        name = nomes_classes[c]
        mask        = yte == c
        mask_acerto = mask & acerto_full
        mask_erro   = mask & ~acerto_full
        ax.scatter(Xte[mask_acerto, feat_i], Xte[mask_acerto, feat_j],
                   c=[col], marker='^', s=65, alpha=0.9,
                   edgecolors='black', linewidths=0.7, zorder=5,
                   label=f"Teste acerto — {name}")
        if mask_erro.any():
            ax.scatter(Xte[mask_erro, feat_i], Xte[mask_erro, feat_j],
                       c=[col], marker='X', s=90, alpha=0.9,
                       edgecolors='red', linewidths=1.5, zorder=6,
                       label=f"Teste erro — {name}")

    ax.set_xlabel(nomes_feat[feat_i], fontsize=12)
    ax.set_ylabel(nomes_feat[feat_j], fontsize=12)
    ax.set_title(titulo, fontsize=13)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # ── Carregar Iris ──────────────────────────────────────────────────────
    iris     = load_iris()
    X, y     = iris.data, iris.target
    nomes_c  = iris.target_names          # ['setosa', 'versicolor', 'virginica']
    nomes_f  = iris.feature_names         # 4 features

    N_REAL     = 20
    FRAC_TESTE = 0.2
    K_KNN      = 3

    print("=" * 60)
    print("   CLASSIFICADORES – DATASET IRIS")
    print(f"   {N_REAL} realizações | {int(FRAC_TESTE*100)}% teste | KNN k={K_KNN}")
    print("=" * 60)

    # ── 8.1 Avaliação com 20 realizações ──────────────────────────────────
    resultados, splits = avaliar(X, y, N_REAL, FRAC_TESTE, K_KNN)

    # ── 8.2 Resumo estatístico ────────────────────────────────────────────
    print(f"\n{'Classificador':<15} {'Média Acc':>10} {'Desvio Padrão':>14}")
    print("-" * 42)
    for nome in ["Bayesiano", "KNN", "DMC"]:
        accs = resultados[nome]
        print(f"{nome:<15} {np.mean(accs)*100:>9.2f}%  {np.std(accs)*100:>12.2f}%")


    fig_box, ax_box = plt.subplots(figsize=(7, 5))
    dados_box = [resultados[n] for n in ["Bayesiano", "KNN", "DMC"]]
    bp = ax_box.boxplot(dados_box, patch_artist=True,
                        labels=["Bayesiano\nGaussiano", f"KNN\n(k={K_KNN})", "DMC"],
                        widths=0.5)
    cores_box = ["#4C9BE8", "#E87C4C", "#4CE87C"]
    for patch, cor in zip(bp["boxes"], cores_box):
        patch.set_facecolor(cor)
    ax_box.set_ylabel("Acurácia", fontsize=12)
    ax_box.set_title(f"Distribuição de Acurácia – {N_REAL} Realizações (Iris)", fontsize=13)
    ax_box.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax_box.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_box.savefig("boxplot_acuracia.png", dpi=150)

    accs_bayes = np.array(resultados["Bayesiano"])
    mediana    = np.median(accs_bayes)
    seed_repr  = int(np.argmin(np.abs(accs_bayes - mediana)))

    print(f"\nRealização escolhida para a matriz de confusão: seed={seed_repr}")
    print(f"  → Critério: acurácia mais próxima da mediana ({mediana*100:.2f}%)")
    print(f"  → Acurácia nessa realização: {accs_bayes[seed_repr]*100:.2f}%")

    # Re-treina com a semente escolhida
    Xtr, ytr, Xte, yte = split_estratificado(X, y, FRAC_TESTE, seed_repr)
    modelo_repr = BayesianoGaussiano().fit(Xtr, ytr)
    pred_repr   = modelo_repr.predict(Xte)

    mc = matriz_confusao(yte, pred_repr, classes=np.unique(y))

    print(f"\nMatriz de Confusão (Bayesiano Gaussiano, seed={seed_repr}):")
    header = "         " + "  ".join(f"{nomes_c[c]:>10}" for c in np.unique(y))
    print(header)
    for i, c in enumerate(np.unique(y)):
        row = f"{nomes_c[c]:>8} " + "  ".join(f"{mc[i,j]:>10}" for j in range(len(np.unique(y))))
        print(row)

    # Plot matriz de confusão
    fig_mc, ax_mc = plt.subplots(figsize=(6, 5))
    plot_matriz_confusao(mc, nomes_c,
                         titulo=f"Matriz de Confusão – Bayesiano Gaussiano\n"
                                f"(seed={seed_repr}, acc={accs_bayes[seed_repr]*100:.1f}%)",
                         ax=ax_mc)
    plt.tight_layout()
    fig_mc.savefig("matriz_confusao.png", dpi=150)

    # ── 8.5 Matrizes de confusão dos 3 classificadores (seed_repr) ───────
    fig_mcs, axes_mc = plt.subplots(1, 3, figsize=(16, 5))
    nomes_clf = ["Bayesiano", "KNN", "DMC"]
    modelos_repr = {
        "Bayesiano": BayesianoGaussiano(),
        "KNN":       KNN(k=K_KNN),
        "DMC":       DMC(),
    }
    for ax_m, nome in zip(axes_mc, nomes_clf):
        modelos_repr[nome].fit(Xtr, ytr)
        pred_m = modelos_repr[nome].predict(Xte)
        acc_m  = (pred_m == yte).mean()
        mc_m   = matriz_confusao(yte, pred_m, classes=np.unique(y))
        plot_matriz_confusao(mc_m, nomes_c,
                             titulo=f"{nome}\n(seed={seed_repr}, acc={acc_m*100:.1f}%)",
                             ax=ax_m)
    plt.suptitle("Matrizes de Confusão – Comparação dos Classificadores", fontsize=13)
    plt.tight_layout()
    fig_mcs.savefig("matrizes_confusao_comparacao.png", dpi=150)

    FEAT_I, FEAT_J = 2, 3  # petal length, petal width

    fig_gauss = plot_gaussianas(
        X, y,
        Xtr, ytr, Xte, yte, pred_repr,
        feat_i=FEAT_I, feat_j=FEAT_J,
        nomes_feat=nomes_f,
        nomes_classes=nomes_c,
        titulo=(f"Gaussianas Bayesianas por Classe – Iris\n"
                f"Atributos: {nomes_f[FEAT_I]} × {nomes_f[FEAT_J]}\n"
                f"(seed={seed_repr}, acc={accs_bayes[seed_repr]*100:.1f}%)")
    )
    fig_gauss.savefig("gaussianas_iris.png", dpi=150)

    # ── 8.7 Gráfico de barras comparando médias ───────────────────────────
    fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
    nomes_plot = ["Bayesiano\nGaussiano", f"KNN\n(k={K_KNN})", "DMC"]
    medias  = [np.mean(resultados[n]) * 100 for n in ["Bayesiano", "KNN", "DMC"]]
    desvios = [np.std(resultados[n])  * 100 for n in ["Bayesiano", "KNN", "DMC"]]
    x_pos   = np.arange(len(nomes_plot))
    bars    = ax_bar.bar(x_pos, medias, yerr=desvios, capsize=8,
                         color=cores_box, edgecolor="black", width=0.5,
                         error_kw={"elinewidth": 2, "ecolor": "black"})
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(nomes_plot, fontsize=11)
    ax_bar.set_ylabel("Acurácia (%)", fontsize=12)
    ax_bar.set_ylim(0, 110)
    ax_bar.set_title(f"Acurácia Média ± Desvio Padrão – {N_REAL} Realizações (Iris)",
                     fontsize=12)
    for bar, med, std in zip(bars, medias, desvios):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 1.5,
                    f"{med:.2f}%\n±{std:.2f}%",
                    ha="center", va="bottom", fontsize=9)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_bar.savefig("comparacao_acuracia.png", dpi=150)

    plt.show()
    print("\nFiguras salvas: boxplot_acuracia.png, matriz_confusao.png,")
    print("  matrizes_confusao_comparacao.png, gaussianas_iris.png,")
    print("  comparacao_acuracia.png")
