import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ─── Dados ───────────────────────────────────────────────────────────────────

iris = load_iris()
X = iris.data
y = iris.target
nomes_classes   = iris.target_names   # ['setosa', 'versicolor', 'virginica']
nomes_atributos = iris.feature_names  # sepal/petal length/width
cores = ['royalblue', 'darkorange', 'seagreen']

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

n_classes = len(nomes_classes)
pares = list(combinations(range(4), 2))   # 6 pares possíveis


def gaussiana_bivariada_pdf(x, mu, sigma):
    """
    PDF da Gaussiana bivariada:
        f(x) = 1/(2π·√|Σ|) · exp(-½ (x−μ)ᵀ Σ⁻¹ (x−μ))
    x, mu : arrays 1D de comprimento 2
    sigma : matriz 2×2 de covariância
    """
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    diff = x - mu
    exponent = -0.5 * diff @ inv @ diff
    return (1.0 / (2 * np.pi * np.sqrt(det))) * np.exp(exponent)


def gaussiana_bivariada_grid(xx1, xx2, mu, sigma):
    """
    Versão vetorizada para um meshgrid (evita loops Python).
    xx1, xx2 : arrays 2D de shape (H, W)
    Retorna  : array 2D (H, W) com os valores da PDF.
    """
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    pos  = np.stack([xx1, xx2], axis=-1)   # (H, W, 2)
    diff = pos - mu                         # (H, W, 2)
    # ∑_ij diff_i · inv_ij · diff_j  para cada ponto
    exponent = -0.5 * np.einsum('...i,ij,...j->...', diff, inv, diff)
    return (1.0 / (2 * np.pi * np.sqrt(det))) * np.exp(exponent)


def treinar_bivariado(X_tr, y_tr, idx1, idx2):
    """
    Estima P(C), μ_c (2D) e Σ_c (2×2) por classe para o par (idx1, idx2).
    """
    n = len(y_tr)
    priors = {}
    medias = {}
    covs   = {}
    for c in range(n_classes):
        X_c = X_tr[y_tr == c][:, [idx1, idx2]]
        priors[c] = len(X_c) / n
        medias[c] = X_c.mean(axis=0)
        covs[c]   = np.cov(X_c.T, bias=True)   # bias=True → divisor N
    return priors, medias, covs





def classificar_mle(x, idx1, idx2, medias, covs):
    """
    MLE — Máxima Verossimilhança:
        C* = argmax  P(x|C)
    Ignora os priors; escolhe a classe cuja Gaussiana
    atribui maior densidade ao ponto x.
    """
    x_2d = np.array([x[idx1], x[idx2]])
    likes = {
        c: gaussiana_bivariada_pdf(x_2d, medias[c], covs[c])
        for c in medias
    }
    return max(likes, key=likes.get), likes


def classificar_map(x, idx1, idx2, medias, covs, priors):
    """
   Probabilidade A Posteriori:
        P(C|x) = P(x|C) * P(C) / Σ P(x|Ci)*P(Ci)
    Retorna (classe_predita, dict com P(C|x) para cada classe).
    """
    x_2d = np.array([x[idx1], x[idx2]])
    # numerador de cada classe: verossimilhança × prior
    nums = {
        c: gaussiana_bivariada_pdf(x_2d, medias[c], covs[c]) * priors[c]
        for c in priors
    }
    # denominador: Σ P(x|Ci)*P(Ci)  — normaliza para obter probabilidade real
    denom = sum(nums.values())
    post  = {c: nums[c] / denom for c in priors}
    return max(post, key=post.get), post

print("  Classificador Gaussiano Bivariado — dataset IRIS")

print(f"  Treino : {len(X_treino)} amostras   Teste : {len(X_teste)} amostras")
print()

# ─── Análise Exploratória + Classificação por par ────────────────────────────

print("=" * 70)
print("  ANÁLISE EXPLORATÓRIA — μ (2D) e Σ (2×2) por classe  (treino)")
print("=" * 70)


acuracia = {}
todos_params  = {}

for (idx1, idx2) in pares:
    priors_p, medias_p, covs_p = treinar_bivariado(X_treino, y_treino, idx1, idx2)
    todos_params[(idx1, idx2)] = (priors_p, medias_p, covs_p)

    print(f"\n  Par [{idx1},{idx2}]  {nomes_atributos[idx1][:18]}  ×  {nomes_atributos[idx2][:18]}")
    for c in range(n_classes):
        mu  = medias_p[c]
        cov = covs_p[c]
        print(f"    {nomes_classes[c].upper():<12s}  μ=[{mu[0]:.3f}, {mu[1]:.3f}]  "
              f"Σ=[[{cov[0,0]:.4f}, {cov[0,1]:.4f}], [{cov[1,0]:.4f}, {cov[1,1]:.4f}]]")
  
    # MAP — argmax P(x|C)*P(C) / Σ P(x|Ci)*P(Ci)
    acertos_map = sum(
        classificar_map(x, idx1, idx2, medias_p, covs_p, priors_p)[0] == yr
        for x, yr in zip(X_teste, y_teste)
    )
   

    
    print(f" → MAP  — Acertos: {acertos_map}/{len(y_teste)} ")
   

print()


# ─── Gráfico 1: Contornos gaussianos bivariados (6 pares) ────────────────────

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()

for ax_idx, (idx1, idx2) in enumerate(pares):
    ax = axes[ax_idx]
    priors_p, medias_p, covs_p = todos_params[(idx1, idx2)]

    # Scatter dos dados de treino
    for c in range(n_classes):
        mask = y_treino == c
        ax.scatter(X_treino[mask, idx1], X_treino[mask, idx2],
                   c=cores[c], label=nomes_classes[c], alpha=0.55, s=22)

    # Grade para contornos
    margin = 0.4
    x1_vals = np.linspace(X[:, idx1].min() - margin, X[:, idx1].max() + margin, 120)
    x2_vals = np.linspace(X[:, idx2].min() - margin, X[:, idx2].max() + margin, 120)
    xx1, xx2 = np.meshgrid(x1_vals, x2_vals)

    for c in range(n_classes):
        zz = gaussiana_bivariada_grid(xx1, xx2, medias_p[c], covs_p[c])
        ax.contour(xx1, xx2, zz, levels=5, colors=[cores[c]], alpha=0.75, linewidths=1.3)
        # Ponto central (média)
        ax.plot(*medias_p[c], marker='x', ms=10, mew=2, color=cores[c])

    
    ax.set_xlabel(nomes_atributos[idx1], fontsize=9)
    ax.set_ylabel(nomes_atributos[idx2], fontsize=9)
    
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

plt.suptitle('Gaussianas Bivariadas — Contornos por Par de Atributos — Iris',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_bivariado_contornos.png', dpi=150)
plt.show()
print("Gráfico salvo: iris_bivariado_contornos.png")

