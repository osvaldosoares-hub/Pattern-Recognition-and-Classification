import math
import numpy as np
import matplotlib.pyplot as plt
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

n_classes   = len(nomes_classes)
n_atributos = X.shape[1]



def gaussiana_pdf(x, mu, sigma2):
    """f(x) = (1/√(2πσ²)) · exp(-(x−μ)²/(2σ²))"""
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(
        -((x - mu) ** 2) / (2 * sigma2)
    )


def treinar(X_tr, y_tr):
    """Estima P(C), μ_cj e σ²_cj a partir do conjunto de treino."""
    n = len(y_tr)
    priors    = {}
    medias    = {}
    variancias = {}
    for c in range(n_classes):
        X_c = X_tr[y_tr == c]
        priors[c]     = len(X_c) / n
        medias[c]     = {j: float(X_c[:, j].mean()) for j in range(n_atributos)}
        variancias[c] = {j: float(X_c[:, j].var())  for j in range(n_atributos)}
    return priors, medias, variancias


def classificar_atributo(x, j, medias, variancias, priors):
    """MAP usando apenas o atributo j."""
    nums = {c: gaussiana_pdf(x[j], medias[c][j], variancias[c][j]) * priors[c]
            for c in priors}
    denom = sum(nums.values())
    post  = {c: nums[c] / denom for c in priors}
    return max(post, key=post.get), post


# ─── Treino ──────────────────────────────────────────────────────────────────

priors, medias, variancias = treinar(X_treino, y_treino)

# ─── Cabeçalho ───────────────────────────────────────────────────────────────

print("=" * 70)
print("   CLASSIFICADOR GAUSSIANO UNIVARIADO — DATASET IRIS")
print("=" * 70)
print(f"  Treino : {len(X_treino)} amostras   Teste : {len(X_teste)} amostras")
print()

# ─── Análise Exploratória ────────────────────────────────────────────────────

print("=" * 70)
print("  ANÁLISE EXPLORATÓRIA — μ e σ² por classe e atributo  (treino)")
print("=" * 70)
for c in range(n_classes):
    print(f"\n  Classe : {nomes_classes[c].upper()}   Prior P(C)={priors[c]:.4f}")
    for j in range(n_atributos):
        print(f"    [{j}] {nomes_atributos[j]:<30s}  μ={medias[c][j]:.4f}   σ²={variancias[c][j]:.4f}")
print()

# ─── Acurácia: um atributo por vez ───────────────────────────────────────────

print("=" * 70)
print("  CLASSIFICAÇÃO — um atributo por vez")
print("=" * 70)

for j in range(n_atributos):
    acertos = sum(
        classificar_atributo(x, j, medias, variancias, priors)[0] == y_real
        for x, y_real in zip(X_teste, y_teste)
    )
    acc = acertos / len(y_teste)
    
    print(f"  [{j}] {nomes_atributos[j]:<30s}  {acertos}/{len(y_teste)}  {acc*100:.1f}%")
print()

# ─── Gráfico 1: Curvas Gaussianas por atributo e classe ──────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for j in range(n_atributos):
    ax = axes[j]
    for c in range(n_classes):
        mu     = medias[c][j]
        sigma2 = variancias[c][j]
        sigma  = math.sqrt(sigma2)
        x_range = np.linspace(mu - 3.5 * sigma, mu + 3.5 * sigma, 400)
        gauss   = np.array([gaussiana_pdf(xi, mu, sigma2) for xi in x_range])

        ax.plot(x_range, gauss, color=cores[c], linewidth=2,
                label=f'{nomes_classes[c]}  μ={mu:.2f}  σ²={sigma2:.3f}')
        ax.fill_between(x_range, gauss, alpha=0.12, color=cores[c])

    # Histograma normalizado dos dados de treino
   

    ax.set_title(f'[{j}] {nomes_atributos[j]}\n ')
    ax.set_xlabel('Valor (cm)')
    ax.set_ylabel('Densidade de probabilidade')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Gaussianas Univariadas por Atributo e Classe — Iris',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_univariado_gaussianas.png', dpi=150)
plt.show()
print("Gráfico salvo: iris_univariado_gaussianas.png")

