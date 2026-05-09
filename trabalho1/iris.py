import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
nomes_classes = iris.target_names
nomes_atributos = iris.feature_names

# Atributo selecionado (0=sepal length, 1=sepal width, 2=petal length, 3=petal width)
atributo_idx = 2
atributo_nome = nomes_atributos[atributo_idx]

print("=" * 60)
print("CONJUNTO TOTAL")
print(f"Total de amostras: {len(X)}")
print(f"Atributo analisado: {atributo_nome}")
print()


for classe_id in range(3):
    X_classe = X[y == classe_id]
    valores = X_classe[:, atributo_idx]
    n = len(valores)
    media = sum(valores) / n
    variancia = sum((xi - media) ** 2 for xi in valores) / n

    print("=" * 60)
    print(f"CLASSE: {nomes_classes[classe_id].upper()} ({n} amostras)")
    print("-" * 60)
    print(f"  {atributo_nome}:")
    print(f"    Média:     {media:.4f}")
    print(f"    Variância: {variancia:.4f}")
    print()


# divisão de treino (80%) / teste (20%)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("DIVISÃO TREINO / TESTE")
print(f"  Treino: {len(X_treino)} amostras")
print(f"  Teste:  {len(X_teste)} amostras")
print()


# ── FUNÇÕES ──────────────────────────────────────────────────────────────────

def calcular_priors(y_treino, nomes_classes):
    """
    Calcula a probabilidade a priori de cada classe.
    P(C) = n_amostras_da_classe / n_total_treino
    Retorna dict {classe_id: prior}.
    """
    n_treino = len(y_treino)
    priors = {}
    for classe_id in range(len(nomes_classes)):
        n_classe = np.sum(y_treino == classe_id)
        priors[classe_id] = n_classe / n_treino
    return priors


def gaussiana_pdf(x, mu, sigma2):
    """
    Densidade gaussiana: f(x) = (1/sqrt(2πσ²)) * exp(-(x-μ)² / (2σ²))
    """
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-((x - mu) ** 2) / (2 * sigma2))


def calcular_posteriori(x_val, classe_id, medias, variancias, priors):
    """
    Calcula a probabilidade a posteriori de uma classe específica dado x.
    P(C|x) = P(x|C) * P(C) / Σ P(x|Ci)*P(Ci)
    Retorna um float (número entre 0 e 1).
    """
    numerador = gaussiana_pdf(x_val, medias[classe_id], variancias[classe_id]) * priors[classe_id]
    denominador = sum(
        gaussiana_pdf(x_val, medias[c], variancias[c]) * priors[c] for c in priors
    )
    return numerador / denominador


# ─────────────────────────────────────────────────────────────────────────────

# probabilidade a priori P(C1), P(C2), P(C3) - do treino

print("=" * 60)
print("PROBABILIDADES A PRIORI (calculadas no treino)")
print("-" * 60)

n_treino = len(y_treino)
priors = calcular_priors(y_treino, nomes_classes)
print(f"Total de amostras no treino: {n_treino}")
for classe_id, prior in priors.items():
    n_classe = int(prior * n_treino)
    print(f"  P({nomes_classes[classe_id]}) = {n_classe}/{n_treino} = {prior:.4f}")

print()


# media e variancia no conjunto de TREINO (atributo único)

print("=" * 60)
print(f"PARÂMETROS DO TREINO — atributo: {atributo_nome}")
print()

medias = {}
variancias = {}

for classe_id in range(3):
    X_classe_treino = X_treino[y_treino == classe_id][:, atributo_idx]
    n = len(X_classe_treino)
    mu = sum(X_classe_treino) / n
    sigma2 = sum((xi - mu) ** 2 for xi in X_classe_treino) / n
    medias[classe_id] = mu
    variancias[classe_id] = sigma2

    print(f"  CLASSE: {nomes_classes[classe_id].upper()} ({n} amostras no treino)")
    print(f"    Média:     {mu:.4f}")
    print(f"    Variância: {sigma2:.4f}")
    print()


# ── PROBABILIDADE A POSTERIORI ───────────────────────────────────────────────
# P(C|x) = P(x|C) * P(C) / Σ P(x|Ci)*P(Ci)
# P(x|C) = Gaussiana(x; μ_C, σ²_C)

print("=" * 60)
print(f"PROBABILIDADE A POSTERIORI — atributo: {atributo_nome}")
print("-" * 60)

acertos = 0
for i, (x_val, y_real) in enumerate(zip(X_teste[:, atributo_idx], y_teste)):
    posteriors = {c: calcular_posteriori(x_val, c, medias, variancias, priors) for c in range(3)}
    pred = max(posteriors, key=posteriors.get)
    acerto = pred == y_real
    if acerto:
        acertos += 1
        print(f"  Amostra {i+1:3d} | x={x_val:.2f} | Real: {nomes_classes[y_real]:<12s} | "
          f"Pred: {nomes_classes[pred]:<12s} | "
          f"P(setosa|x)={posteriors[0]:.3f}  "
          f"P(versicolor|x)={posteriors[1]:.3f}  "
          f"P(virginica|x)={posteriors[2]:.3f}  "
          f"{'OK' if acerto else 'ERRO'}") 

acuracia = acertos / len(y_teste)
print()
print(f"  Acertos: {acertos}/{len(y_teste)}  —  Acurácia: {acuracia:.4f} ({acuracia*100:.1f}%)")

print()


# GRÁFICOS - Separação das 3 classes

cores = ['blue', 'orange', 'green']

# Gráfico 1: Petala Length x Petala Width (melhor separação)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for classe_id in range(3):
    mask = y_treino == classe_id
    ax.scatter(X_treino[mask, 2], X_treino[mask, 3],
               c=cores[classe_id], label=nomes_classes[classe_id], alpha=0.7)
ax.set_xlabel('Petala Length (cm)')
ax.set_ylabel('Petala Width (cm)')
ax.set_title('Treino - Petala Length x Petala Width')
ax.legend()

# Gráfico 2: Sepala Length x Sepala Width

ax = axes[1]
for classe_id in range(3):
    mask = y_treino == classe_id
    ax.scatter(X_treino[mask, 0], X_treino[mask, 1],
               c=cores[classe_id], label=nomes_classes[classe_id], alpha=0.7)
ax.set_xlabel('Sepala Length (cm)')
ax.set_ylabel('Sepala Width (cm)')
ax.set_title('Treino - Sepala Length x Sepala Width')
ax.legend()

plt.tight_layout()
plt.savefig('iris_separacao.png', dpi=150)
plt.show()
print("Gráfico salvo em iris_separacao.png")


# curvas gaussianas - por atributo e por classe
""" 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for j, atributo in enumerate(nomes_atributos):
    ax = axes[j]

    for classe_id in range(3):
        mu = medias[classe_id][j]
        sigma2 = variancias[classe_id][j]

        # Gerar pontos no eixo x
        x = np.linspace(mu - 4 * sigma2**0.5, mu + 4 * sigma2**0.5, 200)

        # Fórmula da gaussiana: f(x) = (1 / sqrt(2π σ²)) * exp(-(x-μ)² / 2σ²)
        gaussiana = (1 / (2 * np.pi * sigma2)**0.5) * np.exp(-((x - mu) ** 2) / (2 * sigma2))

        ax.plot(x, gaussiana, color=cores[classe_id],
                label=f'{nomes_classes[classe_id]} (μ={mu:.2f}, σ²={sigma2:.4f})')
        ax.fill_between(x, gaussiana, alpha=0.15, color=cores[classe_id])

    ax.set_title(atributo)
    ax.set_xlabel('Valor')
    ax.set_ylabel('Densidade')
    ax.legend(fontsize=8)

plt.suptitle('Curvas Gaussianas por Atributo e Classe (Treino)', fontsize=14)
plt.tight_layout()
plt.savefig('iris_gaussianas.png', dpi=150)
plt.show()
print("Gráfico salvo em iris_gaussianas.png") """
