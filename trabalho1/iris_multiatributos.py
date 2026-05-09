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
n_atributos = X.shape[1]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── FUNÇÕES ───────────────────────────────────────────────────────────────────

def calcular_priors(y_treino, nomes_classes):
    """
    P(C) = n_amostras_da_classe / n_total_treino
    Retorna dict {classe_id: float}.
    """
    n = len(y_treino)
    return {c: np.sum(y_treino == c) / n for c in range(len(nomes_classes))}


def calcular_parametros(X_treino, y_treino, n_classes, n_atributos):
    """
    Calcula vetor de médias μ e matriz de covariância Σ para cada classe.
    Σ_ij = (1/n) * Σ_k (xk_i - μ_i)(xk_j - μ_j)
    Retorna medias[c] (array 1D) e covariancias[c] (matriz d x d).
    """
    medias = {}
    covariancias = {}
    for c in range(n_classes):
        X_c = X_treino[y_treino == c]
        n = len(X_c)
        mu = np.array([sum(X_c[:, j]) / n for j in range(n_atributos)])
        sigma = np.zeros((n_atributos, n_atributos))
        for k in range(n):
            diff = X_c[k] - mu
            sigma += np.outer(diff, diff)
        sigma /= n
        medias[c] = mu
        covariancias[c] = sigma
    return medias, covariancias


def gaussiana_multivariada_pdf(x, mu, sigma):
    """
    f(x) = 1/((2π)^(d/2) * |Σ|^(1/2)) * exp(-1/2 * (x-μ)^T Σ^{-1} (x-μ))
    Retorna float.
    """
    d = len(mu)
    diff = x - mu
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    coef = 1.0 / ((2 * math.pi) ** (d / 2) * math.sqrt(det))
    expoente = -0.5 * (diff @ inv @ diff)
    return coef * math.exp(expoente)


def calcular_posteriori(x_amostra, classe_id, medias, covariancias, priors):
    """
    P(C|x) = P(x|C) * P(C) / Σ_Ci [P(x|Ci) * P(Ci)]
    P(x|C) = Gaussiana Multivariada(x; μ_C, Σ_C)
    Retorna float (número entre 0 e 1).
    """
    def verossimilhanca(c):
        return gaussiana_multivariada_pdf(x_amostra, medias[c], covariancias[c]) * priors[c]

    numerador = verossimilhanca(classe_id)
    denominador = sum(verossimilhanca(c) for c in priors)
    return numerador / denominador


# ── CONJUNTO TOTAL ────────────────────────────────────────────────────────────

print("=" * 60)
print("CONJUNTO DE TREINO")
print(f"Total de amostras no treino: {len(X_treino)}")
print(f"Atributos: {list(nomes_atributos)}")
print()

for c in range(3):
    X_c = X_treino[y_treino == c]
    n = len(X_c)
    print("=" * 60)
    print(f"CLASSE: {nomes_classes[c].upper()} ({n} amostras)")
    print("-" * 60)
    for j, nome in enumerate(nomes_atributos):
        valores = X_c[:, j]
        mu = sum(valores) / n
        sigma2 = sum((xi - mu) ** 2 for xi in valores) / n
        print(f"  {nome}:")
        print(f"    Média:     {mu:.4f}")
        print(f"    Variância: {sigma2:.4f}")
    print()

# ── DIVISÃO TREINO / TESTE ────────────────────────────────────────────────────

print("=" * 60)
print("DIVISÃO TREINO / TESTE")
print(f"  Treino: {len(X_treino)} amostras")
print(f"  Teste:  {len(X_teste)} amostras")
print()

# ── PROBABILIDADES A PRIORI ───────────────────────────────────────────────────

priors = calcular_priors(y_treino, nomes_classes)

print("=" * 60)
print("PROBABILIDADES A PRIORI (calculadas no treino)")
print("-" * 60)
n_treino = len(y_treino)
print(f"Total de amostras no treino: {n_treino}")
for c, prior in priors.items():
    n_c = int(prior * n_treino)
    print(f"  P({nomes_classes[c]}) = {n_c}/{n_treino} = {prior:.4f}")
print()

# ── PARÂMETROS DO TREINO ──────────────────────────────────────────────────────

medias, covariancias = calcular_parametros(X_treino, y_treino, 3, n_atributos)

print("=" * 60)
print("PARÂMETROS DO TREINO (vetor μ e matriz de covariância Σ)")
print()
for c in range(3):
    print(f"  CLASSE: {nomes_classes[c].upper()} ({int(priors[c]*n_treino)} amostras no treino)")
    print(f"  Vetor de médias μ:")
    for j, nome in enumerate(nomes_atributos):
        print(f"    {nome}: {medias[c][j]:.4f}")
    print(f"  Matriz de Covariância Σ ({n_atributos}x{n_atributos}):")
    header = "         " + "".join(f"  {nomes_atributos[j][:8]:>10s}" for j in range(n_atributos))
    print(header)
    for i in range(n_atributos):
        linha = f"    {nomes_atributos[i][:8]:<8s}" + "".join(f"  {covariancias[c][i, j]:10.4f}" for j in range(n_atributos))
        print(linha)
    print()

# ── PROBABILIDADE A POSTERIORI ────────────────────────────────────────────────
# P(C|x) = P(x|C) * P(C) / Σ_Ci [P(x|Ci) * P(Ci)]
# P(x|C) = Gaussiana Multivariada(x; μ_C, Σ_C)

print("=" * 60)
print("PROBABILIDADE A POSTERIORI — Gaussiana Multivariada (4 atributos)")
print("-" * 60)

acertos = 0
for i, (x_amostra, y_real) in enumerate(zip(X_teste, y_teste)):
    posteriors = {c: calcular_posteriori(x_amostra, c, medias, covariancias, priors) for c in range(3)}
    pred = max(posteriors, key=posteriors.get)
    acerto = pred == y_real
    if acerto:
        acertos += 1
    print(f"  Amostra {i+1:3d} | Real: {nomes_classes[y_real]:<12s} | "
          f"Pred: {nomes_classes[pred]:<12s} | "
          f"P(setosa|x)={posteriors[0]:.3f}  "
          f"P(versicolor|x)={posteriors[1]:.3f}  "
          f"P(virginica|x)={posteriors[2]:.3f}  "
          f"{'OK' if acerto else 'ERRO'}")

acuracia = acertos / len(y_teste)
print()
print(f"  Acertos: {acertos}/{len(y_teste)}  —  Acurácia: {acuracia:.4f} ({acuracia*100:.1f}%)")
print()

# ── CURVAS GAUSSIANAS MARGINAIS por atributo e por classe ─────────────────────
# A marginal da gaussiana multivariada no atributo j é: f(xj) = N(xj ; μ_j, Σ_jj)

cores = ['blue', 'orange', 'green']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for j, nome in enumerate(nomes_atributos):
    ax = axes[j]
    for c in range(3):
        mu_j = medias[c][j]
        sigma2_j = covariancias[c][j, j]   # diagonal = variância marginal
        x = np.linspace(mu_j - 4 * sigma2_j ** 0.5, mu_j + 4 * sigma2_j ** 0.5, 300)
        g = (1 / (2 * np.pi * sigma2_j) ** 0.5) * np.exp(-((x - mu_j) ** 2) / (2 * sigma2_j))
        ax.plot(x, g, color=cores[c],
                label=f'{nomes_classes[c]} (μ={mu_j:.2f}, Σ_jj={sigma2_j:.3f})')
        ax.fill_between(x, g, alpha=0.15, color=cores[c])
    ax.set_title(nome)
    ax.set_xlabel('Valor (cm)')
    ax.set_ylabel('Densidade marginal')
    ax.legend(fontsize=8)

plt.suptitle('Gaussiana Multivariada — Marginais por Atributo e Classe (Treino)', fontsize=13)
plt.tight_layout()
plt.savefig('iris_gaussianas.png', dpi=150)
plt.show()
print("Gráfico salvo em iris_gaussianas.png")
