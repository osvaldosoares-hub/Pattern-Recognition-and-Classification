# Trabalho 4 — Discriminantes Lineares (LDA) e Quadráticos (QDA)

## Objetivo
Unificar os trabalhos anteriores implementando o **Classificador Bayesiano Gaussiano sob 4 hipóteses distintas** de estrutura da matriz de covariância, e comparar os discriminantes resultantes.

## Entrada de Dados
- **Artificial I**: gerado sinteticamente via NumPy — 3 classes gaussianas 2D (100 amostras cada, total 300)
- **Iris**: `sklearn.datasets.load_iris()` — 150 amostras, 4 atributos, 3 classes
- **Breast Cancer (WDBC)**: arquivo `.csv` do UCI — 569 amostras, 30 atributos, 2 classes
- **Dermatology**: arquivo `.data` do UCI — 366 amostras, 34 atributos, 6 classes
- **Coluna Vertebral (3C)**: arquivo `.arff` — 310 amostras, 6 atributos biomecânicos, 3 classes

### Pré-processamento
- **Normalização z-score** aplicada a **todos os datasets** (estimada apenas no treino)
- **Dermatology**: valores ausentes na coluna "idade" substituídos pela **média** calculada antes do particionamento
- Divisão **80% treino / 20% teste** estratificada por classe
- **20 realizações independentes** com sementes 0 a 19

## Treinamento
Para **todos os 4 casos**, estimam-se os vetores de médias μ̂ᵢ e as matrizes de covariância Σ̂ᵢ por MLE:

1. **Vetor de médias**: μ̂ᵢ = (1/nᵢ) Σₖ∈Cᵢ xₖ
2. **Matriz de covariância por classe**: Σ̂ᵢ = (1/nᵢ) Σₖ∈Cᵢ (xₖ - μ̂ᵢ)(xₖ - μ̂ᵢ)ᵀ
3. **Priors**: P(Cᵢ) = nᵢ / N_total
4. **Regularização de Tikhonov**: adiciona εI à diagonal (ε = 10⁻⁶) antes da inversão

A diferença entre os casos está na **estrutura imposta** à matriz Σ:

## Parâmetros e Hiperparâmetros

| Tipo | Nome | Valor | Descrição |
|:----:|:----:|:-----:|:----------|
| **Hiperparâmetro** | `test_size` | 0.2 | Proporção dos dados para teste |
| **Hiperparâmetro** | `random_state` | 0 a 19 | 20 sementes para 20 realizações independentes |
| **Hiperparâmetro** | `α` (alpha) | 0.5 | Peso da interpolação no QDA Pooled (α·Σᵢ + (1-α)·Σ) |
| **Hiperparâmetro** | `λ` (lambda) | 10⁻⁶ | Regularização de Tikhonov (Σ + λI) |
| **Hiperparâmetro (KNN)** | `k` | 5 | Nº de vizinhos mais próximos |
| **Hiperparâmetro (KNN/DMC)** | `distância` | Euclidiana | Métrica de distância |
| **Parâmetro (aprendido)** | P(Cᵢ) | — | Probabilidade a priori de cada classe |
| **Parâmetro (aprendido)** | μᵢ | — | Vetor de médias d-dimensional da classe i |
| **Parâmetro (aprendido)** | Σᵢ (QDA Full) | — | Matriz de covariância completa d×d por classe |
| **Parâmetro (aprendido)** | Σ (LDA) | — | Matriz de covariância pooled compartilhada |
| **Parâmetro (aprendido)** | diag(Σᵢ) (GNB) | — | Matriz diagonal (variâncias) por classe |
| **Parâmetro (aprendido)** | σ²ᵢ (Isotrópica) | — | Variância escalar por classe |

**Número de parâmetros de covariância por caso:**

| Caso | Modelo | Parâmetros de covariância | Exemplo Iris (d=4, c=3) |
|:----:|:------:|:-------------------------:|:-----------------------:|
| 1 | QDA Full | c · d(d+1)/2 | 3 × 10 = 30 |
| 2 | LDA | d(d+1)/2 | 10 |
| 3 | QDA Diagonal (GNB) | c · d | 3 × 4 = 12 |
| 4 | QDA Isotrópica (DMC) | c | 3 |

Além disso, cada caso adiciona c médias (c·d) e c priors.


## Os 4 Casos de Covariância

| Caso | Modelo | Estrutura de Σ | Fronteira | Parâmetros de covariância |
|:----:|:------:|:--------------:|:---------:|:-------------------------:|
| 1 | **QDA Full** | Σᵢ completa e distinta por classe | Quadrática geral | c · d(d+1)/2 |
| 2 | **LDA** | Σ pooled compartilhada (Σ) | Linear (hiperplanos) | d(d+1)/2 |
| 3 | **QDA Diagonal (GNB)** | diag(Σᵢ) — diagonal por classe | Quadrática alinhada aos eixos | c · d |
| 4 | **QDA Isotrópica (DMC)** | σ²ᵢI — esférica por classe | Esférica/circular | c |

### Caso 1 — QDA Full (Σᵢ individual)
Mantém a função discriminante completa:
```
gᵢ(x) = log P(Cᵢ) - ½ log|Σᵢ| - ½ (x - μᵢ)ᵀ Σᵢ⁻¹ (x - μᵢ)
ŷ = argmaxᵢ gᵢ(x)
```
A fronteira de decisão entre duas classes é uma **superfície quádrica** (elipse, hipérbole, parábola).

### Caso 2 — LDA (Σ pooled)
Impondo Σᵢ = Σ para todo i, os termos quadráticos se cancelam:
```
Σ = (1/n) Σᵢ nᵢ Σ̂ᵢ     (covariância pooled)
gᵢ(x) = xᵀ Σ⁻¹ μᵢ - ½ μᵢᵀ Σ⁻¹ μᵢ + log P(Cᵢ)
```
A fronteira de decisão é um **hiperplano** (linear).

### Caso 3 — QDA Diagonal (GNB)
Σᵢ = diag(σ²ᵢ₁, σ²ᵢ₂, ..., σ²ᵢₔ). A verossimilhança fatora como produto de univariadas:
```
p(x|Cᵢ) = Πⱼ [1 / √(2πσ²ᵢⱼ) · exp(-(xⱼ - μᵢⱼ)² / (2σ²ᵢⱼ))]
```
Equivalente ao **Naive Bayes Gaussiano** do Trabalho 3.

### Caso 4 — QDA Isotrópica (DMC)
Σᵢ = σ²ᵢI. A distância de Mahalanobis reduz-se à Euclidiana:
```
gᵢ(x) = -‖x - μᵢ‖² / (2σ²ᵢ) - (d/2) log(2πσ²ᵢ) + log P(Cᵢ)
```
Quando os priors são iguais, equivale ao **DMC**.

### Variações Adicionais

**QDA Pooled (α=0.5)**: interpolação entre Σ própria e Σ pooled:
```
Σᵢ' = α · Σᵢ + (1 - α) · Σ
```

**QDA Regularized**: regularização de Tikhonov na diagonal:
```
Σᵢ' = Σᵢ + λI
```

## Relação com os trabalhos anteriores

| Trabalho | Modelo | Caso no T4 |
|----------|--------|:----------:|
| T1 — Iris | Gaussiana Univ/Bi/Multivariada | QDA Full (Caso 1) |
| T2 — GBC | Σ completa por classe | QDA Full (Caso 1) |
| T3 — GNB | Σ diagonal (Naive Bayes) | QDA Diagonal (Caso 3) |
| — | DMC (Distância Mínima) | QDA Isotrópica (Caso 4) |

## Resultados principais (acurácia média %)

| Dataset | Melhor | Tipo | Acurácia |
|---------|:------:|:----:|:--------:|
| Artificial I | **QDA Isotrópica** (Caso 4) | Bayesiano | 96,42% |
| Breast Cancer | **KNN (k=5)** | Não-param. | 97,06% |
| Dermatology | **QDA Pooled α=0.5** | Bayesiano | 96,81% |
| Íris | **LDA** (Caso 2) | Bayesiano | 97,67% |
| Coluna Vertebral | **QDA Full** (Caso 1) | Bayesiano | 84,27% |

### Destaques por dataset

| Dataset | QDA Full | LDA | QDA Diag (GNB) | QDA Isotr (DMC) | QDA Pooled | KNN |
|---------|:--------:|:---:|:--------------:|:---------------:|:----------:|:---:|
| Artificial I | 96,25 | 96,25 | 96,25 | **96,42** | 96,25 | 95,58 |
| Breast Cancer | 94,39 | 96,58 | 94,39 | 92,68 | 95,13 | **97,06** |
| Dermatology | 87,22 | 96,25 | 87,22 | 96,39 | **96,81** | 95,49 |
| Íris | 94,33 | **97,67** | 94,33 | 84,50 | 97,50 | 94,67 |
| Coluna Vertebral | **84,27** | 80,48 | 83,23 | 77,74 | 80,97 | 79,03 |

## Conclusão
- **QDA Full** é melhor quando as classes têm covariâncias distintas (Coluna Vertebral), mas sofre overfitting em alta dimensão (Dermatology: 87,22%)
- **LDA** é excelente em datasets com separação aproximadamente linear (Íris: 97,67%, Breast Cancer: 96,58%)
- **QDA Diagonal (GNB)** ignora correlações — equivalente ao Naive Bayes
- **QDA Isotrópica (DMC)** — o mais restritivo, bom apenas quando classes são esféricas
- **QDA Pooled (α=0.5)** — melhor compromisso, venceu no Dermatology (96,81%)
- **KNN** foi competitivo em todos, vencendo no Breast Cancer
- A **escolha da estrutura de covariância** depende do nº de amostras, dimensionalidade e homogeneidade entre classes

## Gráficos gerados
- Matrizes de confusão para datasets 2D
- Superfícies de decisão
- Distribuições gaussianas por classe (projeção PCA 2D para dados multidimensionais)