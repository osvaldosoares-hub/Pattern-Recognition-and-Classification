# Trabalho 3 — Classificador Naive Bayes Gaussiano (GNB)

## Objetivo
Substituir o **GBC** (matriz de covariância completa por classe) pelo **Naive Bayes Gaussiano (GNB)**, que assume **independência condicional** entre os atributos, fatorando a verossimilhança como produto de densidades univariadas. Comparar com **KNN** e **DMC** nos mesmos 5 datasets.

## Entrada de Dados
- **Artificial I**: gerado sinteticamente via NumPy com semente fixa (SEED=42) — 3 classes gaussianas 2D (100 amostras cada, total 300)
- **Iris**: `sklearn.datasets.load_iris()` — 150 amostras, 4 atributos, 3 classes
- **Breast Cancer (WDBC)**: arquivo `.csv` do UCI — 569 amostras, 30 atributos, 2 classes
- **Dermatology**: arquivo `.data` do UCI — 366 amostras, 34 atributos clínicos (excluindo idade), 6 classes
- **Coluna Vertebral (3C)**: arquivo `.arff` — 310 amostras, 6 atributos biomecânicos, 3 classes

### Pré-processamento
- **Normalização z-score** aplicada a **todos os datasets** (diferente do T2, que aplicava só ao WDBC)
- **Dermatology**: valores ausentes na coluna "idade" substituídos pela **média global** da coluna
- Divisão **80% treino / 20% teste** estratificada por classe
- **20 realizações independentes** com sementes 0 a 19

## Treinamento
### GNB (Naive Bayes Gaussiano)
Para cada classe, estima-se diretamente das amostras de treino:
1. **Priors**: P(Cᵢ) = nᵢ / N_total
2. **Média de cada atributo** por classe: μᵢⱼ = (1/nᵢ) Σₖ xₖⱼ
3. **Variância de cada atributo** por classe: σ²ᵢⱼ = (1/nᵢ) Σₖ (xₖⱼ - μᵢⱼ)² + ε
   (ε = 10⁻⁹ adicionado ao desvio padrão para estabilidade numérica)
4. A **matriz de covariância é diagonal**: Σᵢ = diag(σ²ᵢ₁, σ²ᵢ₂, ..., σ²ᵢₔ) — assume independência entre atributos

### KNN e DMC
- Mesmo procedimento do Trabalho 2, porém com k = 5 fixo para todos os datasets

## Parâmetros e Hiperparâmetros

| Tipo | Nome | Valor | Descrição |
|:----:|:----:|:-----:|:----------|
| **Hiperparâmetro** | `test_size` | 0.2 | Proporção dos dados para teste |
| **Hiperparâmetro** | `random_state` | 0 a 19 | 20 sementes para 20 realizações independentes |
| **Hiperparâmetro (GNB)** | `ε` (epsilon) | 10⁻⁹ | Termo de estabilidade adicionado ao desvio padrão |
| **Hiperparâmetro (KNN)** | `k` | 5 (fixo) | Nº de vizinhos mais próximos |
| **Hiperparâmetro (KNN/DMC)** | `distância` | Euclidiana | Métrica de distância |
| **Parâmetro (aprendido)** | P(Cᵢ) | — | Probabilidade a priori de cada classe |
| **Parâmetro (aprendido)** | μᵢⱼ | — | Média do atributo j para a classe i |
| **Parâmetro (aprendido)** | σ²ᵢⱼ | — | Variância do atributo j para a classe i |

**Total de parâmetros por classe (GNB):** 2d + 1 (apenas diagonal!)
- Exemplo Iris (d=4): 2×4 + 1 = **9 parâmetros por classe** → 27 no total
- Exemplo WDBC (d=30): 2×30 + 1 = **61 parâmetros por classe** → 122 no total (2 classes)

**Comparação GNB vs GBC (T2) em número de parâmetros:**

| Dataset | d | GBC (Σ completa) | GNB (Σ diagonal) | Redução |
|---------|:-:|:----------------:|:----------------:|:-------:|
| Iris | 4 | 15 parâmetros/classe | 9 parâmetros/classe | 40% |
| WDBC | 30 | 496 parâmetros/classe | 61 parâmetros/classe | 87,7% |
| Dermatology | 34 | 630 parâmetros/classe | 69 parâmetros/classe | 89% |

## Fórmulas Principais

### Teorema de Bayes com Independência Condicional (GNB)
```
P(Cᵢ|x) = P(Cᵢ) · Πⱼ p(xⱼ|Cᵢ) / p(x)
```

### Gaussiana Univariada para cada atributo j (usada no GNB)
```
p(xⱼ|Cᵢ) = 1 / √(2πσ²ᵢⱼ) · exp(-(xⱼ - μᵢⱼ)² / (2σ²ᵢⱼ))
```

### Regra de Decisão no Domínio Logarítmico (evita underflow numérico)
```
ŷ = argmaxᵢ [ log P(Cᵢ) + Σⱼ log p(xⱼ|Cᵢ) ]
```

### Log-verossimilhança Gaussiana Univariada
```
log p(xⱼ|Cᵢ) = -½ log(2πσ²ᵢⱼ) - (xⱼ - μᵢⱼ)² / (2σ²ᵢⱼ)
```

### Estimação dos Parâmetros (MLE)
```
μ̂ᵢⱼ = (1/nᵢ) Σₖ∈Cᵢ xₖⱼ
σ̂ᵢⱼ = √((1/nᵢ) Σₖ∈Cᵢ (xₖⱼ - μ̂ᵢⱼ)²) + ε
```

### Distância Euclidiana (KNN e DMC) — mesma do T2
```
d(x, xⱼ) = ‖x - xⱼ‖₂ = √(Σₗ (xₗ - xⱼₗ)²)
```

## Diferenças metodológicas em relação ao Trabalho 2

| Aspecto | Trabalho 2 (GBC) | Trabalho 3 (GNB) |
|---------|:----------------:|:-----------------:|
| Classificador principal | Gaussiana Multivariada (Σ completa) | Naive Bayes (Σ diagonal) |
| Nº de parâmetros por classe | d(d+1)/2 + d + 1 | 2d + 1 |
| Normalização z-score | Apenas WDBC | **Todos os datasets** |
| Dataset Artificial I | 120 amostras (3×40) | 300 amostras (3×100) |
| KNN | k variável (3 ou 5) | k = 5 fixo |

## Resultados principais (acurácia média ± DP)

| Dataset | GNB | KNN (k=5) | DMC |
|---------|:---:|:---------:|:---:|
| Artificial I | **95,50±3,17** | 94,42±2,95 | 95,17±3,29 |
| Iris | 94,33±4,36 | **95,17±3,24** | 83,50±3,57 |
| Breast Cancer | 92,70±2,09 | **96,15±1,43** | 92,43±1,89 |
| Dermatology | 87,25±2,73 | 95,99±2,28 | **96,48±2,46** |
| Coluna Vertebral 3C | **82,34±3,40** | 79,84±5,19 | 76,61±3,70 |

## Conclusão
- **GNB** é competitivo quando a independência entre atributos é razoável (Artificial I, Coluna Vertebral)
- **GNB sofre** em datasets com atributos correlacionados (Breast Cancer) ou discretos (Dermatology)
- **DMC surpreende** no Dermatology (96,48%), indicando classes bem separadas e esféricas
- **KNN** é o mais consistente, vencendo em Iris e Breast Cancer
- Nenhum classificador domina todos os cenários — a escolha depende da estrutura do dataset

## Gráficos gerados
- Matrizes de confusão para cada classificador em cada dataset
- Distribuições gaussianas univariadas por classe
- Superfícies de decisão para datasets 2D (Artificial I, Iris com petal length × petal width)