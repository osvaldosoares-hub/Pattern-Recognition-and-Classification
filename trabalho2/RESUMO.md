# Trabalho 2 — Classificador Bayesiano Gaussiano Multivariado (GBC)

## Objetivo
Implementar e comparar o **Classificador Bayesiano Gaussiano Multivariado (GBC)** com **KNN** e **DMC** em **5 conjuntos de dados** de complexidade crescente.

## Entrada de Dados
- **Artificial I**: gerado sinteticamente via NumPy — 3 classes gaussianas 2D (40 amostras cada, total 120)
- **Iris**: `sklearn.datasets.load_iris()` — 150 amostras, 4 atributos, 3 classes
- **Breast Cancer (WDBC)**: arquivo `.csv` do UCI — 569 amostras, 30 atributos, 2 classes
- **Dermatology**: arquivo `.data` do UCI — 366 amostras, 35 atributos (34+idade), 6 classes
- **Coluna Vertebral (2C e 3C)**: arquivo `.arff` — 310 amostras, 6 atributos biomecânicos

### Pré-processamento
- **WDBC**: normalização z-score (média 0, variância 1) estimada **apenas no treino** para evitar data leakage
- **Dermatology**: valores ausentes na coluna "idade" substituídos pela **mediana do treino**
- Divisão **80% treino / 20% teste** estratificada por classe
- **20 realizações independentes** com sementes 0 a 19

## Treinamento
### GBC (Classificador Bayesiano Gaussiano)
Para cada classe, estima-se diretamente das amostras de treino:
1. **Priors**: P(Cᵢ) = nᵢ / N_total
2. **Vetor de médias**: μ̂ᵢ = (1/nᵢ) Σₖ xₖ   (para cada classe i)
3. **Matriz de covariância**: Σ̂ᵢ = (1/nᵢ) Σₖ (xₖ - μ̂ᵢ)(xₖ - μ̂ᵢ)ᵀ   (matriz d×d completa)

### KNN
- Sem treinamento explícito (método não-paramétrico)
- k = 3 para Iris, k = 5 para os demais
- Distância Euclidiana

### DMC
- Calcula centróide de cada classe: cᵢ = (1/nᵢ) Σₖ xₖ⁽ⁱ⁾
- Classifica pela menor distância Euclidiana ao centróide

## Parâmetros e Hiperparâmetros

| Tipo | Nome | Valor | Descrição |
|:----:|:----:|:-----:|:----------|
| **Hiperparâmetro** | `test_size` | 0.2 | Proporção dos dados para teste |
| **Hiperparâmetro** | `random_state` | 0 a 19 | 20 sementes para 20 realizações independentes |
| **Hiperparâmetro (KNN)** | `k` | 3 (Iris) / 5 (demais) | Nº de vizinhos mais próximos considerados |
| **Hiperparâmetro (KNN)** | `distância` | Euclidiana | Métrica de distância entre pontos |
| **Parâmetro (aprendido)** | P(Cᵢ) | — | Probabilidade a priori de cada classe |
| **Parâmetro (aprendido)** | μᵢ | — | Vetor de médias d-dimensional da classe i |
| **Parâmetro (aprendido)** | Σᵢ | — | Matriz de covariância d×d completa da classe i |

**Total de parâmetros por classe (GBC):** d(d+1)/2 + d + 1
- Exemplo Iris (d=4): 10 + 4 + 1 = **15 parâmetros por classe** → 45 no total
- Exemplo WDBC (d=30): 465 + 30 + 1 = **496 parâmetros por classe** → 992 no total (2 classes)

## Fórmulas Principais

### Gaussiana Multivariada (GBC)
```
p(x|Cᵢ) = 1 / ((2π)^(d/2) · |Σᵢ|^(1/2)) · exp(-½(x - μᵢ)ᵀ Σᵢ⁻¹ (x - μᵢ))
```

### Regra de Decisão de Bayes (MAP)
```
ŷ = argmaxᵢ P(Cᵢ|x) = argmaxᵢ p(x|Cᵢ) · P(Cᵢ)
```

### Distância de Mahalanobis (embutida na Gaussiana multivariada)
```
d_M²(x, μᵢ) = (x - μᵢ)ᵀ Σᵢ⁻¹ (x - μᵢ)
```

### Distância Euclidiana (KNN e DMC)
```
d(x, xⱼ) = ‖x - xⱼ‖₂ = √(Σₗ (xₗ - xⱼₗ)²)
```

### Acurácia média após 20 realizações
```
ā = (1/20) Σᵣ aᵣ      s = √((1/20) Σᵣ (aᵣ - ā)²)
```

## Classificadores comparados

| Classificador | Descrição |
|---------------|-----------|
| **GBC (Gaussiano Bayesiano)** | Modela cada classe com uma Gaussiana multivariada completa (matriz Σ própria por classe). Fronteiras de decisão quadráticas. |
| **KNN (k vizinhos)** | Método não-paramétrico: voto majoritário entre os k vizinhos mais próximos. |
| **DMC (Distância Mínima ao Centróide)** | Atribui à classe cujo centróide está mais próximo em distância Euclidiana. |

## Resultados principais (acurácia média ± DP)

| Dataset | GBC | KNN | DMC |
|---------|:---:|:---:|:---:|
| Artificial I | **99,58±0,99** | 99,58±0,99 | **99,86±0,61** |
| Iris | **96,50±3,07** | 95,67±3,18 | 90,83±3,78 |
| Breast Cancer | 94,91±1,79 | **96,27±1,52** | 92,68±1,45 |
| Dermatology | **89,66±2,81** | 86,55±3,93 | 53,72±5,02 |
| Coluna 2C | 80,97±4,69 | **83,31±3,36** | 74,76±5,13 |
| Coluna 3C | **83,23±3,93** | 82,42±4,44 | 75,56±5,38 |

## Conclusão
- **GBC** é robusto e competitivo, vencendo em 4 de 6 cenários
- **DMC** colapsa no Dermatology (53,72%), evidenciando limitação de fronteiras lineares
- **KNN** é superior em alta dimensão (WDBC, Coluna 2C)
- Dados sintéticos esféricos favorecem o DMC (99,86%)

## Gráficos gerados
- Matrizes de confusão para cada classificador em cada dataset
- Distribuições gaussianas por classe (projeção PCA 2D para dados multidimensionais)
- Superfícies de decisão (para Artificial I, 2D)