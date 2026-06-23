# Trabalho 5 — Modelos de Misturas de Gaussianas (GMM-Bayes)

## Objetivo
**Generalizar** os classificadores dos trabalhos anteriores substituindo a Gaussiana única por classe por uma **Mistura de Gaussianas (GMM)**. Cada classe é modelada como uma combinação linear de **J componentes** Gaussianos, permitindo representar distribuições **multimodais** e fronteiras de decisão mais complexas.

## Entrada de Dados
- **Artificial I**: gerado sinteticamente — 3 classes gaussianas 2D (100 amostras cada, total 300)
- **Iris**: `sklearn.datasets.load_iris()` — 150 amostras, 4 atributos, 3 classes
- **Breast Cancer (WDBC)**: arquivo `.csv` do UCI — 569 amostras, 30 atributos, 2 classes
- **Dermatology**: arquivo `.data` do UCI — 366 amostras, 34 atributos, 6 classes
- **Coluna Vertebral (3C)**: arquivo `.arff` — 310 amostras, 6 atributos biomecânicos, 3 classes

### Pré-processamento
- **Normalização z-score** aplicada a **todos os datasets** (estimada apenas no treino)
- **Dermatology**: valores ausentes na coluna "idade" substituídos pela **média** calculada antes do particionamento
- Divisão **80% treino / 20% teste** estratificada por classe
- **20 realizações independentes** com sementes 0 a 19

## Treinamento — Algoritmo EM (Expectativa-Maximização)

Diferentemente dos trabalhos anteriores (que estimavam parâmetros diretamente), o GMM é treinado **iterativamente** via EM porque os rótulos dos componentes são **variáveis latentes** (não observamos de qual componente cada amostra veio).

Para cada classe, ajusta-se uma GMM com J=3 componentes:

### Passo E (Expectation)
Calcula a **responsabilidade** (probabilidade posterior) de cada amostra xₖ pertencer a cada componente j:
```
rₖⱼ = Pⱼ · N(xₖ; μⱼ, Σⱼ) / Σₗ Pₗ · N(xₖ; μₗ, Σₗ)
```

### Passo M (Maximization)
Reestima os parâmetros de cada componente j usando as responsabilidades:
```
Nⱼ = Σₖ rₖⱼ                          (peso efetivo do componente)

P̂ⱼ = Nⱼ / nᵢ                           (novo peso do componente)

μ̂ⱼ = (1/Nⱼ) Σₖ rₖⱼ · xₖ               (nova média)

Σ̂ⱼ = (1/Nⱼ) Σₖ rₖⱼ · (xₖ - μ̂ⱼ)(xₖ - μ̂ⱼ)ᵀ + εI   (nova covariância + regularização)
```

### Critério de Parada
Os passos E e M se repetem até que a variação relativa da log-verossimilhança seja < 10⁻⁴ ou até 200 iterações.

## Parâmetros e Hiperparâmetros

| Tipo | Nome | Valor | Descrição |
|:----:|:----:|:-----:|:----------|
| **Hiperparâmetro** | `test_size` | 0.2 | Proporção dos dados para teste |
| **Hiperparâmetro** | `random_state` | 0 a 19 | 20 sementes para 20 realizações independentes |
| **Hiperparâmetro (GMM)** | `J` (nº componentes) | 3 | Número de componentes Gaussianos por classe |
| **Hiperparâmetro (GMM)** | `max_iter` | 200 | Nº máximo de iterações do algoritmo EM |
| **Hiperparâmetro (GMM)** | `tol` | 10⁻⁴ | Tolerância para convergência da log-verossimilhança |
| **Hiperparâmetro (GMM)** | `ε` (regularização) | 10⁻⁶ | Regularização de Tikhonov (Σ + εI) |
| **Hiperparâmetro (GMM)** | `inicialização` | K-Means | Método de inicialização dos parâmetros do EM |
| **Hiperparâmetro (KNN)** | `k` | 5 | Nº de vizinhos mais próximos |
| **Parâmetro (aprendido)** | P(Cᵢ) | — | Probabilidade a priori de cada classe |
| **Parâmetro (aprendido)** | Pᵢⱼ (GMM) | — | Peso (prior) do componente j na classe i |
| **Parâmetro (aprendido)** | μᵢⱼ (GMM) | — | Vetor de médias do componente j na classe i |
| **Parâmetro (aprendido)** | Σᵢⱼ (GMM) | — | Matriz de covariância completa do componente j na classe i |

**Número de parâmetros do GMM por classe:**
```
J · (d(d+1)/2 + d + 1) - 1
```

| Dataset | d | J | Parâmetros por classe | Total (c classes) |
|---------|:-:|:-:|:---------------------:|:-----------------:|
| Artificial I | 2 | 3 | 3·(3+2+1)-1 = 17 | 51 |
| Iris | 4 | 3 | 3·(10+4+1)-1 = 44 | 132 |
| WDBC | 30 | 3 | 3·(465+30+1)-1 = 1487 | 2.974 |
| Dermatology | 34 | 3 | 3·(595+34+1)-1 = 1889 | 11.334 |
| Coluna Vertebral | 6 | 3 | 3·(21+6+1)-1 = 83 | 249 |

**Comparação com modelos anteriores (Iris, d=4):**

| Modelo | Parâmetros/classe | Total (3 classes) |
|--------|:-----------------:|:-----------------:|
| GBC / QDA Full (T2) | 15 | 45 |
| LDA (T4) | 11 | 11 (+ 3 priors) |
| GNB (T3) | 9 | 27 |
| DMC (T4) | 6 | 18 |
| **GMM-Bayes (J=3)** | **44** | **132** |

### Inicialização
Para reduzir a sensibilidade a mínimos locais, os parâmetros iniciais vêm do **K-Means**:
- μⱼ⁽⁰⁾ = centróides do K-Means
- Σⱼ⁽⁰⁾ = covariância empírica de cada cluster
- Pⱼ⁽⁰⁾ = frequência relativa dos clusters

## Fórmulas Principais

### Modelo de Mistura de Gaussianas (GMM)
```
p(x) = Σⱼ Pⱼ · N(x; μⱼ, Σⱼ)    onde Σⱼ Pⱼ = 1, Pⱼ > 0
```

### GMM por classe (classificação supervisionada)
```
p(x|Cᵢ) = Σⱼ Pᵢⱼ · N(x; μᵢⱼ, Σᵢⱼ)
```

### Gaussiana Multivariada (cada componente)
```
N(x; μⱼ, Σⱼ) = 1 / ((2π)^(d/2) · |Σⱼ|^(1/2)) · exp(-½(x - μⱼ)ᵀ Σⱼ⁻¹ (x - μⱼ))
```

### Regra de Decisão (MAP com GMM)
```
ŷ = argmaxᵢ [ log P(Cᵢ) + log( Σⱼ Pᵢⱼ · N(x; μᵢⱼ, Σᵢⱼ) ) ]
```

### Log-verossimilhança (monitorada durante EM)
```
L = Σₖ log( Σⱼ Pⱼ · N(xₖ; μⱼ, Σⱼ) )
```

### Número de parâmetros do GMM por classe
```
J · (d(d+1)/2 + d + 1) - 1
```
Para J=3 e d=30 (WDBC): 3 · (465 + 30 + 1) - 1 = 1487 parâmetros por classe!

## Classificadores comparados

| Classificador | Descrição |
|---------------|-----------|
| **GMM-Bayes** (J=3) | Mistura de Gaussianas por classe. Este trabalho. |
| **LDA** | Σ pooled compartilhada (fronteiras lineares) |
| **QDA** | Σ completa por classe (1 Gaussiana) |
| **KNN (k=5)** | Não-paramétrico |
| **DMC** | Distância mínima ao centróide |

## Resultados principais (acurácia média %)

| Dataset | GMM-Bayes | LDA | QDA | KNN | DMC |
|---------|:---------:|:---:|:---:|:---:|:---:|
| Artificial I | **100,00** | 97,69 | 98,46 | **100,00** | 88,85 |
| Breast Cancer | 94,12 | 96,23 | 95,09 | **96,49** | 93,46 |
| Dermatology | 89,44 | **96,94** | 30,56 | 96,32 | 96,25 |
| Íris | 95,83 | **98,33** | 97,83 | 95,83 | 86,83 |
| Coluna Vertebral | 80,00 | 77,98 | **83,23** | 77,34 | 73,47 |

## Destaques
- **Artificial I**: GMM-Bayes e KNN atingem **100% de acurácia** (dados multimodais sintéticos)
- **Dermatology**: QDA **colapsa** (30,56% — classe majoritária), evidenciando overfitting severo com 34 atributos e Σ completa por classe
- **GMM-Bayes**: alta variabilidade em dimensões elevadas (Dermatology: DP=5,26%; amplitude de ~18 p.p.)
- **LDA** é o mais estável em alta dimensão (Breast Cancer, Dermatology, Íris)

## Relação com os trabalhos anteriores

| Trabalho | Modelo | Relação com T5 |
|----------|--------|----------------|
| T1 | Gaussiana Univ/Bi/Multivariada (Iris) | Caso particular: J=1 (QDA) |
| T2 | GBC (Σ completa por classe) | Caso particular: J=1, Σ completa |
| T3 | GNB (Naive Bayes, Σ diagonal) | Caso particular: J=1, Σ diagonal |
| T4 | LDA/QDA/DMC/GNB | Generalizado por GMM com J≥1 |
| T5 | **GMM-Bayes (J=3)** | Modelo mais geral: múltiplos componentes |

## Conclusão
- **GMM-Bayes** é superior quando os dados são **genuinamente multimodais** (Artificial I: 100%)
- **Alta dimensionalidade penaliza** o GMM: muitos parâmetros (J·d(d+1)/2 por classe) exigem muitas amostras
- **LDA** mantém robustez em datasets com separação aproximadamente linear
- **QDA sem regularização** pode colapsar (Dermatology: 30,56%)
- A **escolha do número de componentes J** é crítica: J=3 é excessivo para dados unimodais (Íris) ou de alta dimensão
- GMM-Bayes é o modelo mais **flexível** da sequência de trabalhos, mas exige **cuidadoso balanceamento** entre expressividade e overfitting

## Gráficos gerados
- Matrizes de confusão para cada classificador
- Superfícies de decisão (para Artificial I, Iris, Coluna Vertebral)
- Comparação de acurácia entre classificadores (`comparacao_acuracia.png`)