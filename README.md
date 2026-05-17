# Machine Learning — Trabalhos Práticos

Repositório com os trabalhos práticos da disciplina de Machine Learning, cada um implementando classificadores do zero e avaliando-os em diferentes datasets.

---

## Trabalho 1 — Análise Exploratória e Classificador Bayesiano Univariado (Iris)

**Objetivo:** Explorar o dataset Iris e implementar um classificador bayesiano gaussiano univariado (um atributo por vez) e bivariado.

### Scripts

| Arquivo | Descrição |
|---|---|
| `trabalho1/iris.py` | Análise estatística (média, variância) por classe e divisão treino/teste |
| `trabalho1/iris_univariado.py` | Classificador bayesiano gaussiano usando um único atributo por vez; gera acurácia e posterior por atributo |
| `trabalho1/iris_bivariado.py` | Classificador bayesiano gaussiano bivariado para todos os 6 pares de atributos; plota superfícies de decisão 2D |
| `trabalho1/iris_multiatributos.py` | Classificador bayesiano gaussiano multivariado completo (4 atributos) com cálculo manual da matriz de covariância |

### Dataset
- **Iris** (scikit-learn) — 150 amostras, 4 atributos, 3 classes (*setosa*, *versicolor*, *virginica*)

### Destaques
- Cálculo manual de média, variância, PDF gaussiana e probabilidade a posteriori
- Superfícies de decisão e curvas de densidade visualizadas com Matplotlib
- Divisão treino/teste estratificada (80/20)

---

## Trabalho 2 — Classificador Bayesiano Gaussiano Multivariado

**Objetivo:** Implementar o classificador bayesiano gaussiano multivariado considerando que cada classe possui seu **próprio vetor de média** e sua **própria matriz de covariância**. As probabilidades a posteriori são calculadas diretamente pela regra de Bayes sem simplificações (sem LDA, sem diagonalização).

### Formulação

$$P(c \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid c)\, P(c)}{p(\mathbf{x})}$$

onde a verossimilhança é a densidade gaussiana multivariada:

$$p(\mathbf{x} \mid c) = \frac{1}{(2\pi)^{d/2} |\Sigma_c|^{1/2}} \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_c)^\top \Sigma_c^{-1}(\mathbf{x}-\boldsymbol{\mu}_c)\right)$$

### Scripts

| Arquivo | Dataset |
|---|---|
| `trabalho2/artificial/artificial.py` | Dataset artificial (3 classes gaussianas 2D geradas sinteticamente) |
| `trabalho2/iris/iris.py` | Iris (4 atributos, 3 classes) |
| `trabalho2/Breast Cancer/Breast Cancer.py` | Breast Cancer Wisconsin — WDBC (30 atributos, 2 classes: Benigno/Maligno) |
| `trabalho2/dermatology/dermatology.py` | Dermatology (34 atributos, 6 classes) |
| `trabalho2/Vertebral/Vertebral.py` | Vertebral Column (6 atributos, 3 classes: Hérnia, Espondilolistese, Normal) |

### Metodologia
- **20 realizações** com divisão aleatória estratificada 80/20
- Métricas reportadas: acurácia média ± desvio padrão, matriz de confusão
- Comparação com KNN (k=3) e DMC (Distância Mínima ao Centróide)
- Visualização de superfícies de decisão e elipses de covariância por classe

---

## Trabalho 3 — Classificador Naive Bayes

**Objetivo:** Implementar o classificador **Naive Bayes gaussiano**, assumindo que os atributos são **independentes entre si** para o cálculo da verossimilhança. As probabilidades a posteriori são computadas para cada classe.

### Formulação

Sob a suposição de independência condicional:

$$p(\mathbf{x} \mid c) = \prod_{j=1}^{d} p(x_j \mid c)$$

onde cada fator é uma gaussiana univariada:

$$p(x_j \mid c) = \frac{1}{\sqrt{2\pi\,\sigma_{cj}^2}} \exp\!\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2}\right)$$

A posteriori é obtida por:

$$P(c \mid \mathbf{x}) = \frac{P(c)\prod_j p(x_j \mid c)}{\sum_{c'} P(c')\prod_j p(x_j \mid c')}$$

O cálculo usa **log-verossimilhança + log-sum-exp** para estabilidade numérica.

### Scripts

| Arquivo | Dataset |
|---|---|
| `trabalho3/artificial_I/main.py` | Dataset Artificial I (3 classes gaussianas 2D) |
| `trabalho3/iris/main.py` | Iris (4 atributos, 3 classes) |
| `trabalho3/breast_cancer/main.py` | Breast Cancer Wisconsin — WDBC (30 atributos, 2 classes) |
| `trabalho3/dermatology/main.py` | Dermatology (34 atributos, 6 classes) |
| `trabalho3/vertebral/main.py` | Vertebral Column (6 atributos, 3 classes) |

### Metodologia
- **20 realizações** com divisão aleatória estratificada 80/20
- Métricas reportadas: acurácia média ± desvio padrão, matriz de confusão
- Comparação com KNN (k=3) e DMC
- Visualização de superfícies de decisão e distribuições gaussianas marginais por classe

---

## Datasets Utilizados

| Dataset | Amostras | Atributos | Classes |
|---|---|---|---|
| Iris | 150 | 4 | 3 |
| Breast Cancer Wisconsin (WDBC) | 569 | 30 | 2 |
| Dermatology | 366 | 34 | 6 |
| Vertebral Column | 310 | 6 | 3 |
| Artificial I | 120 | 2 | 3 |

---

## Dependências

```
numpy
matplotlib
scikit-learn
scipy
pandas
```

Instale com:

```bash
pip install numpy matplotlib scikit-learn scipy pandas
```
