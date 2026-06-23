# Trabalho 1 — Classificador Bayesiano Gaussiano (Dataset Iris)

## Objetivo
Implementar um **Classificador Bayesiano Gaussiano** aplicado exclusivamente ao dataset **Iris** (3 classes, 4 atributos, 150 amostras), explorando diferentes abordagens de dimensionalidade.

## Entrada de Dados
- Dataset Iris carregado via `sklearn.datasets.load_iris()`
- Dados já estruturados em matriz X (150×4) e vetor de rótulos y (150, valores 0, 1, 2)
- Atributos: comprimento/largura da sépala e comprimento/largura da pétala (cm)
- Classes: *Iris setosa* (0), *Iris versicolor* (1), *Iris virginica* (2)
- Divisão **80% treino / 20% teste** com `train_test_split(stratify=y, random_state=42)`

## Treinamento
Não há um "treinamento iterativo" — o classificador Bayesiano é **paramétrico**: os parâmetros são estimados diretamente das amostras de treino:

1. **Priors**: frequência relativa de cada classe: P(Cᵢ) = nᵢ / N_total
2. **Médias e Variâncias**: estimadas por Máxima Verossimilhança (MLE) para cada classe:
   - Univariado: μᵢⱼ e σ²ᵢⱼ para cada atributo j
   - Bivariado: vetor μ 2D e matriz Σ 2×2
   - Multivariado: vetor μ 4D e matriz Σ 4×4

## Parâmetros e Hiperparâmetros

| Tipo | Nome | Valor | Descrição |
|:----:|:----:|:-----:|:----------|
| **Hiperparâmetro** | `test_size` | 0.2 | Proporção dos dados reservada para teste |
| **Hiperparâmetro** | `random_state` | 42 | Semente para reprodutibilidade da divisão |
| **Hiperparâmetro** | `stratify` | True | Garante mesma proporção de classes no treino/teste |
| **Parâmetro (aprendido)** | P(Cᵢ) | — | Probabilidade a priori de cada classe (frequência no treino) |
| **Parâmetro (aprendido)** | μᵢⱼ | — | Média do atributo j para a classe i |
| **Parâmetro (aprendido)** | σ²ᵢⱼ | — | Variância do atributo j para a classe i |
| **Parâmetro (aprendido)** | Σᵢ (multivariado) | — | Matriz de covariância completa 4×4 da classe i |

**Total de parâmetros (multivariado, 4 atributos, 3 classes):**
- Priors: 3
- Médias: 3 × 4 = 12
- Covariâncias: 3 × (4×5/2) = 3 × 10 = 30
- **Total: 45 parâmetros**

## Fórmulas Principais

### Teorema de Bayes
```
P(Cᵢ|x) = p(x|Cᵢ) · P(Cᵢ) / Σⱼ p(x|Cⱼ) · P(Cⱼ)
```

### Gaussiana Univariada (usada nos códigos univariado e clássico)
```
p(x|Cᵢ) = 1/√(2πσ²ᵢ) · exp(-(x - μᵢ)² / (2σ²ᵢ))
```

### Gaussiana Multivariada (usada no código multiatributos)
```
p(x|Cᵢ) = 1/((2π)^(d/2) · |Σᵢ|^(1/2)) · exp(-½(x - μᵢ)ᵀ Σᵢ⁻¹ (x - μᵢ))
```

### Decisão MAP (Maximum a Posteriori)
```
ŷ = argmaxᵢ P(Cᵢ|x)
```

## Abordagens implementadas

| Abordagem | Código | Descrição |
|-----------|--------|-----------|
| **Univariado** | `iris_univariado.py` | Classifica usando **um único atributo** por vez. Calcula a Gaussiana univariada para cada classe e a posteriori. |
| **Bivariado** | `iris_bivariado.py` | Usa **pares de atributos** (6 combinações). Aplica Gaussiana bivariada com vetor de médias 2D e matriz de covariância 2×2. |
| **Multiatributos** | `iris_multiatributos.py` | Utiliza **todos os 4 atributos** simultaneamente via Gaussiana multivariada completa (vetor μ 4D, matriz Σ 4×4). |
| **Clássico** | `iris.py` | Versão simplificada: análise univariada com cálculo de priors, médias, variâncias e posteriori. |

## Principais resultados
- Acurácia com **4 atributos (multivariado)**: ~96-97%
- Acurácia com **atributos isolados**: petal length (~95%) > petal width (~93%) >> sepal length/width (~70-80%)
- A classe **Setosa** é perfeitamente separável; a maior confusão ocorre entre **Versicolor e Virginica**

## Conclusão
O classificador Bayesiano Gaussiano multivariado (com todos os 4 atributos) apresenta a melhor performance. Os atributos de **pétala** (length e width) são os mais discriminativos. O dataset Iris serve como excelente baseline introdutório para classificação Bayesiana.

## Gráficos gerados
- `iris_separacao.png` — dispersão treino (2 visões: pétalas e sépalas)
- `iris_gaussianas.png` — curvas gaussianas marginais por atributo e classe
- `iris_matriz_confusao.png` — matriz de confusão da predição
- `iris_bivariado_acuracia.png` — acurácia por par de atributos
- `iris_bivariado_contornos.png` — contornos gaussianos bivariados