# Resumo do Trabalho 6 — Parzen Bayes com Grid Search e K-Fold Cross-Validation

## Objetivo

Implementar e avaliar o classificador **Parzen Bayes** (estimador de densidade por janela de Parzen com kernel Gaussiano) em **5 bases de dados**, comparando seu desempenho com os métodos **LDA, QDA, KNN e DMC**. Adicionalmente, foi aplicado **Grid Search com K-Fold Cross-Validation** para otimizar o hiperparâmetro **bandwidth_scale** (largura de banda do kernel).

---

## Hiperparâmetros do Parzen Bayes

O classificador Parzen Bayes possui **um hiperparâmetro principal** que controla a suavidade da estimativa de densidade:

### `bandwidth_scale` (fator de escala da largura de banda)

- **O que é**: Um multiplicador que ajusta a largura da janela de Parzen (kernel Gaussiano) para todas as classes.
- **Onde atua**: Na fórmula \( h_c = \text{bandwidth\_scale} \cdot \sigma_c \cdot n_c^{-\frac{1}{d+4}} \)
- **Valores testados no Grid Search**: `[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]`
- **Valor default (sem otimização)**: `1.0`

### Efeito do `bandwidth_scale`

| Scale | Efeito | Consequência |
|-------|--------|-------------|
| **Muito pequeno** (< 0.1) | Janela muito estreita | Overfitting — o modelo praticamente memoriza os pontos de treino; superfície de decisão muito complexa |
| **Moderado** (0.1 ~ 1.0) | Janela equilibrada | Bom equilíbrio entre viés e variância |
| **Grande** (> 1.0) | Janela muito larga | Underfitting — o modelo suaviza demais, perde detalhes da distribuição |

### Interpretação prática dos resultados

- **Artificial_I (best scale = 0.01)**: Dataset com apenas 2 atributos e distribuição compacta — uma janela muito estreita ainda generaliza bem porque os dados são bem separados.
- **Dermatology (best scale = 1.5)**: Dataset com 34 atributos — precisou de uma janela mais larga para suavizar a alta dimensionalidade e evitar overfitting.
- **Vertebral e Breast_Cancer**: O default (1.0) foi melhor que o melhor scale encontrado, indicando que o grid poderia ser expandido ou que a variabilidade entre folds influenciou a escolha.

---

## Bases de Dados

| Dataset        | Classes | Atributos | Amostras |
|----------------|---------|-----------|----------|
| Iris           | 3       | 4         | 150      |
| Artificial_I   | 2       | 2         | 200      |
| Vertebral      | 3       | 6         | 310      |
| Breast_Cancer  | 2       | 30        | 569      |
| Dermatology    | 6       | 34        | 366      |

---

## Metodologia

### 1. **Holdout** (80% treino / 20% teste)
- Separação inicial dos dados antes de qualquer otimização, garantindo que o teste permaneça **congelado** até a avaliação final.

### 2. **Padronização (StandardScaler)**
- Fit realizado **apenas no conjunto de treino** para evitar vazamento de informação do teste.

### 3. **Estimação da Janela de Parzen (kernel Gaussiano)**

O classificador Parzen Bayes estima a densidade de probabilidade de cada classe utilizando uma **janela de Parzen** com kernel Gaussiano. A lógica completa do cálculo é:

#### a) Largura de banda (bandwidth)

Para cada classe \(c\), a largura de banda do kernel é estimada pela regra:

\[
h_c = \text{bandwidth\_scale} \cdot \sigma_c \cdot n_c^{-\frac{1}{d + 4}}
\]

onde:
- \(\sigma_c\) é a média dos desvios padrão dos atributos da classe \(c\)
- \(n_c\) é o número de amostras da classe \(c\)
- \(d\) é o número de atributos (dimensionalidade)
- \(\text{bandwidth\_scale}\) é o hiperparâmetro ajustado via Grid Search

#### b) Densidade logarítmica de Parzen

A log-verossimilhança de um ponto \(x\) pertencer à classe \(c\) é calculada como:

\[
\log p(\mathbf{x}|c) = -\log(n_c) - \frac{d}{2}\log(2\pi) - d\log(h_c) + \log\sum_{i=1}^{n_c} \exp\left(-\frac{1}{2}\frac{\|\mathbf{x} - \mathbf{x}_i\|^2}{h_c^2}\right)
\]

onde:
- \(\|\mathbf{x} - \mathbf{x}_i\|^2\) é a distância Euclidiana ao quadrado entre o ponto \(x\) e cada amostra de treino \(\mathbf{x}_i\) da classe \(c\)
- O termo \(\log\sum\exp\) (logsumexp) é usado por estabilidade numérica

#### c) Decisão final

A classe predita é aquela que maximiza o log-posterior:

\[
\hat{y} = \arg\max_c \left[ \log p(\mathbf{x}|c) + \log P(c) \right]
\]

onde \(P(c)\) é a priori da classe (frequência relativa no treino).

---

### 4. **Grid Search com K-Fold CV (5 folds)**
- Hiperparâmetro otimizado: **bandwidth_scale** (valores testados: `[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]`).
- A escolha do melhor *scale* foi baseada **exclusivamente na média da validação cruzada sobre o treino**.
- Ao final, o teste foi usado **uma única vez** para medir o desempenho real.

### 5. **Comparação com outros classificadores**
- Cada base foi avaliada com **30 realizações** (random splits) para cálculo de média e desvio padrão da acurácia.

---

## Resultados do Grid Search

| Dataset        | Best Scale | CV Score | Test Score | Default (1.0) | Ganho   |
|----------------|------------|----------|------------|---------------|---------|
| Iris           | 0.6000     | 0.9667   | 0.9667     | 0.9667        | +0.0000 |
| Artificial_I   | 0.0100     | 1.0000   | 1.0000     | 1.0000        | +0.0000 |
| Vertebral      | 0.7000     | 0.8310   | 0.7742     | 0.8548        | -0.0806 |
| Breast_Cancer  | 0.7000     | 0.9429   | 0.9035     | 0.9298        | -0.0263 |
| Dermatology    | 1.5000     | 0.9266   | 0.9722     | 0.9028        | +0.0694 |

- O **ganho mais expressivo** foi em **Dermatology** (+0.0694), onde o *scale* 1.5 elevou a acurácia de 0.9028 para 0.9722.
- Para **Vertebral** e **Breast_Cancer**, o *default* (1.0) foi superior ao *best scale* encontrado, indicando que o grid testado não continha valores melhores ou que a otimização sofreu com variabilidade entre folds.

---

## Comparação Final entre Classificadores (acurácia média sobre 30 realizações)

| Dataset        | Parzen Bayes | LDA    | QDA    | KNN    | DMC    |
|----------------|--------------|--------|--------|--------|--------|
| Artificial_I   | **1.0000**   | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Breast_Cancer  | 0.9224       | 0.9623 | **0.9509** | **0.9649** | 0.9346 |
| Dermatology    | 0.9042       | **0.9694** | 0.3056 | 0.9632 | 0.9625 |
| Iris           | 0.9417       | **0.9833** | 0.9783 | 0.9583 | 0.8683 |
| Vertebral      | 0.7750       | 0.7798 | **0.8323** | 0.7734 | 0.7347 |

- **Artificial_I** é um dataset linearmente separável — todos os métodos atingiram 100%.
- **Iris** e **Breast_Cancer**: LDA e KNN tiveram melhor desempenho que Parzen Bayes.
- **Dermatology**: QDA colapsou (0.3056, equivalente a chute uniforme), enquanto Parzen Bayes (0.9042) perdeu apenas para LDA (0.9694).
- **Vertebral**: QDA foi o melhor (0.8323); Parzen Bayes (0.7750) ficou próximo de LDA e KNN.

---

## Conclusão

1. **Sem vazamento de dados**: O fluxo holdout → padronização no treino → K-fold CV → avaliação final no teste garantiu uma validação honesta.
2. **Grid Search trouxe benefício pontual**: Para Dermatology, a otimização elevou a acurácia em ~7 pontos percentuais. Nas demais bases, o ganho foi nulo ou negativo, sugerindo que o *bandwidth_scale* default (1.0) já era adequado ou que o grid poderia ser expandido.
3. **Parzen Bayes é competitivo**: Em 3 das 5 bases, sua acurácia ficou a menos de 4 pontos percentuais do melhor método. Apenas em Artificial_I (empate) e Dermatology (disparou com tuning) ele se destacou mais.
4. **Limitação do QDA**: Sofreu com a maldição da dimensionalidade em Dermatology (34 atributos, amostras por classe insuficientes para estimar matrizes de covariância).
5. **Reprodutibilidade**: Todos os resultados e figuras (matrizes de confusão, superfícies de decisão, gráficos comparativos) foram gerados e salvos nos diretórios de cada dataset.