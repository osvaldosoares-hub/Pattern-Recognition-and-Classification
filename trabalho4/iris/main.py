import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import sys
import os

# Importar os classificadores LDA e QDA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linear_discriminant.lda import LinearDiscriminantAnalysis
from quadratic_discriminant.qda  import (
    QDAFull,
    QDADiagonal,
    QDASpherical,
    QDARegularized,
    QDAPooled,
)

# Configurações
N_REAL = 20
TEST_SIZE = 0.2
RANDOM_STATE_BASE = 42
FEAT_NAMES = ['sepal length', 'sepal width', 'petal length', 'petal width']
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']
F1, F2 = 2, 3  # Índices dos atributos para visualização (petal length, petal width)

class KNN:
    """K-Nearest Neighbors para comparação."""
    
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_, self.y_ = X.copy(), y.copy()
    
    def predict(self, X):
        preds = []
        for x in X:
            d = np.sqrt(((self.X_ - x)**2).sum(axis=1))
            lbls = self.y_[np.argsort(d)[:self.k]]
            vals, cnt = np.unique(lbls, return_counts=True)
            preds.append(vals[np.argmax(cnt)])
        return np.array(preds)


class DMC:
    """Distância Mínima entre Centroides para comparação."""
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = {c: X[y == c].mean(axis=0) for c in self.classes_}
    
    def predict(self, X):
        preds = []
        for x in X:
            d = {c: np.sqrt(((x - self.centroids_[c])**2).sum()) for c in self.classes_}
            preds.append(min(d, key=d.get))
        return np.array(preds)

def plot_decision_surface(X_train, y_train, X_test, y_test, clf, title, filename, feat_indices=(F1, F2)):
    """Plota a superfície de decisão do classificador."""
    f1, f2 = feat_indices
    
    # Usar apenas os dois atributos selecionados
    X_train_2d = X_train[:, [f1, f2]]
    X_test_2d = X_test[:, [f1, f2]]
    
    # Criar grid
    h = 0.02  # passo da malha
    x_min, x_max = X_train_2d[:, 0].min() - 0.5, X_train_2d[:, 0].max() + 0.5
    y_min, y_max = X_train_2d[:, 1].min() - 0.5, X_train_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Criar dados para predição na mesma dimensionalidade
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            # Usar apenas os dois atributos
            point_2d = np.array([[xx[i, j], yy[i, j]]])
            # Precisamos adaptar o classificador para aceitar apenas 2 features
            # Vamos usar o atributo do modelo treinado
            Z[i, j] = predict_2d(clf, point_2d, f1, f2)
    
    # Plotar
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Superfície de decisão
    ax.contourf(xx, yy, Z, levels=15, alpha=0.3, cmap='viridis')
    ax.contour(xx, yy, Z, levels=len(CLASS_NAMES), alpha=0.5, colors='black', linewidths=0.5)
    
    # Dados de treinamento
    for c in np.unique(y_train):
        mask_train = y_train == c
        ax.scatter(X_train_2d[mask_train, 0], X_train_2d[mask_train, 1],
                  marker='o', s=100, c=[plt.cm.viridis(c/2)], label=f'{CLASS_NAMES[int(c)]} (train)',
                  alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Dados de teste
    for c in np.unique(y_test):
        mask_test = y_test == c
        ax.scatter(X_test_2d[mask_test, 0], X_test_2d[mask_test, 1],
                  marker='x', s=200, c=[plt.cm.viridis(c/2)], label=f'{CLASS_NAMES[int(c)]} (test)',
                  alpha=0.7, linewidth=2)
    
    ax.set_xlabel(FEAT_NAMES[f1], fontsize=12)
    ax.set_ylabel(FEAT_NAMES[f2], fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def predict_2d(clf, X_point, f1, f2):
    """Adapta o classificador para trabalhar com 2D durante a predição na superfície."""
    # Aqui temos um problema: o modelo foi treinado com 4 features
    # Precisamos expandir o ponto para 4 features usando valores médios
    
    # Usar a média dos atributos não usados
    point_full = np.zeros((1, 4))
    point_full[0, [f1, f2]] = X_point[0]
    
    # Para os outros atributos, usar a média global (ou 0)
    if hasattr(clf, 'means_'):
        for i in range(4):
            if i not in [f1, f2]:
                point_full[0, i] = 0  # Usar 0 ou média
    
    pred = clf.predict(point_full)
    return pred[0]


def main():
    # Carregar dados
    iris = load_iris()
    X = iris.data
    y = iris.target
 
    print("=" * 80)
    print("ANÁLISE DE DISCRIMINANTES LINEARES E QUADRÁTICOS - IRIS DATASET")
    print("=" * 80)
    print(f"\nDataset: Iris")
    print(f"Número de amostras: {X.shape[0]}")
    print(f"Número de atributos: {X.shape[1]}")
    print(f"Número de classes: {len(np.unique(y))}")
    print(f"Número de realizações: {N_REAL}")
    print(f"Tamanho do conjunto de teste: {TEST_SIZE*100:.0f}%")
 
    # Normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    CLASSIFIERS = {
        'LDA (Caso 2 — Σ pool.)    ': LinearDiscriminantAnalysis(),
        'QDA Full (Caso 1 — Σ_k)   ': QDAFull(),
        'QDA Diagonal (Caso 3)      ': QDADiagonal(),
        'QDA Isotrópica (Caso 4)    ': QDASpherical(),
        'QDA Regularized            ': QDARegularized(),
        'QDA Pooled α=0.5           ': QDAPooled(reg_strength=0.5),
        'KNN k=5                    ': KNN(k=5),
        'DMC                        ': DMC(),
    }
 
    results = {name: {'accuracies': [], 'y_test': [], 'y_pred': []}
               for name in CLASSIFIERS}
 
    best_realization = {'accuracy': -1, 'name': None, 'clf': None,
                        'X_train': None, 'X_test': None,
                        'y_train': None, 'y_test': None, 'y_pred': None}
 
    print("\n" + "-" * 80)
    print("Executando 20 realizações...")
    print("-" * 80)
 
    for real in range(N_REAL):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE_BASE + real
        )
 
        for name, clf in CLASSIFIERS.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
 
            results[name]['accuracies'].append(acc)
            results[name]['y_test'].append(y_test)
            results[name]['y_pred'].append(y_pred)
 
            if acc > best_realization['accuracy']:
                best_realization.update(accuracy=acc, name=name.strip(), clf=clf,
                                        X_train=X_train, X_test=X_test,
                                        y_train=y_train, y_test=y_test, y_pred=y_pred)
 
        if (real + 1) % 5 == 0:
            print(f"  Realização {real+1:2d}/20 concluída")
 
    # ── Tabela de resultados ordenada do melhor para o pior ───────────────────
    print("\n" + "=" * 80)
    print("RESULTADOS FINAIS")
    print(f"  {'Classificador':<40} {'Média':>7} {'DP':>7} {'Mín':>7} {'Máx':>7}")
    print("-" * 80)
 
    ranking = sorted(results.items(),
                     key=lambda x: np.mean(x[1]['accuracies']),
                     reverse=True)
 
    for name, res in ranking:
        accs = np.array(res['accuracies'])
        print(f"  {name:<40} {accs.mean()*100:6.2f}%  {accs.std()*100:5.2f}%"
              f"  {accs.min()*100:6.2f}%  {accs.max()*100:6.2f}%")
 
    print("=" * 80)
    print(f"\n  Melhor realização: {best_realization['name']}"
          f"  (acc = {best_realization['accuracy']*100:.2f}%)")
 
    # ── Matriz de confusão ────────────────────────────────────────────────────
    cm = confusion_matrix(best_realization['y_test'], best_realization['y_pred'])
    print(f"\nMatriz de Confusão - {best_realization['name']}:")
    print(cm)
 
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Verdadeiro', fontsize=12)
    ax.set_title(f'Matriz de Confusão - {best_realization["name"]}\n'
                 f'Acurácia: {best_realization["accuracy"]*100:.2f}%',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ matriz_confusao.png")
 
    # ── Superfície de decisão ─────────────────────────────────────────────────
    print("\nGerando superfícies de decisão...")
    plot_decision_surface(
        best_realization['X_train'], best_realization['y_train'],
        best_realization['X_test'],  best_realization['y_test'],
        best_realization['clf'],
        f'Superfície de Decisão - {best_realization["name"]} (Iris)\n'
        f'{FEAT_NAMES[F1]} vs {FEAT_NAMES[F2]}',
        'superficies_decisao.png'
    )
    print("✓ superficies_decisao.png")
 
    # ── Boxplot + Barplot comparativo ─────────────────────────────────────────
    names_short = [n.strip().split('(')[0].strip() for n in results]
    data        = [np.array(results[n]['accuracies']) * 100 for n in results]
    colors_box  = plt.cm.Set2(np.linspace(0, 1, len(data)))
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
 
    bp = axes[0].boxplot(data, patch_artist=True, tick_labels=names_short)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)
    axes[0].set_ylabel('Acurácia (%)', fontsize=12)
    axes[0].set_title('Distribuição de Acurácias (20 realizações)',
                      fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
 
    means = [d.mean() for d in data]
    stds  = [d.std()  for d in data]
    x_pos = np.arange(len(names_short))
    axes[1].bar(x_pos, means, yerr=stds, capsize=6,
                alpha=0.8, color=colors_box, edgecolor='black')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(names_short, rotation=45, ha='right')
    axes[1].set_ylabel('Acurácia (%)', fontsize=12)
    axes[1].set_title('Acurácia Média ± Desvio Padrão',
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[1].text(i, m + s + 0.3, f'{m:.1f}%',
                     ha='center', fontsize=8, fontweight='bold')
 
    plt.tight_layout()
    plt.savefig('comparacao_acuracia.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ comparacao_acuracia.png")
 
    print("\n" + "=" * 80)
    print("Análise concluída!")
    print("=" * 80)
 
 
if __name__ == '__main__':
    main()
 