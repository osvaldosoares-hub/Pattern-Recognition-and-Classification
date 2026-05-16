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
    QuadraticDiscriminantAnalysis,
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
    
    # Armazenar resultados
    results = {
        'LDA': {'accuracies': [], 'predictions': [], 'y_test': []},
        'QDA': {'accuracies': [], 'predictions': [], 'y_test': []},
       
    }
    
    best_realization = {'accuracy': 0, 'method': None, 'clf': None, 'X_train': None, 'X_test': None,
                       'y_train': None, 'y_test': None, 'y_pred': None}
    
    # Executar 20 realizações
    print("\n" + "-" * 80)
    print("Executando 20 realizações...")
    print("-" * 80)
    
    for real in range(N_REAL):
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_BASE + real
        )
        
        # Treinar LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred_lda = lda.predict(X_test)
        acc_lda = accuracy_score(y_test, y_pred_lda)
        results['LDA']['accuracies'].append(acc_lda)
        results['LDA']['predictions'].append(y_pred_lda)
        results['LDA']['y_test'].append(y_test)
        
        if acc_lda > best_realization['accuracy']:
            best_realization['accuracy'] = acc_lda
            best_realization['method'] = 'LDA'
            best_realization['clf'] = lda
            best_realization['X_train'] = X_train
            best_realization['X_test'] = X_test
            best_realization['y_train'] = y_train
            best_realization['y_test'] = y_test
            best_realization['y_pred'] = y_pred_lda
        
        # Treinar QDA
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)
        y_pred_qda = qda.predict(X_test)
        acc_qda = accuracy_score(y_test, y_pred_qda)
        results['QDA']['accuracies'].append(acc_qda)
        results['QDA']['predictions'].append(y_pred_qda)
        results['QDA']['y_test'].append(y_test)
        
        if acc_qda > best_realization['accuracy']:
            best_realization['accuracy'] = acc_qda
            best_realization['method'] = 'QDA'
            best_realization['clf'] = qda
            best_realization['X_train'] = X_train
            best_realization['X_test'] = X_test
            best_realization['y_train'] = y_train
            best_realization['y_test'] = y_test
            best_realization['y_pred'] = y_pred_qda
        
        
        
        
        if (real + 1) % 5 == 0:
            print(f"Realização {real + 1:2d}/20 - LDA: {acc_lda:.4f} | QDA: {acc_qda:.4f} ")
    
    # Calcular estatísticas
    print("\n" + "=" * 80)
    print("RESULTADOS FINAIS")
    print("=" * 80)
    
    for method in ['LDA', 'QDA']:
        accs = np.array(results[method]['accuracies'])
        mean_acc = accs.mean()
        std_acc = accs.std()
        min_acc = accs.min()
        max_acc = accs.max()
        
        print(f"\n{method}:")
        print(f"  Acurácia Média:       {mean_acc:.4f}")
        print(f"  Desvio Padrão:        {std_acc:.4f}")
        print(f"  Acurácia Mínima:      {min_acc:.4f}")
        print(f"  Acurácia Máxima:      {max_acc:.4f}")
    
    # Melhor realização
    print(f"\n" + "-" * 80)
    print(f"Melhor realização: {best_realization['method']} com acurácia {best_realization['accuracy']:.4f}")
    print("-" * 80)
    
    # Matriz de confusão da melhor realização
    cm = confusion_matrix(best_realization['y_test'], best_realization['y_pred'])
    print(f"\nMatriz de Confusão - {best_realization['method']}:")
    print(cm)
    
    # Plotar matriz de confusão
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Verdadeiro', fontsize=12)
    ax.set_title(f'Matriz de Confusão - {best_realization["method"]} (Acurácia: {best_realization["accuracy"]:.4f})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nMatriz de confusão salva em: matriz_confusao.png")
    
    # Plotar superfícies de decisão
    print("\nGerando superfícies de decisão...")
    
    plot_decision_surface(
        best_realization['X_train'][:, :4], best_realization['y_train'],
        best_realization['X_test'][:, :4], best_realization['y_test'],
        best_realization['clf'],
        f'Superfície de Decisão - {best_realization["method"]} (Iris)\n{FEAT_NAMES[F1]} vs {FEAT_NAMES[F2]}',
        'superficies_decisao.png'
    )
    
    print("Superfícies de decisão salvas em: superficies_decisao.png")
    
    # Comparação gráfica de acurácias
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Boxplot
    data = [results['LDA']['accuracies'], results['QDA']['accuracies']]
    axes[0].boxplot(data, tick_labels=['LDA', 'QDA'])
    axes[0].set_ylabel('Acurácia', fontsize=12)
    axes[0].set_title('Distribuição de Acurácias (20 realizações)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.9, 1.01])
    
    # Comparação de médias
    methods = ['LDA', 'QDA']
    means = [np.mean(results[m]['accuracies']) for m in methods]
    stds = [np.std(results[m]['accuracies']) for m in methods]
    
    x_pos = np.arange(len(methods))
    axes[1].bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel('Acurácia', fontsize=12)
    axes[1].set_title('Acurácia Média ± Desvio Padrão', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0.9, 1.01])
    
    # Adicionar valores no gráfico
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[1].text(i, m + s + 0.005, f'{m:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comparacao_acuracia.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Comparação de acurácias salva em: comparacao_acuracia.png")
    
    print("\n" + "=" * 80)
    print("Análise concluída!")
    print("=" * 80)


if __name__ == '__main__':
    main()
