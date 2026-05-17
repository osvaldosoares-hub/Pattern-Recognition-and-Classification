import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import sys
import os

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
F1, F2 = 0, 1  # erythema, scaling

FEAT_NAMES = [
    'erythema', 'scaling', 'definite borders', 'itching',
    'koebner phenomenon', 'polygonal papules', 'follicular papules',
    'oral mucosal involvement', 'knee and elbow involvement',
    'scalp involvement', 'family history', 'melanin incontinence',
    'eosinophils infiltrate', 'PNL infiltrate', 'fibrosis papillary dermis',
    'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis',
    'clubbing rete ridges', 'elongation rete ridges',
    'thinning suprapapillary epidermis', 'spongiform pustule',
    'munro microabcess', 'focal hypergranulosis',
    'disappearance granular layer', 'vacuolisation basal layer',
    'spongiosis', 'saw tooth appearance', 'follicular horn plug',
    'perifollicular parakeratosis', 'inflammatory mononuclear infiltrate',
    'band like infiltrate', 'age'
]

CLASS_NAMES = [
    'psoriasis', 'seboreic dermatitis', 'lichen planus',
    'pityriasis rosea', 'cronic dermatitis', 'pityriasis rubra pilaris'
]



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



def load_data():
    """Carrega dados de Dermatology."""
    rows = []
    try:
        with open('dermatology.data', 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('@'):
                    continue
                parts = line.split(',')
                if len(parts) == 35:  # 34 features + 1 class
                    try:
                        feats = list(map(float, parts[:34]))
                        label = int(parts[34])
                        rows.append((feats, label))
                    except ValueError:
                        continue
    except FileNotFoundError:
        print("ERRO: Arquivo 'dermatology.data' não encontrado!")
        return None, None
    
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] - 1 for r in rows])  # Labels 1-6 -> 0-5
    
    return X, y


def plot_decision_surface(X_train, y_train, X_test, y_test, clf, title, filename,
                         f1=F1, f2=F2, feat_names=FEAT_NAMES):
    """Plota a superfície de decisão do classificador."""
    X_train_2d = X_train[:, [f1, f2]]
    X_test_2d = X_test[:, [f1, f2]]
    
    # Criar grid
    h = 0.1
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predições na grid
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point_2d = np.array([[xx[i, j], yy[i, j]]])
            # Expandir para 34 dimensões
            point_full = np.zeros((1, 34))
            point_full[0, [f1, f2]] = point_2d[0]
            # Para outros atributos, usar média
            for k in range(34):
                if k not in [f1, f2]:
                    point_full[0, k] = X_train[:, k].mean()
            Z[i, j] = clf.predict(point_full)[0]
    
    # Plotar
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Superfície de decisão
    ax.contourf(xx, yy, Z, levels=15, alpha=0.3, cmap='viridis')
    ax.contour(xx, yy, Z, alpha=0.5, colors='black', linewidths=0.5)
    
    # Dados de treinamento
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(y_train))))
    for i, c in enumerate(np.unique(y_train)):
        mask_train = y_train == c
        ax.scatter(X_train_2d[mask_train, 0], X_train_2d[mask_train, 1],
                  marker='^', s=100, c=[colors[i]], label=f'{CLASS_NAMES[int(c)]} (train)',
                  alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Dados de teste
    for i, c in enumerate(np.unique(y_test)):
        mask_test = y_test == c
        ax.scatter(X_test_2d[mask_test, 0], X_test_2d[mask_test, 1],
                  marker='o', s=150, c=[colors[i]], label=f'{CLASS_NAMES[int(c)]} (test)',
                  alpha=0.7, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel(feat_names[f1], fontsize=12)
    ax.set_ylabel(feat_names[f2], fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Carregar dados
    X, y = load_data()
    
    if X is None:
        return
    
    print("=" * 80)
    print("ANÁLISE DE DISCRIMINANTES LINEARES E QUADRÁTICOS - DERMATOLOGY DATASET")
    print("=" * 80)
    print(f"\nDataset: Dermatology")
    print(f"Número de amostras: {X.shape[0]}")
    print(f"Número de atributos: {X.shape[1]}")
    print(f"Número de classes: {len(np.unique(y))}")
    dist = {c: int(sum(y==c)) for c in np.unique(y)}
    print(f"Distribuição: {dist}")
    print(f"Número de realizações: {N_REAL}")
    print(f"Tamanho do conjunto de teste: {TEST_SIZE*100:.0f}%")
    
    # Normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Armazenar resultados
    results = {
        'LDA': {'accuracies': []},
        'QDA': {'accuracies': []},
        
        'KNN': {'accuracies': []},
        'DMC': {'accuracies': []}
    }
    
    best_realization = {'accuracy': 0, 'method': None, 'clf': None, 'X_train': None, 'X_test': None,
                       'y_train': None, 'y_test': None, 'y_pred': None}
    
    # Executar 20 realizações
    print("\n" + "-" * 80)
    print("Executando 20 realizações...")
    print("-" * 80)
    
    for real in range(N_REAL):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_BASE + real, stratify=y
        )
        
        # LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred_lda = lda.predict(X_test)
        acc_lda = accuracy_score(y_test, y_pred_lda)
        results['LDA']['accuracies'].append(acc_lda)
        
        if acc_lda > best_realization['accuracy']:
            best_realization['accuracy'] = acc_lda
            best_realization['method'] = 'LDA'
            best_realization['clf'] = lda
            best_realization['X_train'] = X_train
            best_realization['X_test'] = X_test
            best_realization['y_train'] = y_train
            best_realization['y_test'] = y_test
            best_realization['y_pred'] = y_pred_lda
        
        # QDA
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)
        y_pred_qda = qda.predict(X_test)
        acc_qda = accuracy_score(y_test, y_pred_qda)
        results['QDA']['accuracies'].append(acc_qda)
        
        if acc_qda > best_realization['accuracy']:
            best_realization['accuracy'] = acc_qda
            best_realization['method'] = 'QDA'
            best_realization['clf'] = qda
            best_realization['X_train'] = X_train
            best_realization['X_test'] = X_test
            best_realization['y_train'] = y_train
            best_realization['y_test'] = y_test
            best_realization['y_pred'] = y_pred_qda
        
      
        
        # KNN
        knn = KNN(k=5)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        results['KNN']['accuracies'].append(acc_knn)
        
        # DMC
        dmc = DMC()
        dmc.fit(X_train, y_train)
        y_pred_dmc = dmc.predict(X_test)
        acc_dmc = accuracy_score(y_test, y_pred_dmc)
        results['DMC']['accuracies'].append(acc_dmc)
        
        if (real + 1) % 5 == 0:
            print(f"Realização {real + 1:2d}/20 - LDA: {acc_lda:.4f} | QDA: {acc_qda:.4f} ")
    
    # Calcular estatísticas
    print("\n" + "=" * 80)
    print("RESULTADOS FINAIS")
    print("=" * 80)
    
    for method in ['LDA', 'QDA', 'KNN', 'DMC']:
        accs = np.array(results[method]['accuracies'])
        mean_acc = accs.mean()
        std_acc = accs.std()
        
        print(f"\n{method}:")
        print(f"  Acurácia Média:       {mean_acc:.4f}")
        print(f"  Desvio Padrão:        {std_acc:.4f}")
        print(f"  Acurácia Mínima:      {accs.min():.4f}")
        print(f"  Acurácia Máxima:      {accs.max():.4f}")
    
    # Melhor realização
    print(f"\n" + "-" * 80)
    print(f"Melhor realização: {best_realization['method']} com acurácia {best_realization['accuracy']:.4f}")
    print("-" * 80)
    
    # Matriz de confusão
    cm = confusion_matrix(best_realization['y_test'], best_realization['y_pred'])
    print(f"\nMatriz de Confusão - {best_realization['method']}:")
    print(cm)
    
    # Plotar matriz de confusão
    fig, ax = plt.subplots(figsize=(12, 10))
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
        best_realization['X_train'], best_realization['y_train'],
        best_realization['X_test'], best_realization['y_test'],
        best_realization['clf'],
        f'Superfície de Decisão - {best_realization["method"]} (Dermatology)\n{FEAT_NAMES[F1]} vs {FEAT_NAMES[F2]}',
        'superficies_decisao.png'
    )
    
    print("Superfícies de decisão salvas em: superficies_decisao.png")
    
    # Comparação gráfica
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Boxplot
    methods_list = ['LDA', 'QDA',  'KNN', 'DMC']
    data = [results[m]['accuracies'] for m in methods_list]
    axes[0].boxplot(data, tick_labels=methods_list)
    axes[0].set_ylabel('Acurácia', fontsize=12)
    axes[0].set_title('Distribuição de Acurácias (20 realizações)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Barplot
    means = [np.mean(results[m]['accuracies']) for m in methods_list]
    stds = [np.std(results[m]['accuracies']) for m in methods_list]
    
    x_pos = np.arange(len(methods_list))
    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    axes[1].bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, color=colors_bar)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(methods_list, rotation=45, ha='right')
    axes[1].set_ylabel('Acurácia', fontsize=12)
    axes[1].set_title('Acurácia Média ± Desvio Padrão', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[1].text(i, m + s + 0.01, f'{m:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comparacao_acuracia.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Comparação de acurácias salva em: comparacao_acuracia.png")
    
    print("\n" + "=" * 80)
    print("Análise concluída!")
    print("=" * 80)


if __name__ == '__main__':
    main()
