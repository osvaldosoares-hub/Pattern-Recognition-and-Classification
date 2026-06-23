import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAB6_DIR = os.path.dirname(THIS_DIR)
if TRAB6_DIR not in sys.path:
    sys.path.insert(0, TRAB6_DIR)

from parzen_bayes import ParzenBayesClassifier

N_REAL = 20
TEST_SIZE = 0.2
RANDOM_STATE_BASE = 42
DATASET_SEED = 42
FEAT_NAMES = ['Atributo 1', 'Atributo 2']
CLASS_NAMES = ['Classe 0', 'Classe 1']
F1, F2 = 0, 1


class DMC:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = {c: X[y == c].mean(axis=0) for c in self.classes_}
        return self

    def predict(self, X):
        preds = []
        for x in X:
            d = {c: np.linalg.norm(x - self.centroids_[c]) for c in self.classes_}
            preds.append(min(d, key=d.get))
        return np.array(preds)


def make_dataset(seed=DATASET_SEED):
    rng = np.random.default_rng(seed)

    # Classe 0 (3 padrões com 30 pontos cada) em (0,4), (0,0), (4,0)
    X0_a = rng.multivariate_normal(
        mean=[0.0, 4.0], cov=[[0.05, 0.00], [0.00, 0.05]], size=30
    )
    X0_b = rng.multivariate_normal(
        mean=[0.0, 0.0], cov=[[0.05, 0.00], [0.00, 0.05]], size=30
    )
    X0_c = rng.multivariate_normal(
        mean=[4.0, 0.0], cov=[[0.05, 0.00], [0.00, 0.05]], size=30
    )
    X0 = np.vstack([X0_a, X0_b, X0_c])
    y0 = np.zeros(X0.shape[0], dtype=int)

    # Classe 1 (1 padrão com 40 pontos) em (4,4)
    X1 = rng.multivariate_normal(
        mean=[4.0, 4.0], cov=[[0.06, 0.00], [0.00, 0.06]], size=40
    )
    y1 = np.ones(X1.shape[0], dtype=int)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    return X, y


def plot_reference_style(X, y, filename='grafico_referencia.png'):
    centers = {
        'Classe 0 - Foco (0,0)': np.array([0.0, 0.0]),
        'Classe 0 - Foco (0,4)': np.array([0.0, 4.0]),
        'Classe 0 - Foco (4,0)': np.array([4.0, 0.0]),
        'Classe 1 - Foco (4,4)': np.array([4.0, 4.0]),
    }
    X0 = X[y == 0]
    X1 = X[y == 1]

    c00, c04, c40 = centers['Classe 0 - Foco (0,0)'], centers['Classe 0 - Foco (0,4)'], centers['Classe 0 - Foco (4,0)']
    d00 = np.linalg.norm(X0 - c00, axis=1)
    d04 = np.linalg.norm(X0 - c04, axis=1)
    d40 = np.linalg.norm(X0 - c40, axis=1)
    g = np.argmin(np.c_[d00, d04, d40], axis=1)

    X0_00 = X0[g == 0]
    X0_04 = X0[g == 1]
    X0_40 = X0[g == 2]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X0_00[:, 0], X0_00[:, 1], c='#3b4eff', s=38, alpha=0.85, label='Classe 0 - Foco (0,0)')
    ax.scatter(X0_40[:, 0], X0_40[:, 1], c='#56f0ff', s=38, alpha=0.85, label='Classe 0 - Foco (4,0)')
    ax.scatter(X0_04[:, 0], X0_04[:, 1], c='#38c9ff', s=38, alpha=0.85, label='Classe 0 - Foco (0,4)')
    ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='^', s=46, alpha=0.90, label='Classe 1 - Foco (4,4)')

    for k, c in centers.items():
        ax.scatter(c[0], c[1], c='black', marker='x', s=180, linewidths=2.5, label=None)

    ax.axhline(0, color='gray', lw=1, alpha=0.7)
    ax.axvline(0, color='gray', lw=1, alpha=0.7)
    ax.plot([-0.2, 5.0], [4.3, -0.5], color='gray', lw=1.2, alpha=0.85)

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Artificial I - Gráfico de Referência', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.25)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_decision_surface(X_train, y_train, X_test, y_test, clf, title, filename):
    h = 0.02
    x_min, x_max = X_train[:, F1].min() - 0.6, X_train[:, F1].max() + 0.6
    y_min, y_max = X_train[:, F2].min() - 0.6, X_train[:, F2].max() + 0.6
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, levels=np.arange(-0.5, 2, 1), alpha=0.28, cmap='viridis')
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.2, alpha=0.9)

    for c in np.unique(y_train):
        m = (y_train == c)
        ax.scatter(X_train[m, F1], X_train[m, F2], marker='o', s=55, alpha=0.80, edgecolors='k',
                   label=f'{CLASS_NAMES[int(c)]} - treino')

    for c in np.unique(y_test):
        m = (y_test == c)
        ax.scatter(X_test[m, F1], X_test[m, F2], marker='^', s=70, alpha=0.95, edgecolors='k',
                   label=f'{CLASS_NAMES[int(c)]} - teste')

    ax.set_xlabel(FEAT_NAMES[F1])
    ax.set_ylabel(FEAT_NAMES[F2])
    ax.set_title(title, fontweight='bold')
    ax.grid(alpha=0.25, linestyle=':')
    ax.legend(loc='best', fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs('.', exist_ok=True)

    X, y = make_dataset()
    dataset_csv = np.column_stack([X, y])
    np.savetxt('artificial_I.csv', dataset_csv, delimiter=',', header='x1,x2,class', comments='')
    plot_reference_style(X, y, filename='grafico_referencia.png')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ParzenBayes com bandwidth ajustado para este dataset
    classifiers = {
        'ParzenBayes': ParzenBayesClassifier(bandwidth=None, bandwidth_scale=0.5),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(reg_param=1e-4),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DMC': DMC(),
    }

    results = {name: {'accuracies': []} for name in classifiers}

    best_parzen = {
        'accuracy': -1.0,
        'realizacao': None,
        'X_train': None, 'X_test': None,
        'y_train': None, 'y_test': None, 'y_pred': None,
        'justificativa': ''
    }

    for real in range(N_REAL):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_BASE + real, stratify=y
        )

        local_acc = {}

        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name]['accuracies'].append(acc)
            local_acc[name] = acc

            if name == 'ParzenBayes' and acc > best_parzen['accuracy']:
                margem = acc - max([v for k, v in local_acc.items() if k != 'ParzenBayes'] + [0.0])
                best_parzen.update({
                    'accuracy': acc,
                    'realizacao': real + 1,
                    'X_train': X_train.copy(),
                    'X_test': X_test.copy(),
                    'y_train': y_train.copy(),
                    'y_test': y_test.copy(),
                    'y_pred': y_pred.copy(),
                    'justificativa': (
                        f"Realização {real+1} escolhida por apresentar a maior acurácia do Parzen-Bayes "
                        f"({acc:.4f}). Margem parcial observada para os demais métodos: {margem:.4f}."
                    )
                })

    # Resultado textual
    with open('resultado.txt', 'w', encoding='utf-8') as f:
        f.write("TRABALHO 6 - ARTIFICIAL I - PARZEN BAYES\n")
        f.write("=" * 60 + "\n\n")
        for name in classifiers:
            accs = np.array(results[name]['accuracies'])
            f.write(f"{name}:\n")
            f.write(f"  Acurácia média: {accs.mean():.4f}\n")
            f.write(f"  Desvio padrão : {accs.std():.4f}\n")
            f.write(f"  Mín/Máx       : {accs.min():.4f} / {accs.max():.4f}\n\n")

        f.write("Matriz de confusão (realização escolhida):\n")
        f.write(best_parzen['justificativa'] + "\n")

    # Matriz de confusão
    cm = confusion_matrix(best_parzen['y_test'], best_parzen['y_pred'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title(f'Matriz de Confusão - ParzenBayes (Acc={best_parzen["accuracy"]:.4f})')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Superfície de decisão
    clf_plot = ParzenBayesClassifier(bandwidth=None, bandwidth_scale=0.5)
    clf_plot.fit(best_parzen['X_train'], best_parzen['y_train'])
    plot_decision_surface(
        best_parzen['X_train'], best_parzen['y_train'],
        best_parzen['X_test'], best_parzen['y_test'],
        clf_plot,
        'Superfície de decisão - Parzen-Bayes (Artificial I)',
        'superficies_decisao.png'
    )

    # Comparação de acurácia
    methods = list(classifiers.keys())
    data = [results[m]['accuracies'] for m in methods]
    means = [np.mean(results[m]['accuracies']) for m in methods]
    stds = [np.std(results[m]['accuracies']) for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].boxplot(data, tick_labels=methods)
    axes[0].set_title('Distribuição de Acurácia (20 realizações)')
    axes[0].set_ylabel('Acurácia')
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=20)

    x = np.arange(len(methods))
    axes[1].bar(x, means, yerr=stds, capsize=8, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20)
    axes[1].set_title('Acurácia média ± desvio padrão')
    axes[1].set_ylabel('Acurácia')
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('comparacao_acuracia.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Arquivos gerados: artificial_I.csv, grafico_referencia.png, resultado.txt, matriz_confusao.png, superficies_decisao.png, comparacao_acuracia.png")


if __name__ == '__main__':
    main()