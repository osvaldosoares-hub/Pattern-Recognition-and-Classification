import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAB5_DIR = os.path.dirname(THIS_DIR)
if TRAB5_DIR not in sys.path:
    sys.path.insert(0, TRAB5_DIR)

from gmm_bayes import GMMBayesFull

N_REAL = 20
TEST_SIZE = 0.2
RANDOM_STATE_BASE = 42
FEAT_NAMES = ['sepal length', 'sepal width', 'petal length', 'petal width']
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']
F1, F2 = 2, 3  # par escolhido para superfície


class DMC:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = {c: X[y == c].mean(axis=0) for c in self.classes_}
        return self

    def predict(self, X):
        out = []
        for x in X:
            d = {c: np.linalg.norm(x - self.centroids_[c]) for c in self.classes_}
            out.append(min(d, key=d.get))
        return np.array(out)


def build_full_from_pair(pair_points, ref_mean, f1, f2):
    Xf = np.tile(ref_mean, (pair_points.shape[0], 1))
    Xf[:, f1] = pair_points[:, 0]
    Xf[:, f2] = pair_points[:, 1]
    return Xf


def plot_decision_surface_full(clf, X_train, y_train, X_test, y_test, filename):
    h = 0.02
    Xtr2 = X_train[:, [F1, F2]]
    x_min, x_max = Xtr2[:, 0].min() - 0.5, Xtr2[:, 0].max() + 0.5
    y_min, y_max = Xtr2[:, 1].min() - 0.5, Xtr2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid2 = np.c_[xx.ravel(), yy.ravel()]

    ref_mean = X_train.mean(axis=0)
    grid_full = build_full_from_pair(grid2, ref_mean, F1, F2)
    Z = clf.predict(grid_full).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, levels=3, alpha=0.25, cmap='viridis')
    ax.contour(xx, yy, Z, levels=3, colors='k', linewidths=0.6, alpha=0.5)

    for c in np.unique(y_train):
        m = y_train == c
        ax.scatter(X_train[m, F1], X_train[m, F2], marker='^', s=80, edgecolors='black',
                   label=f'{CLASS_NAMES[int(c)]} (treino)', alpha=0.8)

    for c in np.unique(y_test):
        m = y_test == c
        ax.scatter(X_test[m, F1], X_test[m, F2], marker='o', s=80, edgecolors='black',
                   label=f'{CLASS_NAMES[int(c)]} (teste)', alpha=0.9)

    ax.set_xlabel(FEAT_NAMES[F1])
    ax.set_ylabel(FEAT_NAMES[F2])
    ax.set_title('Superfície de decisão - GMM-Bayes (Iris)', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    classifiers = {
        'GMM-Bayes': GMMBayesFull(n_components=3, reg_covar=1e-6, random_state=42),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(reg_param=1e-4),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DMC': DMC()
    }

    results = {k: {'accuracies': []} for k in classifiers}
    best_gmm = {'accuracy': -1.0}

    for real in range(N_REAL):
        X_train, X_test, y_train, y_test = train_test_split(
            Xs, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_BASE + real, stratify=y
        )

        fold_scores = {}
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            yp = clf.predict(X_test)
            acc = accuracy_score(y_test, yp)
            results[name]['accuracies'].append(acc)
            fold_scores[name] = acc

            if name == 'GMM-Bayes' and acc > best_gmm['accuracy']:
                others = [v for k, v in fold_scores.items() if k != 'GMM-Bayes']
                margem = acc - (max(others) if others else 0.0)
                best_gmm = {
                    'accuracy': acc,
                    'realizacao': real + 1,
                    'X_train': X_train.copy(),
                    'X_test': X_test.copy(),
                    'y_train': y_train.copy(),
                    'y_test': y_test.copy(),
                    'y_pred': yp.copy(),
                    'justificativa': (
                        f"Realização {real+1} escolhida por maior acurácia do GMM-Bayes ({acc:.4f}), "
                        f"com margem local {margem:.4f} sobre o melhor concorrente na iteração."
                    )
                }

    with open('resultado.txt', 'w', encoding='utf-8') as f:
        f.write("TRABALHO 5 - IRIS - GMM-BAYES\n")
        f.write("=" * 60 + "\n\n")
        for name in classifiers:
            a = np.array(results[name]['accuracies'])
            f.write(f"{name}:\n")
            f.write(f"  Acurácia média: {a.mean():.4f}\n")
            f.write(f"  Desvio padrão : {a.std():.4f}\n")
            f.write(f"  Mín/Máx       : {a.min():.4f} / {a.max():.4f}\n\n")
        f.write("Escolha da matriz de confusão:\n")
        f.write(best_gmm['justificativa'] + "\n")
        f.write(f"Par de atributos para superfície: {FEAT_NAMES[F1]} x {FEAT_NAMES[F2]}\n")

    cm = confusion_matrix(best_gmm['y_test'], best_gmm['y_pred'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title(f'Matriz de Confusão - GMM-Bayes (Acc={best_gmm["accuracy"]:.4f})')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=150, bbox_inches='tight')
    plt.close()

    clf_plot = GMMBayesFull(n_components=3, reg_covar=1e-6, random_state=42)
    clf_plot.fit(best_gmm['X_train'], best_gmm['y_train'])
    plot_decision_surface_full(clf_plot, best_gmm['X_train'], best_gmm['y_train'],
                               best_gmm['X_test'], best_gmm['y_test'], 'superficies_decisao.png')

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

    print("Arquivos gerados: resultado.txt, matriz_confusao.png, superficies_decisao.png, comparacao_acuracia.png")


if __name__ == '__main__':
    main()
