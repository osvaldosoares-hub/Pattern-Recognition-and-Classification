import numpy as np
import pandas as pd
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

F1, F2 = 0, 2
FEAT_NAMES = [
    'pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
    'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis'
]


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


def load_vertebral_data():
    possible = [
        os.path.join(THIS_DIR, 'column_3C.dat'),
        os.path.join(TRAB6_DIR, '..', 'trabalho5', 'vertebral', 'column_3C.dat'),
        os.path.join(TRAB6_DIR, '..', 'trabalho4', 'vertebral', 'column_3C.dat'),
        os.path.join(TRAB6_DIR, '..', 'trabalho2', 'Vertebral', 'vertebral+column', 'column_3C.dat')
    ]

    data_path = None
    for p in possible:
        p_abs = os.path.abspath(p)
        if os.path.exists(p_abs):
            data_path = p_abs
            break

    if data_path is None:
        raise FileNotFoundError("Arquivo column_3C.dat não encontrado.")

    df = pd.read_csv(data_path, sep=r'\s+', header=None)
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].astype(str).values

    classes = sorted(np.unique(y_raw))
    map_lbl = {c: i for i, c in enumerate(classes)}
    y = np.array([map_lbl[v] for v in y_raw])
    class_names = classes
    return X, y, class_names, data_path


def build_full_from_pair(pair_points, ref_mean, f1, f2):
    Xf = np.tile(ref_mean, (pair_points.shape[0], 1))
    Xf[:, f1] = pair_points[:, 0]
    Xf[:, f2] = pair_points[:, 1]
    return Xf


def plot_decision_surface_full(clf, X_train, y_train, X_test, y_test, class_names, filename):
    h = 0.03
    Xtr2 = X_train[:, [F1, F2]]
    x_min, x_max = Xtr2[:, 0].min() - 0.5, Xtr2[:, 0].max() + 0.5
    y_min, y_max = Xtr2[:, 1].min() - 0.5, Xtr2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid2 = np.c_[xx.ravel(), yy.ravel()]

    ref_mean = X_train.mean(axis=0)
    grid_full = build_full_from_pair(grid2, ref_mean, F1, F2)
    Z = clf.predict(grid_full).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, levels=len(class_names), alpha=0.25, cmap='viridis')
    ax.contour(xx, yy, Z, levels=len(class_names), colors='k', linewidths=0.6, alpha=0.5)

    for c in np.unique(y_train):
        m = y_train == c
        ax.scatter(X_train[m, F1], X_train[m, F2], marker='^', s=70, edgecolors='black',
                   label=f'{class_names[int(c)]} (treino)', alpha=0.8)

    for c in np.unique(y_test):
        m = y_test == c
        ax.scatter(X_test[m, F1], X_test[m, F2], marker='o', s=70, edgecolors='black',
                   label=f'{class_names[int(c)]} (teste)', alpha=0.9)

    ax.set_xlabel(FEAT_NAMES[F1])
    ax.set_ylabel(FEAT_NAMES[F2])
    ax.set_title('Superfície de decisão - Parzen-Bayes (Vertebral)', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    X, y, class_names, data_path = load_vertebral_data()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    classifiers = {
        'ParzenBayes': ParzenBayesClassifier(bandwidth=None, bandwidth_scale=0.5),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(reg_param=1e-4),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DMC': DMC()
    }

    results = {k: {'accuracies': []} for k in classifiers}
    best_parzen = {'accuracy': -1.0}

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

            if name == 'ParzenBayes' and acc > best_parzen['accuracy']:
                others = [v for k, v in fold_scores.items() if k != 'ParzenBayes']
                margem = acc - (max(others) if others else 0.0)
                best_parzen = {
                    'accuracy': acc,
                    'realizacao': real + 1,
                    'X_train': X_train.copy(),
                    'X_test': X_test.copy(),
                    'y_train': y_train.copy(),
                    'y_test': y_test.copy(),
                    'y_pred': yp.copy(),
                    'justificativa': (
                        f"Realização {real+1} escolhida por maior acurácia do Parzen-Bayes ({acc:.4f}), "
                        f"com margem local {margem:.4f}."
                    )
                }

    with open('resultado.txt', 'w', encoding='utf-8') as f:
        f.write("TRABALHO 6 - VERTEBRAL - PARZEN BAYES\n")
        f.write("=" * 60 + "\n")
        f.write(f"Fonte dos dados: {data_path}\n\n")
        for name in classifiers:
            a = np.array(results[name]['accuracies'])
            f.write(f"{name}:\n")
            f.write(f"  Acurácia média: {a.mean():.4f}\n")
            f.write(f"  Desvio padrão : {a.std():.4f}\n")
            f.write(f"  Mín/Máx       : {a.min():.4f} / {a.max():.4f}\n\n")
        f.write("Escolha da matriz de confusão:\n")
        f.write(best_parzen['justificativa'] + "\n")
        f.write(f"Par de atributos para superfície: {FEAT_NAMES[F1]} x {FEAT_NAMES[F2]}\n")

    cm = confusion_matrix(best_parzen['y_test'], best_parzen['y_pred'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title(f'Matriz de Confusão - ParzenBayes (Acc={best_parzen["accuracy"]:.4f})')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=150, bbox_inches='tight')
    plt.close()

    clf_plot = ParzenBayesClassifier(bandwidth=None, bandwidth_scale=0.5)
    clf_plot.fit(best_parzen['X_train'], best_parzen['y_train'])
    plot_decision_surface_full(clf_plot, best_parzen['X_train'], best_parzen['y_train'],
                               best_parzen['X_test'], best_parzen['y_test'], class_names,
                               'superficies_decisao.png')

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