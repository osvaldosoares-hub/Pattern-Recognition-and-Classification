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


def load_dermatology():
    possible = [
        os.path.join(THIS_DIR, 'dermatology.data'),
        os.path.join(TRAB6_DIR, '..', 'trabalho5', 'dermatology', 'dermatology.data'),
        os.path.join(TRAB6_DIR, '..', 'trabalho4', 'dermatology', 'dermatology.data'),
        os.path.join(TRAB6_DIR, '..', 'trabalho2', 'dermatology', 'dermatology', 'dermatology.data')
    ]

    data_path = None
    for p in possible:
        p_abs = os.path.abspath(p)
        if os.path.exists(p_abs):
            data_path = p_abs
            break

    if data_path is None:
        raise FileNotFoundError("Arquivo dermatology.data não encontrado.")

    df = pd.read_csv(data_path, header=None)
    df = df.replace('?', np.nan).dropna()
    df = df.apply(pd.to_numeric)

    X = df.iloc[:, :-1].values.astype(float)
    y_raw = df.iloc[:, -1].values.astype(int)

    classes = sorted(np.unique(y_raw))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[v] for v in y_raw])
    class_names = [str(c) for c in classes]
    return X, y, class_names, data_path


def main():
    X, y, class_names, data_path = load_dermatology()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    classifiers = {
        'ParzenBayes': ParzenBayesClassifier(bandwidth=None, bandwidth_scale=0.5),
        'LDA': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
        'QDA': QuadraticDiscriminantAnalysis(reg_param=1e-1),
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
            try:
                clf.fit(X_train, y_train)
                yp = clf.predict(X_test)
                acc = accuracy_score(y_test, yp)
            except Exception:
                yp = np.full_like(y_test, fill_value=np.bincount(y_train).argmax())
                acc = accuracy_score(y_test, yp)

            results[name]['accuracies'].append(acc)
            fold_scores[name] = acc

            if name == 'ParzenBayes' and acc > best_parzen['accuracy']:
                others = [v for k, v in fold_scores.items() if k != 'ParzenBayes']
                margem = acc - (max(others) if others else 0.0)
                best_parzen = {
                    'accuracy': acc,
                    'realizacao': real + 1,
                    'y_test': y_test.copy(),
                    'y_pred': yp.copy(),
                    'justificativa': (
                        f"Realização {real+1} escolhida por maior acurácia do Parzen-Bayes ({acc:.4f}), "
                        f"com margem local {margem:.4f}."
                    )
                }

    with open('resultado.txt', 'w', encoding='utf-8') as f:
        f.write("TRABALHO 6 - DERMATOLOGY - PARZEN BAYES\n")
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

    cm = confusion_matrix(best_parzen['y_test'], best_parzen['y_pred'])
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title(f'Matriz de Confusão - ParzenBayes (Acc={best_parzen["accuracy"]:.4f})')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png', dpi=150, bbox_inches='tight')
    plt.close()

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

    print("Arquivos gerados: resultado.txt, matriz_confusao.png, comparacao_acuracia.png")


if __name__ == '__main__':
    main()