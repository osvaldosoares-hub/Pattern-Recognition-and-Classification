"""
Demonstração do Grid Search com K-Fold CV para Parzen Bayes em todos os datasets.

FLUXO SEM VAZAMENTO (passo a passo):
1. Carrega dataset completo
2. Separa TREINO/TESTE (holdout 80/20) -> TESTE FICA CONGELADO
3. Aplica StandardScaler APENAS no TREINO (fit), transforma ambos
4. No TREINO: K-Fold CV para testar vários bandwidth_scale
   - Cada fold: treina em K-1 partes, valida na 1 parte restante
   - O TESTE NUNCA É USADO aqui
5. Escolhe o melhor bandwidth_scale pela média da CV
6. Retreina ParzenBayes com o melhor scale no TREINO completo
7. Avalia no TESTE (única vez que o teste é usado)

POR QUE NÃO VAZA DADOS?
- O conjunto de teste é separado ANTES do CV e nunca participa da escolha
- O CV opera exclusivamente nos dados de treino
- O scaler é ajustado só no treino (não vê estatísticas do teste)
- A avaliação final no teste é uma estimativa honesta da generalização
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from parzen_bayes import ParzenBayesClassifier
from grid_search_parzen import grid_search_cv, DEFAULT_BANDWIDTH_SCALES

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

np.set_printoptions(precision=4, suppress=True)


def plot_cv_scores(cv_scores, dataset_name, filename):
    """Plota os scores de CV para cada bandwidth_scale."""
    scales = sorted(cv_scores.keys())
    means = [np.mean(cv_scores[s]) for s in scales]
    stds = [np.std(cv_scores[s]) for s in scales]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(range(len(scales)), means, yerr=stds, fmt='o-', capsize=6, capthick=2, markersize=8, linewidth=2)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f'{s:.2f}' for s in scales], rotation=45)
    ax.set_xlabel('bandwidth_scale')
    ax.set_ylabel('Acurácia média no CV (5 folds)')
    ax.set_title(f'Grid Search - {dataset_name}\nMelhor scale: {scales[np.argmax(means)]:.4f} (CV={max(means):.4f})', fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':')
    ax.axhline(y=max(means), color='red', linestyle='--', alpha=0.7, label=f'Melhor: {max(means):.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, acc, dataset_name, filename):
    """Plota matriz de confusão."""
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.5), max(5, len(class_names) * 1.2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')
    ax.set_title(f'Matriz de Confusão - ParzenBayes tuned ({dataset_name})\nAcc={acc:.4f}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def run_grid_search_cv(name, X, y, class_names, feat_names=None, test_size=0.2, random_state=42, output_dir='.'):
    """
    Executa Grid Search com K-Fold CV para um dataset.
    Fluxo sem vazamento documentado em cada passo.
    Salva resultados em arquivos.
    """
    print(f"\n{'='*70}")
    print(f"  DATASET: {name}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # PASSO 1: Separar TREINO e TESTE (holdout)
    # O TESTE é separado ANTES de qualquer tuning
    # e fica CONGELADO até o final
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Amostras: {len(X)} | Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"  Classes: {class_names}")
    if feat_names:
        print(f"  Atributos: {feat_names}")
    print(f"  Dimensões: {X.shape[1]}")

    # ------------------------------------------------------------------
    # PASSO 2: Normalização (StandardScaler)
    # Ajusta APENAS no TREINO, transforma ambos
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n  >>> [PASSO 1] Holdout: Treino={len(X_train)}, Teste={len(X_test)}")
    print(f"      Teste congelado até o fim.")
    print(f"  >>> [PASSO 2] StandardScaler: fit no treino, transform em ambos.")

    # ------------------------------------------------------------------
    # PASSO 3: Grid Search com K-Fold CV (APENAS NO TREINO)
    # K-Fold divide o TREINO em K partes
    # A cada iteração: treina em K-1, valida em 1
    # NUNCA toca no TESTE
    # ------------------------------------------------------------------
    n_folds = 5
    print(f"\n  >>> [PASSO 3] Grid Search com K={n_folds}-Fold CV (apenas no TREINO)")

    best_scale, cv_scores, cv_mean = grid_search_cv(
        X_train_scaled, y_train,
        bandwidth_scales=DEFAULT_BANDWIDTH_SCALES,
        n_folds=n_folds,
        random_state=random_state
    )

    print(f"\n      Grid testado: {len(DEFAULT_BANDWIDTH_SCALES)} valores de bandwidth_scale")
    print(f"      Melhor bandwidth_scale: {best_scale:.4f}")
    print(f"      Acurácia média no CV (treino): {cv_mean:.4f}")
    print(f"      Scores por fold do melhor scale: {cv_scores[best_scale]}")

    sorted_scales = sorted(cv_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
    print(f"\n      Top 3 bandwidth_scale pela CV:")
    for i, (s, scores) in enumerate(sorted_scales[:3]):
        print(f"        {i+1}. scale={s:.4f} -> média={np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # ------------------------------------------------------------------
    # PASSO 4: Retreinar com melhor scale no TREINO completo
    # ------------------------------------------------------------------
    print(f"\n  >>> [PASSO 4] Retreinando ParzenBayes com scale={best_scale:.4f} no TREINO completo")

    best_clf = ParzenBayesClassifier(bandwidth=None, bandwidth_scale=best_scale)
    best_clf.fit(X_train_scaled, y_train)

    bandwidths_str = ""
    for c in best_clf.classes_:
        c_name = class_names[int(c)] if class_names and int(c) < len(class_names) else str(c)
        bandwidths_str += f"        Classe {c} ({c_name}): h = {best_clf.bandwidths_[c]:.6f}\n"

    print(f"      Bandwidths por classe:")
    print(bandwidths_str, end='')

    # ------------------------------------------------------------------
    # PASSO 5: Avaliar no TESTE (única vez)
    # ------------------------------------------------------------------
    y_pred = best_clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n  >>> [PASSO 5] Avaliação no TESTE (congelado):")
    print(f"      Acurácia no teste: {test_acc:.4f}")

    print(f"\n      Comparação CV vs Teste:")
    print(f"        Estimativa CV (treino): {cv_mean:.4f}")
    print(f"        Performance real (teste): {test_acc:.4f}")
    print(f"        Diferença: {abs(cv_mean - test_acc):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n      Matriz de Confusão (teste):")
    print(f"      {cm}")

    # Comparar com heurística default
    default_clf = ParzenBayesClassifier(bandwidth=None, bandwidth_scale=0.5)
    default_clf.fit(X_train_scaled, y_train)
    y_pred_default = default_clf.predict(X_test_scaled)
    default_acc = accuracy_score(y_test, y_pred_default)

    print(f"\n      Comparação com heurística default (scale=0.5):")
    print(f"        Default (scale=0.5): {default_acc:.4f}")
    print(f"        Tuned (scale={best_scale:.4f}): {test_acc:.4f}")
    print(f"        Ganho: {test_acc - default_acc:.4f}")

    # ------------------------------------------------------------------
    # Salvar resultados em arquivo
    # ------------------------------------------------------------------
    result_filename = os.path.join(output_dir, f'grid_search_{name.lower().replace(" ", "_")}.txt')
    with open(result_filename, 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\n")
        f.write(f"  GRID SEARCH COM K-FOLD CV - {name}\n")
        f.write(f"{'='*70}\n\n")

        f.write(f"Configuração:\n")
        f.write(f"  Dataset: {name}\n")
        f.write(f"  Amostras totais: {len(X)}\n")
        f.write(f"  Treino: {len(X_train)} ({(1-test_size)*100:.0f}%)\n")
        f.write(f"  Teste:  {len(X_test)} ({test_size*100:.0f}%)\n")
        f.write(f"  Classes: {class_names}\n")
        if feat_names:
            f.write(f"  Atributos: {feat_names}\n")
        f.write(f"  Dimensões: {X.shape[1]}\n")
        f.write(f"  K-Folds: {n_folds}\n")
        f.write(f"  Grid testado: {DEFAULT_BANDWIDTH_SCALES}\n\n")

        f.write(f"FLUXO SEM VAZAMENTO:\n")
        f.write(f"  [PASSO 1] Holdout: Treino={len(X_train)}, Teste={len(X_test)}\n")
        f.write(f"            Teste congelado até o fim.\n")
        f.write(f"  [PASSO 2] StandardScaler: fit no treino, transform em ambos.\n")
        f.write(f"  [PASSO 3] Grid Search com {n_folds}-Fold CV (apenas no TREINO)\n\n")

        f.write(f"Resultados do Grid Search:\n")
        f.write(f"  {'bandwidth_scale':<18} {'Média CV':<10} {'Desvio':<10} {'Scores por fold':<30}\n")
        f.write(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*30}\n")
        for s in sorted(cv_scores.keys()):
            scores = cv_scores[s]
            f.write(f"  {s:<18.4f} {np.mean(scores):<10.4f} {np.std(scores):<10.4f} {str([round(x,4) for x in scores]):<30}\n")

        f.write(f"\n  Melhor bandwidth_scale: {best_scale:.4f}\n")
        f.write(f"  Acurácia média no CV (treino): {cv_mean:.4f}\n")
        f.write(f"  Scores por fold do melhor scale: {cv_scores[best_scale]}\n\n")

        f.write(f"  Top 3 bandwidth_scale pela CV:\n")
        for i, (s, scores) in enumerate(sorted_scales[:3]):
            f.write(f"    {i+1}. scale={s:.4f} -> média={np.mean(scores):.4f} ± {np.std(scores):.4f}\n")

        f.write(f"\n[PASSO 4] Retreinado com scale={best_scale:.4f} no TREINO completo\n")
        f.write(f"Bandwidths por classe:\n{bandwidths_str}\n")

        f.write(f"[PASSO 5] Avaliação no TESTE (congelado):\n")
        f.write(f"  Acurácia no teste: {test_acc:.4f}\n\n")

        f.write(f"Comparação CV vs Teste:\n")
        f.write(f"  Estimativa CV (treino): {cv_mean:.4f}\n")
        f.write(f"  Performance real (teste): {test_acc:.4f}\n")
        f.write(f"  Diferença: {abs(cv_mean - test_acc):.4f}\n\n")

        f.write(f"Matriz de Confusão (teste):\n")
        f.write(f"  {cm}\n\n")

        f.write(f"Comparação com heurística default (scale=0.5):\n")
        f.write(f"  Default (scale=0.5): {default_acc:.4f}\n")
        f.write(f"  Tuned (scale={best_scale:.4f}): {test_acc:.4f}\n")
        f.write(f"  Ganho: {test_acc - default_acc:.4f}\n\n")

        f.write(f"{'='*70}\n")
        f.write(f"  POR QUE NÃO HOUVE VAZAMENTO DE DADOS?\n")
        f.write(f"{'='*70}\n")
        f.write(f"1. O conjunto de TESTE foi separado ANTES do Grid Search (PASSO 1)\n")
        f.write(f"2. O K-Fold CV operou EXCLUSIVAMENTE nos dados de TREINO (PASSO 3)\n")
        f.write(f"3. O StandardScaler foi ajustado apenas no TREINO (PASSO 2)\n")
        f.write(f"4. O melhor hiperparâmetro foi escolhido com base APENAS no CV do treino\n")
        f.write(f"5. O TESTE foi usado UMA ÚNICA VEZ no final (PASSO 5) para medir a\n")
        f.write(f"   performance real de generalização, sem ter influenciado nenhuma\n")
        f.write(f"   etapa anterior.\n")

    print(f"\n  => Resultados salvos em: {result_filename}")

    # Plotar gráfico de CV scores
    cv_plot = os.path.join(output_dir, f'grid_search_{name.lower().replace(" ", "_")}_cv.png')
    plot_cv_scores(cv_scores, name, cv_plot)
    print(f"  => Gráfico CV salvo em: {cv_plot}")

    # Plotar matriz de confusão
    cm_plot = os.path.join(output_dir, f'grid_search_{name.lower().replace(" ", "_")}_cm.png')
    plot_confusion_matrix(cm, class_names, test_acc, name, cm_plot)
    print(f"  => Matriz de confusão salva em: {cm_plot}")

    return {
        'best_scale': best_scale,
        'cv_mean': cv_mean,
        'test_acc': test_acc,
        'default_acc': default_acc,
        'cv_scores': cv_scores,
        'bandwidths': best_clf.bandwidths_,
        'cm': cm,
        'result_file': result_filename,
        'cv_plot': cv_plot,
        'cm_plot': cm_plot
    }


def main():
    os.makedirs(THIS_DIR, exist_ok=True)
    results = {}

    # ===== Iris =====
    iris = load_iris()
    results['Iris'] = run_grid_search_cv(
        'Iris', iris.data, iris.target,
        class_names=['setosa', 'versicolor', 'virginica'],
        feat_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
        output_dir=THIS_DIR
    )

    # ===== Artificial I =====
    rng = np.random.default_rng(42)
    X0_a = rng.multivariate_normal([0.0, 4.0], [[0.05, 0.00], [0.00, 0.05]], 30)
    X0_b = rng.multivariate_normal([0.0, 0.0], [[0.05, 0.00], [0.00, 0.05]], 30)
    X0_c = rng.multivariate_normal([4.0, 0.0], [[0.05, 0.00], [0.00, 0.05]], 30)
    X0 = np.vstack([X0_a, X0_b, X0_c])
    y0 = np.zeros(X0.shape[0], dtype=int)
    X1 = rng.multivariate_normal([4.0, 4.0], [[0.06, 0.00], [0.00, 0.06]], 40)
    y1 = np.ones(X1.shape[0], dtype=int)
    X_art = np.vstack([X0, X1])
    y_art = np.concatenate([y0, y1])

    results['Artificial_I'] = run_grid_search_cv(
        'Artificial_I', X_art, y_art,
        class_names=['Classe 0', 'Classe 1'],
        feat_names=['Atributo 1', 'Atributo 2'],
        output_dir=THIS_DIR
    )

    # ===== Vertebral =====
    try:
        import pandas as pd
        possible = [
            os.path.join(THIS_DIR, 'vertebral', 'column_3C.dat'),
            os.path.join(THIS_DIR, '..', 'trabalho5', 'vertebral', 'column_3C.dat'),
            os.path.join(THIS_DIR, '..', 'trabalho4', 'vertebral', 'column_3C.dat'),
        ]
        data_path = None
        for p in possible:
            if os.path.exists(p):
                data_path = p
                break
        if data_path:
            df = pd.read_csv(data_path, sep=r'\s+', header=None)
            X_v = df.iloc[:, :-1].values
            y_raw_v = df.iloc[:, -1].astype(str).values
            classes_v = sorted(np.unique(y_raw_v))
            map_v = {c: i for i, c in enumerate(classes_v)}
            y_v = np.array([map_v[v] for v in y_raw_v])

            results['Vertebral'] = run_grid_search_cv(
                'Vertebral', X_v, y_v,
                class_names=classes_v,
                feat_names=['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
                            'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis'],
                output_dir=THIS_DIR
            )
    except Exception as e:
        print(f"\n  [!] Vertebral: {e}")

    # ===== Breast Cancer =====
    try:
        import pandas as pd
        possible_bc = [
            os.path.join(THIS_DIR, 'breast_cancer', 'wdbc.data'),
            os.path.join(THIS_DIR, '..', 'trabalho5', 'breast_cancer', 'wdbc.data'),
            os.path.join(THIS_DIR, '..', 'trabalho4', 'breast_cancer', 'wdbc.data'),
        ]
        data_path_bc = None
        for p in possible_bc:
            if os.path.exists(p):
                data_path_bc = p
                break
        if data_path_bc:
            df_bc = pd.read_csv(data_path_bc, header=None)
            y_raw_bc = df_bc.iloc[:, 1].values
            X_bc = df_bc.iloc[:, 2:].values.astype(float)
            classes_bc = sorted(np.unique(y_raw_bc))
            map_bc = {c: i for i, c in enumerate(classes_bc)}
            y_bc = np.array([map_bc[v] for v in y_raw_bc])

            results['Breast_Cancer'] = run_grid_search_cv(
                'Breast_Cancer', X_bc, y_bc,
                class_names=classes_bc,
                output_dir=THIS_DIR
            )
    except Exception as e:
        print(f"\n  [!] Breast Cancer: {e}")

    # ===== Dermatology =====
    try:
        import pandas as pd
        possible_d = [
            os.path.join(THIS_DIR, 'dermatology', 'dermatology.data'),
            os.path.join(THIS_DIR, '..', 'trabalho5', 'dermatology', 'dermatology.data'),
            os.path.join(THIS_DIR, '..', 'trabalho4', 'dermatology', 'dermatology.data'),
        ]
        data_path_d = None
        for p in possible_d:
            if os.path.exists(p):
                data_path_d = p
                break
        if data_path_d:
            df_d = pd.read_csv(data_path_d, header=None)
            df_d = df_d.replace('?', np.nan).dropna()
            df_d = df_d.apply(pd.to_numeric)
            X_d = df_d.iloc[:, :-1].values.astype(float)
            y_raw_d = df_d.iloc[:, -1].values.astype(int)
            classes_d = sorted(np.unique(y_raw_d))
            map_d = {c: i for i, c in enumerate(classes_d)}
            y_d = np.array([map_d[v] for v in y_raw_d])

            results['Dermatology'] = run_grid_search_cv(
                'Dermatology', X_d, y_d,
                class_names=[str(c) for c in classes_d],
                output_dir=THIS_DIR
            )
    except Exception as e:
        print(f"\n  [!] Dermatology: {e}")

    # ===== Resumo final =====
    print(f"\n{'='*70}")
    print(f"  RESUMO FINAL - GRID SEARCH COM K-FOLD CV")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'Best Scale':<12} {'CV Score':<10} {'Test Score':<12} {'Default':<10} {'Ganho':<10}")
    print(f"{'-'*20} {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    for name, res in results.items():
        ganho = res['test_acc'] - res['default_acc']
        print(f"{name:<20} {res['best_scale']:<12.4f} {res['cv_mean']:<10.4f} "
              f"{res['test_acc']:<12.4f} {res['default_acc']:<10.4f} {ganho:<+10.4f}")

    # Salvar resumo geral
    summary_file = os.path.join(THIS_DIR, 'grid_search_RESUMO.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\n")
        f.write(f"  RESUMO FINAL - GRID SEARCH COM K-FOLD CV (Parzen Bayes)\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"  Grid testado: {DEFAULT_BANDWIDTH_SCALES}\n")
        f.write(f"  K-Folds: 5\n")
        f.write(f"  Holdout: 80% treino / 20% teste\n\n")
        f.write(f"  {'Dataset':<20} {'Best Scale':<12} {'CV Score':<10} {'Test Score':<12} {'Default':<10} {'Ganho':<10}\n")
        f.write(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*10}\n")
        for name, res in results.items():
            ganho = res['test_acc'] - res['default_acc']
            f.write(f"  {name:<20} {res['best_scale']:<12.4f} {res['cv_mean']:<10.4f} "
                    f"{res['test_acc']:<12.4f} {res['default_acc']:<10.4f} {ganho:<+10.4f}\n")

        f.write(f"\n{'='*70}\n")
        f.write(f"  CONCLUSÃO\n")
        f.write(f"{'='*70}\n")
        f.write(f"  O Grid Search com K-Fold CV ajustou o bandwidth_scale usando\n")
        f.write(f"  APENAS os dados de TREINO. O TESTE ficou congelado até o final,\n")
        f.write(f"  garantindo uma avaliação SEM VAZAMENTO de dados.\n\n")
        f.write(f"  Explicação do fluxo sem vazamento:\n")
        f.write(f"  1. Holdout: Teste separado ANTES do tuning\n")
        f.write(f"  2. StandardScaler: fit apenas no treino\n")
        f.write(f"  3. K-Fold CV: opera exclusivamente no treino\n")
        f.write(f"  4. Escolha do best scale: baseada apenas na média CV\n")
        f.write(f"  5. Avaliação final: teste usado UMA única vez\n")

    print(f"\n  => Resumo salvo em: {summary_file}")

    # Plot comparativo final
    names = list(results.keys())
    best_scales = [results[n]['best_scale'] for n in names]
    cv_scores_mean = [results[n]['cv_mean'] for n in names]
    test_scores = [results[n]['test_acc'] for n in names]
    default_scores = [results[n]['default_acc'] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(names))
    w = 0.25

    axes[0].bar(x - w, cv_scores_mean, w, alpha=0.8, label='CV Score (treino)', color='steelblue')
    axes[0].bar(x, test_scores, w, alpha=0.8, label='Test Score', color='orange')
    axes[0].bar(x + w, default_scores, w, alpha=0.8, label='Default (scale=0.5)', color='lightgreen')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30)
    axes[0].set_ylabel('Acurácia')
    axes[0].set_title('Comparação: CV vs Teste vs Default', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')

    axes[1].bar(x, best_scales, alpha=0.7, color='crimson')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30)
    axes[1].set_ylabel('Melhor bandwidth_scale')
    axes[1].set_title('Melhor scale escolhido por CV', fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(THIS_DIR, 'grid_search_comparacao_final.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Gráfico comparativo salvo em: grid_search_comparacao_final.png")

    print(f"\n{'='*70}")
    print(f"  CONCLUSÃO: O Grid Search com K-Fold CV ajustou o bandwidth_scale")
    print(f"  usando APENAS os dados de TREINO. O TESTE ficou congelado até o")
    print(f"  final, garantindo uma avaliação SEM VAZAMENTO de dados.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()