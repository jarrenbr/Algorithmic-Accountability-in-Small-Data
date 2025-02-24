import time
from collections import defaultdict

import folktables as ft
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics as skm

import common as cmn
import folktables_data as ft_data
from binary_confusion_matrix import BinaryConfusionMatrix
from common import DEBUG

np.random.seed(0)


def efficient_confusion_matrix(y_true, y_pred):
    # Optimized confusion matrix calculation for binary classification
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[TP, FN], [FP, TN]])


def get_rnd_cms(sample_size: int, label_pred: np.ndarray, nsamples: int) -> BinaryConfusionMatrix:
    sample_size = int(sample_size)
    nsamples = int(nsamples)
    # 0: TN, 1: FP, 2: FN, 3: TP
    codes = (label_pred[:, 0] * 2 + label_pred[:, 1]).astype(int)

    cms = np.zeros((nsamples, 2, 2), dtype=int)
    for i in range(nsamples):
        counts = np.bincount(
            np.random.choice(codes, sample_size, replace=False),
            minlength=4
        )

        cms[i] = [
            # [TP, FN],
            # [FP, TN]
            [counts[3], counts[2]],
            [counts[1], counts[0]]
        ]

    return BinaryConfusionMatrix(cms)


rnd_cm_times = []
score_times = []


def monte_carlo_scores(
        label_pred: pd.DataFrame,
        target_col: str,
        pred_col: str,
        sample_size: int,
        nsamples: int,
        metric_funcs: dict[str, callable],
) -> dict[str, BinaryConfusionMatrix]:
    all_scores = defaultdict(dict)

    rnd_cm_start = time.time()
    group_cms = get_rnd_cms(
        sample_size,
        label_pred[[target_col, pred_col]].to_numpy(),
        nsamples,
    )
    rnd_cm_lapse = time.time() - rnd_cm_start
    score_time_start = time.time()
    group_cms = BinaryConfusionMatrix(group_cms, normalize=True)

    for metric_name, metric_func in metric_funcs.items():
        scores = metric_func(group_cms)
        all_scores[metric_name][sample_size] = scores
    score_time_lapse = time.time() - score_time_start
    rnd_cm_times.append(rnd_cm_lapse)
    score_times.append(score_time_lapse)

    df = pd.DataFrame(all_scores)
    df.index.names = ['Sample Size']
    return df

def jaggedness(
        group_names: list[str] = 'all',
        max_sample_size: int = 25 if DEBUG else 100,
        save_dir=cmn.RESULTS_DIR
):
    label_preds = ft_data.get_sample_labels()

    for group_name, df in label_preds.groupby(ft.ACSIncome.group):
        if group_names != 'all' and group_name not in group_names:
            continue
        print(f"Group: {group_name}")
        label_preds = df[[ft.ACSIncome.target, 'pred']]
        monte_carlo_results = []
        sample_sizes = [i for i in range(5, max_sample_size)]
        for sample_size in sample_sizes:
            if df.shape[0] < sample_size:
                break
            print(f"Sample Size: {sample_size}")
            monte_carlo_results.append(
                monte_carlo_scores(
                    label_preds,
                    ft.ACSIncome.target,
                    'pred',
                    sample_size,
                    50 if DEBUG else int(1e3),
                    # bcm.top_name_funcs
                    {
                        'Accuracy': lambda cm: cm.accuracy(),
                        'Predicted Positive': lambda cm: cm.pp(),
                    }
                )
            )
        # continue
        if len(monte_carlo_results) < 2:
            print(f"Insufficient data for {group_name}")
            continue
        monte_carlo_df = pd.concat(monte_carlo_results).reset_index()

        # metric = 'Accuracy'
        metric = "Predicted Positive"

        monte_carlo_df = monte_carlo_df.explode(metric)

        plt.figure(figsize=(10, 6))
        # Lineplot for the median, error band for 10 and 90th percentiles
        sns.lineplot(data=monte_carlo_df, x='Sample Size', y=metric, marker=None,
                     estimator='median', errorbar=('pi', 80), label='10th-90th Percentile', alpha=1, linewidth=1)
        # Error band for 25th and 75th percentiles
        sns.lineplot(data=monte_carlo_df, x='Sample Size', y=metric, marker=None,
                     estimator='median', errorbar=('pi', 50), label='25th-75th Percentile', linewidth=1)
        sns.lineplot(data=monte_carlo_df, x='Sample Size', y=metric, marker=None,
                     estimator='median', errorbar=None, label='Median')
        # plt.gca().set_xscale('log')
        if metric == 'Accuracy':
            plt.title(f"Variability of Predictive Accuracy for Wealth Classification Among {group_name} Individuals:\n"
                      "A Monte Carlo Simulation Study Based on Sample Size")
        elif metric == 'Predicted Positive':
            plt.title(f"Variability of Predicted Positive for Wealth Classification Among {group_name} Individuals:\n"
                      "A Monte Carlo Simulation Study Based on Sample Size")

        if metric == "Accuracy":
            score = skm.accuracy_score(label_preds[ft.ACSIncome.target], label_preds['pred'])
        elif metric == "Predicted Positive":
            cm = efficient_confusion_matrix(label_preds[ft.ACSIncome.target], label_preds['pred'])
            score = BinaryConfusionMatrix(cm, normalize=True).pp().item()

        plt.axhline(score, color='gray', linestyle='--', label=f'{metric} Considering All', linewidth=1.5)
        plt.xlabel("Sample Size")
        plt.ylabel(metric)
        plt.legend(title='Legend')
        plt.tight_layout()
        plt.savefig(save_dir / f"{group_name}_{metric}_income_data_monte_carlo_sampling.png")
