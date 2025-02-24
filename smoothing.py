from typing import Callable

import folktables as ft
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn import metrics as skm

import binary_confusion_matrix as bcm
import common as cmn
import folktables_data as ft_data
from common import DEBUG

IMG_DIR = cmn.RESULTS_DIR

_max_n = 15 if DEBUG else 50
_min_n = 5

# We recommend physical core count, not logical core count
import psutil

njobs = max(psutil.cpu_count(logical=False), 1)


def apply_dirichlet_smoothing(group_cm: np.ndarray, priori_cm: bcm.BinaryConfusionMatrix, lambda_: float = 1.0):
    """Apply Dirichlet smoothing to a batch of confusion matrices based on a reference confusion matrix (prior).

    Args:
        group_cm (np.ndarray): Array of shape (nsamples, 2, 2) containing multiple confusion matrices.
        priori_cm (bcm.BinaryConfusionMatrix): The reference confusion matrix to use as the prior.
        lambda_ (float): Scaling factor (λ) for the prior counts to control prior strength.

    Returns:
        np.ndarray: Smoothed confusion matrices with shape (nsamples, 2, 2).
    """
    # Ensure group_cm has the correct shape (nsamples, 2, 2)
    assert group_cm.ndim <= 3 and group_cm.shape[-2:] == (2, 2), "group_cm must be of shape (2,2) or (nsamples, 2, 2)"
    # Posterior α = Observed counts + λ * Prior counts
    posterior_counts = group_cm + lambda_ * priori_cm

    # Compute the mean of the Dirichlet posterior for each confusion matrix
    # The Dirichlet mean is α_i / sum(α)
    posterior_probs = posterior_counts / np.sum(posterior_counts, axis=(1, 2), keepdims=True)

    # Rescale probabilities back to counts matching the original sample sizes
    return bcm.BinaryConfusionMatrix(
        posterior_probs * np.sum(group_cm, axis=(1, 2), keepdims=True)
    )


def oversize_n_wrapper(func: Callable):
    quantiles = np.linspace(0, 1, 1000)

    def wrapper(*args, **kwargs):
        return np.nanquantile(func(*args, **kwargs), quantiles)

    return wrapper


def small_n_wrapper(func: Callable):
    def wrapper(*args, **kwargs):
        values = np.array([func(*args, **kwargs)])
        return values[np.isfinite(values)]

    return wrapper


def get_scores_over_n(
        samples_per_n: int,
        metric_func: Callable[[bcm.BinaryConfusionMatrix, float], float],
        metric_name: str,
        group_cm: bcm.BinaryConfusionMatrix,
        priori_cm: bcm.BinaryConfusionMatrix,
        priori_scales: list[float] | float = 1.0,
        max_n: int = _max_n,
        min_n: int = _min_n,
        epsilons: list[float] | float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert priori_cm.sum() == 1.0, "Priori confusion matrix must be normalized"
    group_probs = (group_cm / group_cm.total).flatten()
    if samples_per_n > 1000:  # save space but keep distribution of larger sample size
        metric_func = oversize_n_wrapper(metric_func)
    else:
        metric_func = small_n_wrapper(metric_func)

    def compute_for_n(n: int, epsilons: list[float], priori_scales: list[float]):
        group_cm_samples = np.random.multinomial(n, group_probs, size=samples_per_n)
        group_cm_samples = bcm.BinaryConfusionMatrix(group_cm_samples.reshape((samples_per_n, 2, 2)))

        # Metric for unsmoothed confusion matrices
        df_scores = []
        for epsilon in epsilons:
            df_scores.append(pd.DataFrame(metric_func(group_cm_samples, epsilon), columns=[metric_name]))
            df_scores[-1]['Epsilon'] = epsilon
        df_scores = pd.concat(df_scores)
        df_scores['Sample Size'] = n

        # Metric for smoothed confusion matrices
        df_scores_smooth = []
        for priori_scale in priori_scales:
            group_cm_samples_smooth = apply_dirichlet_smoothing(
                group_cm_samples, priori_cm, lambda_=priori_scale
            )
            values_smooth = metric_func(group_cm_samples_smooth)
            df_scores_smooth.append(pd.DataFrame(values_smooth, columns=[metric_name]))
            df_scores_smooth[-1]['Lambda'] = priori_scale

        df_scores_smooth = pd.concat(df_scores_smooth)
        df_scores_smooth['Sample Size'] = n
        return df_scores, df_scores_smooth

    if DEBUG:
        results = [compute_for_n(n, epsilons, priori_scales)
                   for n in range(min_n, max_n)
                   ]
    else:
        results = Parallel(n_jobs=njobs)(
            delayed(compute_for_n)(n, epsilons, priori_scales)
            for n in range(min_n, max_n)
        )
    scores, score_smooths = zip(*results)
    return pd.concat(scores), pd.concat(score_smooths)


def plot_scores_over_n(
        scores: pd.DataFrame,
        metric_name: str,
        whole_group_score: float,
):
    plt.figure(figsize=(5, 5))

    # Error band for 1st and 99th percentiles
    sns.lineplot(data=scores, x='Sample Size', y=metric_name, marker=None,
                 estimator='median', errorbar=('pi', 98), label='1st-99th Percentile', alpha=0.5, linewidth=1)

    # Error band for 10 and 90th percentiles
    sns.lineplot(data=scores, x='Sample Size', y=metric_name, marker=None,
                 estimator='median', errorbar=('pi', 80), label='10th-90th Percentile', alpha=1, linewidth=1)
    # Error band for 25th and 75th percentiles
    sns.lineplot(data=scores, x='Sample Size', y=metric_name, marker=None,
                 estimator='median', errorbar=('pi', 50), label='25th-75th Percentile', linewidth=1)
    sns.lineplot(data=scores, x='Sample Size', y=metric_name, marker=None,
                 estimator='median', errorbar=None, label='Median')

    plt.axhline(whole_group_score, color='gray', linestyle='--', label=f'Using All Samples', linewidth=1.5)
    plt.xlabel("Sample Size")
    plt.ylabel(metric_name)
    plt.legend(title='Legend')
    return


def score_offset_for_n_range(scores, score_smooths, metric_name, whole_group_score):
    percentiles = [10, 25, 50, 75, 90]

    def score_offsets(scores, metric_name, whole_group_score):
        return np.percentile(scores[metric_name] - whole_group_score, percentiles)

    score_percentiles = score_offsets(scores, metric_name, whole_group_score)
    score_smooth_percentiles = score_offsets(score_smooths, metric_name, whole_group_score)
    print(
        f"{metric_name} Offset Differences: {dict(zip(percentiles, np.abs(score_percentiles) - np.abs(score_smooth_percentiles)))}")
    print(
        f"Variance of Scores: {np.var(scores[metric_name]):.8f}; Variance of Smooths: {np.var(score_smooths[metric_name]):.8f}")


def over_n_ranges(scores, score_smooths, metric_name, whole_group_score):
    n_max = scores['Sample Size'].max()
    for n_range in [(5, 10), (11, 25), (26, 50), (51, 100)]:
        if n_range[0] > n_max:
            continue
        scores_subset = scores[scores['Sample Size'].between(*n_range)]
        score_smooths_subset = score_smooths[score_smooths['Sample Size'].between(*n_range)]
        print(f"Sample Size Range: {n_range}")
        score_offset_for_n_range(scores_subset, score_smooths_subset, metric_name, whole_group_score)