import warnings
from typing import Iterable, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import RESULTS_DIR
from generate_cms import get_exhaustive_cms


def plot_percentile_subplot(ax, n: int, finite_scores: np.ndarray):
    ranks = np.linspace(0, 100, num=1000)
    percentiles = np.percentile(finite_scores, ranks, method='nearest')
    sns.lineplot(x=ranks, y=percentiles, ax=ax, label=f"Group Size: {n:,}", legend=False, linewidth=1)
    return ax


def plot_density_subplot(ax, n: int, finite_scores: np.ndarray):
    sns.kdeplot(finite_scores, label=f"Group Size: {n:,}", fill=False, ax=ax, linewidth=1)
    return ax


def get_finite_scores(func: Callable, cms: np.ndarray):
    np.seterr(divide='ignore', invalid='ignore')
    all_scores = func(cms)
    np.seterr(divide='warn', invalid='warn')
    return all_scores[np.isfinite(all_scores)]


def get_stats(n: int, name: str, all_scores) -> pd.DataFrame:
    return pd.DataFrame({
        'name': name,
        'mean': all_scores.mean(),
        'median': np.median(all_scores),
        'std': all_scores.std(),
        'min': all_scores.min(),
        'max': all_scores.max()
    }, index=[n])


def plot_density_percentile(
        n_values: Iterable[int],
        plot_density=True,
        get_cm_space_func: Callable = get_exhaustive_cms,
        cm_space_kwargs: dict = {},
        name_funcs=None,
        main_paper=False,
        subplots_shape: tuple[int, int] = (3, 3),
        figsize: tuple[float, float] = (8, 8),
        save_dir=RESULTS_DIR,
        plot_percentile=True
):
    if name_funcs is None:
        import binary_confusion_matrix as bcm
        name_funcs = {
            "Accuracy": bcm.BinaryConfusionMatrix.accuracy,
            "Predicted Positive": bcm.BinaryConfusionMatrix.pp,
            # "MCC with Limits": bcm.BinaryConfusionMatrix.mcc_with_limit,
            "MCC without Limits": bcm.BinaryConfusionMatrix.mcc,
            # "Equalized Odds Part": bcm.BinaryConfusionMatrix.equalized_odds_part,
            "Treatment Equality Part": bcm.BinaryConfusionMatrix.treatment_equality_part,
            # "Predicted Positive (DI Part)": bcm.BinaryConfusionMatrix.disparate_impact_part,
            # "Predictive Parity Part": bcm.BinaryConfusionMatrix.predictive_parity_part,
            "PPV": bcm.BinaryConfusionMatrix.ppv,
            # "Negative Predictive Value": bcm.BinaryConfusionMatrix.npv,
            "True Positive Rate": bcm.BinaryConfusionMatrix.tpr,
            # "False Positive Rate": bcm.BinaryConfusionMatrix.fpr,
            # "True Negative Rate": bcm.BinaryConfusionMatrix.tnr,
            # "False Negative Rate": bcm.BinaryConfusionMatrix.fnr,
            # "False Discovery Rate": bcm.BinaryConfusionMatrix.fdr,
            # "False Omission Rate": bcm.BinaryConfusionMatrix.for_
            "F1 Score": bcm.BinaryConfusionMatrix.f1_score,
            "Prevalence Threshold": bcm.BinaryConfusionMatrix.prevalence_threshold,
            "Marginal Benefit": bcm.BinaryConfusionMatrix.marginal_benefit,
        }

    def init_plots():
        fig, axes = plt.subplots(*subplots_shape, figsize=figsize)
        axes = axes.flatten()
        axes[-1].axis('off')
        return fig, axes

    if plot_percentile:
        fig_percentile, axes_percentile = init_plots()
    if plot_density:
        fig_density, axes_density = init_plots()

    max_n_funcs = 8
    if len(name_funcs) > max_n_funcs:
        warnings.warn(f"Too many functions to plot. Only plotting the first {max_n_funcs}.")
    for n in n_values:
        print(f"Beginning Group Size: {n}")
        cms = get_cm_space_func(n, **cm_space_kwargs)
        cms.normalize()
        for func_pos, name in enumerate(name_funcs.keys()):
            if func_pos >= max_n_funcs:
                continue

            all_scores = get_finite_scores(name_funcs[name], cms)

            if plot_percentile:
                percentile_ax = axes_percentile[func_pos]
                plot_percentile_subplot(
                    percentile_ax, n, all_scores
                )
                percentile_ax.set_title(name)
            if plot_density:
                density_ax = axes_density[func_pos]
                plot_density_subplot(
                    density_ax, n, all_scores
                )
                density_ax.set_title(name)

    def fig_legend(fig, axes):
        legend_fontsize = 'large'
        fig.tight_layout()
        fig.subplots_adjust(top=0.88, bottom=0.1, left=0.1, right=0.9)
        # Dummy subplot for figure to host the legend
        ax_dummy_kurt = fig.add_subplot(111, frame_on=False)
        ax_dummy_kurt.set_xticks([])
        ax_dummy_kurt.set_yticks([])
        handles, labels = axes[0].get_legend_handles_labels()
        ax_dummy_kurt.legend(handles, labels, loc='lower right', fontsize=legend_fontsize)
        return fig

    if plot_percentile:
        for ax in axes_percentile:
            ax.set_xlabel('Percentile')
            ax.set_ylabel('Value')
        fig_legend(fig_percentile, axes_percentile)
        plt.tight_layout()
        fig_percentile.savefig(save_dir / f"percentile_plots.png")

    if plot_density:
        for ax in axes_density:
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
        fig_legend(fig_density, axes_density)
        plt.tight_layout()
        fig_density.savefig(save_dir / f"density_plots.png")

    return
