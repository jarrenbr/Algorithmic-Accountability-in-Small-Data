# Using COMPAS CMs from https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb

"""
Compas was originally formatted like this:
           	Low	High
Survived   	1679 380 # [TN, FP]
Recidivated	129	 77  # [FN, TP]

Changed to this:
 [TP, FN]
 [FP, TN]
"""
from pathlib import Path
from typing import Callable

import folktables as ft
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skm
from matplotlib.ticker import MaxNLocator

import binary_confusion_matrix as bcm
import common as cmn
import folktables_data as ft_data
import smoothing
from common import DEBUG

BCM = bcm.BinaryConfusionMatrix
np.random.seed(0)
IMG_DIR = cmn.RESULTS_DIR / "cps"
IMG_DIR.mkdir(exist_ok=True, parents=True)

from collections import namedtuple

m_info = namedtuple("metric_info", ["name", "abbrev", "func"])


def eps_str(eps: float) -> str:
    start = f"ε = "
    start += f"{int(eps)}" if int(eps) == eps else f"{eps:.0e}"
    return start


def lambda_str(lambda_: float) -> str:
    return f"λ = {lambda_}"

metric_infos = [
    # Binomial
    m_info("Accuracy", "ACC", BCM.accuracy),
    m_info("Predicted Positive Rate", "PPR", BCM.predicted_positive_rate),
    m_info("Prevalence", "PREV", BCM.prevalence),

    # JMR
    m_info("False Omission Rate", "FOR", BCM.for_),
    m_info("False Positive Rate", "FPR", BCM.fpr),
    m_info("False Negative Rate", "FNR", BCM.fnr),
    m_info("True Positive Rate", "TPR", BCM.tpr),
    m_info("True Negative Rate", "TNR", BCM.tnr),
    m_info("Positive Predictive Value", "PPV", BCM.ppv),
    m_info("Negative Predictive Value", "NPV", BCM.npv),
    m_info("False Discovery Rate", "FDR", BCM.fdr),

    # OFI
    m_info("Marginal Benefit", "B", BCM.marginal_benefit),

    # Other
    m_info("MCC", "MCC", BCM.mcc),
    m_info("F1 Score", "F1", BCM.f1_score),
    m_info("Prevalence Threshold", "PT", BCM.prevalence_threshold)
]

ft_income_group_cms = {
    'AI and AN': BCM([[13, 26], [31, 275]]),
    'Amer. Indian': BCM([[82, 113], [149, 810]]),
    'Asian': BCM([[4679, 2817], [3624, 13928]]),
    'Black': BCM([[755, 730], [1028, 5574]]),
    'Multiracial': BCM([[757, 609], [724, 3866]]),
    'Other': BCM([[746, 930], [1700, 14713]]),
    'Pacific Islander': BCM([[40, 31], [74, 341]]),
    'White': BCM([[19246, 10933], [12741, 54347]]),
}
ft_income_all_cm = BCM([[26318, 16189], [20071, 93854]])

compas_group_cms = {
    "Black": BCM([[273, 170], [1043, 1692]]),
    "White": BCM([[77, 129], [380, 1679]])
}
all_defendants_cm = BCM([[389, 347], [1597, 4121]])


def get_ft_income_priori_cm(group_key: str) -> BCM:
    return BCM(ft_income_all_cm - ft_income_group_cms[group_key])


def get_compas_priori_cm(group_key: str) -> BCM:
    return BCM(all_defendants_cm - compas_group_cms[group_key])


def get_mse_mae_scores(
        group_nm: str,
        group_cm: BCM,
        priori_cm: BCM,
        metric_func: Callable[[BCM, float], float],
        metric_name: str,
        max_n: int,
        min_n: int = 5,
        people_name: str = "People",
        epsilons: list[float] = [0],
        priori_scales=[5],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    priori_cm = BCM(priori_cm / priori_cm.sum())
    print(f"Lambda Scale: {priori_scales}")
    samples_per_n = 10_000 if DEBUG else 1_000_000
    scores, score_smooths = smoothing.get_scores_over_n(
        samples_per_n,
        metric_func,
        metric_name,
        group_cm,
        priori_cm,
        priori_scales=priori_scales,
        max_n=max_n,
        min_n=min_n,
        epsilons=epsilons
    )
    print(f"Epsilons: {epsilons}")
    group_score = metric_func(group_cm).item()
    print(f"{group_nm} Defendants {metric_name}: {group_score:.8f}")
    priori_score = metric_func(priori_cm).item()
    print(f"Priori {metric_name}: {priori_score:.8f}")
    print(f"Priori {metric_name} - {group_nm} {people_name} {metric_name}: {priori_score - group_score:.8f}")

    scores['MSE'] = (scores[metric_name] - group_score) ** 2
    score_smooths['MSE'] = (score_smooths[metric_name] - group_score) ** 2
    scores['MAE'] = np.abs(scores[metric_name] - group_score)
    score_smooths['MAE'] = np.abs(score_smooths[metric_name] - group_score)
    # assert scores['MSE'].isna().sum() == 0
    return scores, score_smooths, group_score


def plot_mse_maes(
        scores: pd.DataFrame,
        score_smooths: pd.DataFrame,
        group_nm: str,
        metric_name: str,
        metric_abbrev: str,
        save_dir: Path,
        max_n: int,
        min_n: int = 5,
        people_name: str = "People",
):
    for_paper = False
    title_font_size = 10.5

    save_dir.mkdir(exist_ok=True, parents=True)
    linewidth = 1
    figsize = (3, 2.75) if for_paper else (5.25, 4.5)

    alpha = 1

    # Plot MSE for each sample size
    errorbar = ('ci', 95)
    plt.figure(figsize=figsize)

    for epsilon, df in scores.groupby("Epsilon"):
        label = "Laplacian with " + eps_str(epsilon).replace(' ', '')
        sns.lineplot(
            data=df, x="Sample Size", y="MSE", label=label, alpha=alpha,
            errorbar=errorbar, linewidth=linewidth,
        )

    for lambda_, df in score_smooths.groupby("Lambda"):
        label = "CPS with " + lambda_str(lambda_).replace(' ', '')
        sns.lineplot(
            data=df, x="Sample Size", y="MSE", label=label, alpha=alpha,
            errorbar=errorbar, linewidth=linewidth
        )

    plt.ylabel("Mean-Squared Error")
    #despine
    plt.xlim(min_n, max_n)

    if for_paper:
        if metric_abbrev == "PT":
            plt.ylim(0, 0.105)
    else:
        plt.title(
            f'MSE of {metric_name} Across Sample Sizes:'
            f'\n Comparing {group_nm} {people_name} to Their Full Group Score',
            fontsize=title_font_size
        )
        plt.gca().xaxis.set_major_locator(MaxNLocator(prune=None))

    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{group_nm}_{metric_abbrev}_mse.png")
    return
    # # Plot MAE for each sample size
    # plt.figure(figsize=figsize)
    # errorbar = ('ci', 95)
    # sns.lineplot(
    #     data=scores, x="Sample Size", y="MAE", label=metric_name, alpha=alpha, errorbar=errorbar, linewidth=linewidth
    # )
    # sns.lineplot(
    #     data=score_smooths, x="Sample Size", y="MAE", label=f'{metric_abbrev} with CPS', alpha=alpha,
    #     errorbar=errorbar, linewidth=linewidth
    # )
    # plt.xlim(min_n, max_n)
    # if not for_paper:
    #     plt.title(
    #         f'MAE of {metric_name} Across Sample Sizes:'
    #         f'\n Comparing {group_nm} {people_name} to Their Full Group Score',
    #         fontsize=title_font_size
    #     )
    # plt.ylabel("Mean-Absolute Error")
    # sns.despine()
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(save_dir / f"{group_nm}_{metric_abbrev}_mae.png")


def plot_percentile_over_n(
        scores: pd.DataFrame,
        score_smooths: pd.DataFrame,
        group_score: float,
        group_nm: str,
        metric_name: str,
        metric_abbrev: str,
        save_dir: Path,
        max_n: int,
        min_n: int = 5,
        people_name: str = "People",
):
    save_dir.mkdir(exist_ok=True, parents=True)

    smoothing.plot_scores_over_n(scores, metric_name, group_score)
    plt.xlim(min_n, max_n)
    plt.title(f'{metric_name} for {group_nm} {people_name}')
    sns.despine()
    plt.tight_layout()
    y_min, y_max = plt.ylim()
    plt.savefig(save_dir / f"{group_nm}_{metric_abbrev}_og.png")

    smoothing.plot_scores_over_n(score_smooths, metric_name, group_score)
    plt.ylim(y_min, y_max)
    plt.xlim(min_n, max_n)
    plt.title(f'{metric_name} with CPS for {group_nm} {people_name}')
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_dir / f"{group_nm}_{metric_abbrev}_smooth.png")
    return


def get_ft_income_cms() -> dict[str, BCM]:
    label_preds = ft_data.get_sample_labels()
    cms = {}
    for race, df in label_preds.groupby(ft.ACSIncome.group):
        cm = bcm.BinaryConfusionMatrix(
            skm.confusion_matrix(df[ft.ACSIncome.target], df['pred'], labels=[0, 1]),
            from_sklearn=True,
            normalize=False
        )
        cms[race] = cm

    cms['All'] = bcm.BinaryConfusionMatrix(sum(cms.values()), normalize=False)

    return cms


def over_epsilons_priori_scales(
        epsilons: list[float],
        priori_scales: list[float],
        metric_infos=metric_infos,
        save_dir=IMG_DIR
):
    max_n = 30 if DEBUG else 150

    for info in metric_infos:
        # We decrease n here to better see the bump. For other metrics, this lower n
        # makes the scale too large.
        min_n = 5  #1 if info.abbrev in ["TPR", "PT"] else 5
        scores_all = []
        score_smooths_all = []
        columns_keep = ['Sample Size', 'MSE', 'MAE']
        score_cols_keep = columns_keep + ['Epsilon']
        smooth_cols_keep = columns_keep + ['Lambda']
        for group_nm, cm in compas_group_cms.items():
            scores, score_smooths, group_score = get_mse_mae_scores(
                group_nm, cm, get_compas_priori_cm(group_nm), info.func, info.name, max_n, min_n,
                "Defendants", epsilons=epsilons, priori_scales=priori_scales
            )
            scores_all.append(scores[score_cols_keep])
            score_smooths_all.append(score_smooths[smooth_cols_keep])

        for group_nm, cm in ft_income_group_cms.items():
            scores, score_smooths, group_score = get_mse_mae_scores(
                group_nm, cm, get_ft_income_priori_cm(group_nm), info.func, info.name, max_n, min_n,
                "Individuals", epsilons=epsilons, priori_scales=priori_scales
            )
            scores_all.append(scores[score_cols_keep])
            score_smooths_all.append(score_smooths[smooth_cols_keep])

        scores_all = pd.concat(scores_all)
        score_smooths_all = pd.concat(score_smooths_all)
        sub_dir = (
                save_dir / (
                    'eps ' + ' '.join(map(str, epsilons)) + '_' + 'lambda ' + ' '.join(map(str, priori_scales)))
        )
        sub_dir.mkdir(exist_ok=True, parents=True)
        plot_mse_maes(
            scores=scores_all, score_smooths=score_smooths_all, group_nm="All", metric_name=info.name,
            metric_abbrev=info.abbrev, save_dir=sub_dir, max_n=max_n, min_n=min_n,
            people_name="Individuals"
        )
        if DEBUG:
            break

    return