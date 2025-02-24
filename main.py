import numpy as np
np.random.seed(0)

from common import DEBUG, RESULTS_DIR

import binary_confusion_matrix as bcm
from binary_confusion_matrix import BinaryConfusionMatrix as BCM

import analyzing_cps as acps
from analyzing_cps import m_info

import cdf
from sample_cms import jaggedness
import generate_cms as gcms

import warnings
# We omit all undefined values in plots
warnings.filterwarnings("ignore")

# Binomial
accuracy = m_info("Accuracy", "ACC", BCM.accuracy)
predicted_positive_rate = m_info("Predicted Positive Rate", "PPR", BCM.predicted_positive_rate)
prevalence = m_info("Prevalence", "PREV", BCM.prevalence)

# JMR
false_omission_rate = m_info("False Omission Rate", "FOR", BCM.for_)
false_positive_rate = m_info("False Positive Rate", "FPR", BCM.fpr)
false_negative_rate = m_info("False Negative Rate", "FNR", BCM.fnr)
true_positive_rate = m_info("True Positive Rate", "TPR", BCM.tpr)
true_negative_rate = m_info("True Negative Rate", "TNR", BCM.tnr)
positive_predictive_value = m_info("Positive Predictive Value", "PPV", BCM.ppv)
negative_predictive_value = m_info("Negative Predictive Value", "NPV", BCM.npv)
false_discovery_rate = m_info("False Discovery Rate", "FDR", BCM.fdr)

# OFI
marginal_benefit = m_info("Marginal Benefit", "B", BCM.marginal_benefit)

# Other
mcc = m_info("MCC", "MCC", BCM.mcc)
f1_score = m_info("F1 Score", "F1", BCM.f1_score)
prevalence_threshold = m_info("Prevalence Threshold", "PT", BCM.prevalence_threshold)


def main_paper_results():
    save_dir = RESULTS_DIR / "main_paper"
    save_dir.mkdir(exist_ok=True)
    # Figure 1
    jaggedness(["Multiracial"], 50, save_dir)

    # Figure 2
    cdf.plot_density_percentile(
        n_values=[5, 25, 100, 1000],
        plot_density=False,
        get_cm_space_func=lambda n, samples: gcms.sample_confusion_matrix(
            n, samples, p=bcm.BinaryConfusionMatrix([[.25, .25], [.25, .25]]),
        ),
        cm_space_kwargs={'samples': 100 if DEBUG else 1e6},
        name_funcs={
            "Accuracy": bcm.BinaryConfusionMatrix.accuracy,
            "Predicted Positive": bcm.BinaryConfusionMatrix.pp,
            "MCC with Limits": bcm.BinaryConfusionMatrix.mcc_with_limit,
            "MCC without Limits": bcm.BinaryConfusionMatrix.mcc,
            "True Positive Rate": bcm.BinaryConfusionMatrix.tpr,
        },
        subplots_shape=(3, 2),
        figsize=(5, 7),
        save_dir=save_dir
    )

    # Figure 3
    acps.over_epsilons_priori_scales(
        [1e-10], [5],
        metric_infos=[mcc, false_positive_rate, true_positive_rate, prevalence_threshold],
        save_dir=save_dir
    )
    acps.over_epsilons_priori_scales(
        [0, 1],
        [5, 10, 20],
        metric_infos=[prevalence],
        save_dir=save_dir
    )
    return


def supplementary_paper_results():
    save_dir = RESULTS_DIR / "supplementary_paper"
    save_dir.mkdir(exist_ok=True)

    # Section F
    acps.over_epsilons_priori_scales(
        [0, 1e-10],
        [5, 20],
        metric_infos=[true_positive_rate, mcc, accuracy, marginal_benefit],
        save_dir=save_dir
    )
    acps.over_epsilons_priori_scales(
        [0, 1],
        [5, 10, 20],
        metric_infos=[prevalence, false_negative_rate],
        save_dir=save_dir
    )
    return


if __name__ == "__main__":
    import warnings

    # We omit undefined values in plots
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main_paper_results()
    supplementary_paper_results()
    pass
