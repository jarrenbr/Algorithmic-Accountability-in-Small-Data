import numpy as np
from scipy.special import comb

import binary_confusion_matrix as bcm


def sample_confusion_matrix(
        n: int,
        samples: int | float = 1000,
        p: bcm.BinaryConfusionMatrix | list[int] = bcm.BinaryConfusionMatrix([[0.25, 0.25], [0.25, 0.25]]),
        dtype=np.uint64,
        ret_t: type = bcm.BinaryConfusionMatrix
) -> np.ndarray | bcm.BinaryConfusionMatrix:
    """
    Generate samples from the distribution of confusion matrices for n instances.
    Args:
    n (int): Total number of instances.
    samples (int): Number of samples to generate.
    dtype (np.dtype): Data type of the confusion matrices.

    Returns:
    np.array: Array of sampled confusion matrices.
    """
    if isinstance(samples, float):
        samples = int(samples)
    if isinstance(p, bcm.BinaryConfusionMatrix | np.ndarray):
        p = p.flatten().tolist()

    result = np.zeros((samples, 4), dtype=dtype)  # TP, FP, FN, TN
    for i in range(samples):
        result[i] = np.random.multinomial(n, p, size=1).flatten()

    if ret_t is bcm.BinaryConfusionMatrix:
        return bcm.BinaryConfusionMatrix(result.reshape((-1, 2, 2)))
    else:
        return result.reshape((-1, 2, 2))


def get_exhaustive_cms(
        n: int, dtype=np.float32,
        ret_t: type = bcm.BinaryConfusionMatrix,
) -> bcm.BinaryConfusionMatrix | np.ndarray:
    nrows = comb(n + 3, 3, exact=True)
    all_cms = np.empty((nrows, 4), dtype=dtype)
    i = 0
    for xi in range(n + 1):
        nminusxi = n - xi
        for yi in range(nminusxi + 1):
            nminusxi_yi = nminusxi - yi
            for zi in range(nminusxi_yi + 1):
                all_cms[i] = (xi, yi, zi, nminusxi_yi - zi)
                i += 1
    if ret_t is bcm.BinaryConfusionMatrix:
        return bcm.BinaryConfusionMatrix(all_cms.reshape((-1, 2, 2)))
    else:
        return all_cms.reshape((-1, 2, 2))
