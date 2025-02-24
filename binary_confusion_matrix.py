import numpy as np

"""
LEGEND
pp = predicted positive
p = positive label
pn = predicted negative
n = negative label
tp, fn, fp, tn = confusion matrix cells
cm = confusion matrix (np.ndarray)
"""
"""CM Format
[[tp, fn],
 [fp, tn]]
"""


class BinaryConfusionMatrix(np.ndarray):
    def __new__(
            cls,
            input_array: np.ndarray | list,
            normalize=False,
            dtype=np.float32,
            from_sklearn=False,
    ):
        if from_sklearn:
            input_array = np.flip(input_array, axis=[-2, -1])
        obj = np.asarray(input_array, dtype=dtype).view(cls)
        if obj.shape[-2:] != (2, 2):
            raise ValueError("Array shape must be (Any, 2,2)")
        obj.counts = obj.copy()
        obj._total = np.sum(obj, axis=(-1, -2))
        if normalize:
            obj.normalize()
        return obj

    def __array_finalize__(self, obj):
        if obj is None or hasattr(self, '_is_normalized'):
            return
        self._is_normalized = False

    @property
    def total(self):
        return self._total

    @property
    def tp(self):
        return self._tp()

    def _tp(self, eps: float = 0):
        return self[..., 0, 0] + eps

    @property
    def fn(self):
        return self._fn()

    def _fn(self, eps: float = 0):
        return self[..., 0, 1] + eps

    @property
    def fp(self):
        return self._fp()

    def _fp(self, eps: float = 0):
        return self[..., 1, 0] + eps

    @property
    def tn(self):
        return self._tn()

    def _tn(self, eps: float = 0):
        return self[..., 1, 1] + eps

    @property
    def is_normalized(self) -> bool:
        return self._is_normalized

    def to_dict(self):
        return {
            "TP": self.tp,
            "FN": self.fn,
            "FP": self.fp,
            "TN": self.tn
        }

    def to_numpy(self):
        return np.array([[self.tp, self.fn], [self.fp, self.tn]])

    def normalize(self):
        self._is_normalized = True
        self /= np.sum(self, axis=(-1, -2), keepdims=True)

    def pp(self, eps: float = 0):
        """Predicted positive"""
        return self.tp + self.fp + 2 * eps

    def pn(self, eps: float = 0):
        """Predicted negative"""
        return self.fn + self.tn + 2 * eps

    def p(self, eps: float = 0):
        """Labelled positive"""
        return self.tp + self.fn + 2 * eps

    def n(self, eps: float = 0):
        """Labelled negative"""
        return self.fp + self.tn + 2 * eps

    def __repr__(self):
        return f"{self.tp}, {self.fp}\n{self.fn}, {self.tn}"

    def __str__(self):
        if (isinstance(self.tp, float) or isinstance(self.fp, float) or
                isinstance(self.fn, float) or isinstance(self.tn, float)):
            return f"{self.tp:.8f}, {self.fp:.8f}\n{self.fn:.8f}, {self.tn:.8f}"
        else:
            return self.__repr__()

    ###self metrics
    # bounded
    def accuracy(self, eps: float = 0):
        return (self.tp + self.tn + 2 * eps) / (self.p(eps) + self.n(eps))

    def prevalence(self, eps: float = 0):
        return self.p(eps) / (self.p(eps) + self.n(eps))

    def predicted_positive_rate(self, eps: float = 0):
        return self.pp(eps) / (self.p(eps) + self.n(eps))

    def mcc(self, eps: float = 0):
        """phi coefficient (φ or r_φ) or Matthews correlation coefficient (MCC)"""
        numerator = self._tp(eps) * self._tn(eps) - self._fp(eps) * self._fn(eps)
        denominator = np.sqrt(self.pp(eps) * self.p(eps) * self.n(eps) * self.pn(eps))
        return numerator / denominator

    def mcc_with_limit(self, eps: float = 0):
        """phi coefficient (φ or r_φ) or Matthews correlation coefficient (MCC) with limits"""
        # eps isn't used here, only for compatibility purposes
        scores = self.mcc()
        scores[~np.isfinite(scores)] = 0
        total = self.total
        indeterminates = (self.tp == total) | (self.fp == total) | (self.fn == total) | (self.tn == total)
        scores[indeterminates] = np.nan
        return scores

    # unbounded
    def ppv(self, eps: float = 0):
        """precision or positive predictive value (PPV)"""
        return self._tp(eps) / self.pp(eps)

    def npv(self, eps: float = 0):
        """negative predictive value (NPV)"""
        return self._tn(eps) / self.pn(eps)

    def tpr(self, eps: float = 0):
        """sensitivity, recall, hit rate, or true positive rate (TPR)"""
        return self._tp(eps) / self.p(eps)

    def fpr(self, eps: float = 0):
        """fall-out or false positive rate (FPR)"""
        return self._fp(eps) / self.n(eps)

    def tnr(self, eps: float = 0):
        """specificity, selectivity or true negative rate (TNR)"""
        return self._tn(eps) / self.n(eps)

    def fnr(self, eps: float = 0):
        """miss rate or false negative rate (FNR)"""
        return self._fn(eps) / self.p(eps)

    def fdr(self, eps: float = 0):
        """false discovery rate (FDR)"""
        return self._fp(eps) / self.pp(eps)

    def for_(self, eps: float = 0):
        """false omission rate (FOR)"""
        return self._fn(eps) / self.pn(eps)

    def equalized_odds_part(self, eps: float = 0):
        return self.tpr(eps) + self.fpr(eps)

    def treatment_equality_part(self, eps: float = 0):
        return self._fn(eps) / self._fp(eps)

    def disparate_impact_part(self, eps: float = 0):
        return self.pp(eps)

    def predictive_parity_part(self, eps: float = 0):
        return self._tp(eps) / self.p(eps)

    def prevalence_threshold(self, eps: float = 0):
        tpr = self.tpr(eps)
        fpr = self.fpr(eps)
        return (np.sqrt(tpr * fpr) - fpr) / (tpr - fpr)

    def f1_score(self, eps: float = 0):
        return self._tp(eps) / (self.tp + (self.fp + self.fn) / 2 + 2 * eps)

    def marginal_benefit(self, eps: float = 0):
        # epsilon cancels out in numerator
        return (self.fp - self.fn) / (self.p(eps) + self.n(eps))
