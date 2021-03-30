import numpy as np
from abc import ABC, abstractmethod

from numpy.lib import expand_dims

# Defining base abstract activation function class


class Activation(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, X):
        pass

    @abstractmethod
    def derivative(self, X):
        pass


class LeakyRelu(Activation):
    def __init__(self, leakage_ratio=0.1, *args, **kwargs):
        self.leakage_ratio = 0.1
        super(Activation, self).__init__(*args, **kwargs)

    def __call__(self, X):
        neg_indices = X < 0
        transformed = X.copy()
        transformed[neg_indices] *= self.leakage_ratio
        return transformed

    def derivative(self, X):
        neg_indices = X < 0
        der = np.ones_like(X)
        der[neg_indices] = self.leakage_ratio
        return der


class Softmax(Activation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, X):
        exp_X = np.exp(X)
        return exp_X/exp_X.sum(axis=1, keepdims=True)

    def derivative(self, X):
        m = X.shape[-1]
        delta = np.zeros(X.shape+(m,), dtype=X.dtype)
        idx = np.arange(m)
        rev_idx = idx[::-1]
        delta[..., idx, rev_idx] = - X[..., idx] * X[..., rev_idx]
        delta[..., idx, idx] = X[..., idx] * (1 - X[..., idx])
        return delta
