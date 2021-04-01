
import numpy as np
from abc import ABC, abstractmethod

# Defining base loss class


class Loss(ABC):
    @abstractmethod
    def __call__(self, pred, target):
        pass

    @abstractmethod
    def gradient(self, *args, **kwargs):
        pass


class MSELoss(Loss):
    def __call__(self, pred, target):
        return np.square(pred-target).mean(axis=0) / 2

    def gradient(self, pred, target):
        return (pred - target).mean(axis=0)


class L2RegularizationLoss(Loss):
    def __call__(self,  weights):
        return sum([np.square(w).sum() for w in weights]) / 2

    def gradient(self, w):
        return w


class CrossEntropyLoss(Loss):
    def __call__(self, pred, target):
        return -np.sum(target * np.log(pred)).mean()

    def gradient(self, pred, target):
        return target/pred + (1-target)/(1-pred)


class CrossEntropyLossWithSoftmax(CrossEntropyLoss):
    def gradient(self, pred, target):
        return target - pred