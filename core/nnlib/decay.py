import numpy as np
from abc import ABC, abstractmethod

# Defining base lr decay class
class LearningRateDecay(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ConstantDecay(LearningRateDecay):
    def __init__(self, decay=0.97):
        super().__init__(self)
        self.decay = decay

    def __call__(self, lr, *args, **kwargs):
        return lr * self.decay


class TimeStepDecay(LearningRateDecay):
    def __init__(self, decay=0.5, *args, **kwargs):
        self.decay = decay
        super().__init__(*args, **kwargs)

    def __call__(self, epoch, *args, **kwargs):
        return self.lr * 1 / (1 + epoch * self.decay)
