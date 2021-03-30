import numpy as np
from abc import ABC, abstractmethod

#Define base abstract weight initailizer class
class WeightInitializer(ABC):
  def __init__(self, *args, **kwargs):
    pass
  
  @abstractmethod
  def __call__(self):
    pass

class NormalWeightInitializer(WeightInitializer):
  def __init__(self, mean=0.0, std=0.02, *args, **kwargs):
    self.std = std
    self.mean = mean
    super().__init__()

  def __call__(self, shape):
    return np.random.normal(loc=self.mean, scale=self.std, size=shape)