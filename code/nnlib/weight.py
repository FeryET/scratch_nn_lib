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
  def __init__(self, std=0.1, *args, **kwargs):
    self.std = std
    super(WeightInitializer, self).__init__()

  def __call__(self, shape):
    return np.random.normal(loc=0, scale=self.std, size=shape)