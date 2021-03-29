
import numpy as np
from abc import ABC, abstractmethod

#Defining base loss class
class Loss(ABC):
  @abstractmethod
  def __call__(self, y_pred, y_loss):
    pass
  @abstractmethod
  def gradient(*args, **kwargs):
    pass

class MSELoss(Loss):
  def __call__(self, y_pred, y_true):
    return np.square(y_pred-y_true).mean() / 2
  def gradient(y_pred, y_true):
    return (y_pred - y_true).mean()

class L2RegularizationLoss(Loss):
  def __call__(self,  params):
    return np.square([p["weights"] for key in params.values()]) / 2
  def gradient(self, w):
    return w