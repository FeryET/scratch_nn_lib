
import numpy as np
from abc import ABC, abstractmethod

#Defining base loss class
class Loss(ABC):
  @abstractmethod
  def __call__(self, y_pred, y_loss):
    pass
  @abstractmethod
  def gradient(self, *args, **kwargs):
    pass

class MSELoss(Loss):
  def __call__(self, y_pred, y_true):
    return np.square(y_pred-y_true).mean(axis=0) / 2
  def gradient(self, y_pred, y_true):
    return (y_pred - y_true).mean(axis=0)

class L2RegularizationLoss(Loss):
  def __call__(self,  params):
    return np.square([params["weights"] for key in params.values()]) / 2
  def gradient(self, w):
    return w