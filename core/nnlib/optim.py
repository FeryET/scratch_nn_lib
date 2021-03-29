import numpy as np
from abc import ABC, abstractmethod

class SGDOptimizer():
  def __init__(self, lr=0.001, decay_strategy=None, lambda_reg=None):
      self.lr = lr
      self.decay_strategy = decay_strategy
      self.lambda_reg = lambda_reg
  
  def _update_lr(self):
    if self.decay_strategy is not None:
      self.lr = self.decay_strategy(self.lr)
  
  def step(self, params, batch_size, **kwargs):
    self.reg_loss_function = kwargs.get("reg_loss_function", None)
    for p in params.values():
      if self.reg_loss_function is not None:
          p["weights"] += - p["grads"]["weights"] * self.lr + reg_loss_function.gradient(p["weights"]) / batch_size
      else:
          p["weights"] += - p["grads"]["weights"] * self.lr
      p["bias"] += -p["grads"]["bias"] * self.lr
    self._update_lr()