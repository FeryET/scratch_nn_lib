import numpy as np
from abc import ABC, abstractmethod

#Defining base lr decay class
class LearningRateDecay():
    def __init__(self, *args, **kwargs):
      pass

class ConstantDecay(LearningRateDecay):
  def __init__(self, decay=0.97):
    LearningRateDecay.__init__(self)
    self.decay = decay
  def __call__(self, lr):
    return lr * self.decay