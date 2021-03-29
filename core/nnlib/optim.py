import numpy as np
from abc import ABC, abstractmethod


class SGDOptimizer():
    def __init__(self, lr=0.001, decay_strategy=None, lambda_reg=0.02):
        self.lr = lr
        self.decay_strategy = decay_strategy
        self.lambda_reg = lambda_reg

    def update_lr(self):
        if self.decay_strategy is not None:
            self.lr = self.decay_strategy(self.lr)

    def step(self, params, batch_size, **kwargs):
        self.reg_loss_function = kwargs.get("reg_loss_function", None)
        for p in list(params.values())[1:]:
            if self.reg_loss_function is not None:
                p["weights"] -= self.lr * (
                    p["grads"]["weights"] - self.lambda_reg *
                    self.reg_loss_function.gradient(p["weights"]) / batch_size)
            else:
                p["weights"] += - p["grads"]["weights"] * self.lr
            p["bias"] -= p["grads"]["bias"] * self.lr
        
