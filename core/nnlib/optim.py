import numpy as np
from abc import ABC, abstractmethod


class SGDOptimizer():
    def __init__(
            self, loss, lr=0.001, momentum=0.9, decay_strategy=None, lambda_reg=None,
            regularizaton_loss=None):
        self.lr = lr
        self.decay_strategy = decay_strategy
        self.momentum = momentum
        self.regularization_loss = regularizaton_loss
        self.lambda_reg = lambda_reg
        self.loss = loss

        if self.loss is None:
            raise ValueError("please provide a loss function.")

        if None in [self.regularization_loss, self.lambda_reg] and \
                self.regularization_loss != self.lambda_reg:
            raise ValueError(
                "regularization loss and regularization lambda should both be given values.")

    def update_lr(self):
        if self.decay_strategy is not None:
            self.lr = self.decay_strategy(self.lr)

    def parameters(self, params):
        self.params = params
        self.prev_grads = {k: {"weights": 0, "bias": 0}
                           for k in list(self.params.keys())[1:]}

    def compute_loss(self, y_true, y_pred, **kwargs):
        if self.regularization_loss is not None:
            weights = kwargs["weights"]
            return self.loss(
                y_true=y_true, y_pred=y_pred) + self.lambda_reg * self.regularization_loss(
                weights)
        else:
            return self.loss(y_true=y_true, y_pred=y_pred)

    def compute_loss_grad(self, y_true, y_pred):
        return self.loss.gradient(y_true=y_true, y_pred=y_pred)

    def step(self, batch_size, **kwargs):
        for name, p in list(self.params.items())[1:]:
            weights_grad =  self.lr * (p["grads"]["weights"])
            if self.lambda_reg is not None:
                weights_grad += self.lambda_reg * \
                    self.regularization_loss.gradient(p["weights"]) / batch_size

            bias_grad = p["grads"]["bias"] * self.lr

            weights_grad += self.momentum * self.prev_grads[name]["weights"]
            bias_grad += self.momentum * self.prev_grads[name]["bias"]

            self.params[name]["weights"] -= weights_grad
            self.params[name]["bias"] -= bias_grad

            self.prev_grads[name]["weights"] = weights_grad
            self.prev_grads[name]["bias"] = bias_grad
