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

        # Setting learning rate for decay startegies if needed
        if self.decay_strategy != None: 
            self.decay_strategy.lr = self.lr

        if self.loss is None:
            raise ValueError("please provide a loss function.")

        if None in [self.regularization_loss, self.lambda_reg] and \
                self.regularization_loss != self.lambda_reg:
            raise ValueError(
                "regularization loss and regularization lambda should both be given values.")

    def update_lr(self, *args, **kwargs):
        if self.decay_strategy is not None:
            self.lr = self.decay_strategy(lr=self.lr, *args, **kwargs)
        else:
            raise RuntimeWarning("No decay strategy was found, applying no decay.")

    def parameters(self, params):
        self.params = params
        self.prev_grads = {k: {"weights": 0, "bias": 0}
                           for k in list(self.params.keys())[1:]}

    def compute_loss(self, target, pred, **kwargs):
        if self.regularization_loss is not None:
            weights = kwargs["weights"]
            return self.loss(
                target=target, pred=pred) + self.lambda_reg * self.regularization_loss(
                weights)
        else:
            return self.loss(target=target, pred=pred)

    def compute_loss_grad(self, target, pred):
        return self.loss.gradient(target=target, pred=pred)

    def step(self, batch_size, **kwargs):
        for name, p in list(self.params.items())[1:]:
            # computing weights gradient
            weights_grad = self.lr * (p["grads"]["weights"])
            if self.lambda_reg is not None:  # if regularization is needed
                weights_grad += self.lambda_reg * \
                    self.regularization_loss.gradient(p["weights"]) / batch_size

            # computing bias gradient
            bias_grad = p["grads"]["bias"] * self.lr

            # applying momentum
            weights_grad += self.momentum * self.prev_grads[name]["weights"]
            bias_grad += self.momentum * self.prev_grads[name]["bias"]

            # applying gradients
            self.params[name]["weights"] -= weights_grad
            self.params[name]["bias"] -= bias_grad

            # saving gradients for next iterations momentum
            self.prev_grads[name]["weights"] = weights_grad
            self.prev_grads[name]["bias"] = bias_grad
