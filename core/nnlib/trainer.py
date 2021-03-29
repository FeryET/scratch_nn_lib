import numpy as np
from abc import ABC, abstractmethod
from sklearn.utils import validation

from tqdm.notebook import tqdm, trange
import logging


class Trainer():
    def __init__(self, opt, loss_function, reg_loss_function=None, epochs=100,
                 batchsize=128, shuffle=True, validation_split=0.1):
        self.opt = opt
        self.loss_function = loss_function
        self.reg_loss_function = reg_loss_function
        self.total_epochs = epochs
        self.batchsize = batchsize
        self.shuffle = True
        self.validation_split = validation_split

    def train(self, model, dataloader):
        training_info = []
        X_val, y_val = dataloader.validation_set()
        logging.info("started training")
        for epoch in range(self.total_epochs):
            info = {"epoch": epoch + 1, "training_loss": [], "validation_loss": []}
            # Train Set
            for X_batch, y_batch in dataloader.batches():
                y_pred = model.forward(X_batch, train=True)
                loss = self.loss_function(y_pred, y_batch)
                if self.reg_loss_function is not None:
                    reg_loss = self.reg_loss_function(model.weights) / len(X_batch)
                    reported_loss = loss + self.opt.lambda_reg * reg_loss
                else:
                    reported_loss = loss
                output_grad = self.loss_function.gradient(y_pred=y_pred, y_true=y_batch)
                model.backward(output_grad)
                self.opt.step(model.params, reg_loss_function=self.reg_loss_function,
                              batch_size=len(X_batch))
                info["training_loss"].append(reported_loss)
                dataloader.update_progress(loss=reported_loss)

            self.opt.update_lr()

            # Validation Set
            logging.info(f"validation set computations in epoch #{epoch + 1}.")
            y_pred = model.forward(X_val, train=True)
            loss = self.loss_function(y_pred, y_val)
            if self.reg_loss_function is not None:
                reg_loss = self.opt.lambda_reg * \
                    self.reg_loss_function(model.params) / len(X_val)
                reported_loss = loss + reg_loss
            else:
                reported_loss = loss
            info["validation_loss"] = float(reported_loss)
            info["training_loss"] = float(np.mean(info["training_loss"]))
            dataloader.update_progress(training_loss=info["training_loss"],validation_loss=info["validation_loss"])
            training_info.append(info)

        return training_info
