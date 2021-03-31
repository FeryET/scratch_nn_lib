import numpy as np
from abc import ABC, abstractmethod
from sklearn.utils import validation

from tqdm.notebook import tqdm, trange
import logging


class Trainer():
    def __init__(self, opt, epochs=100,
                 batchsize=128, validation_split=0.1):
        self.opt = opt
        self.total_epochs = epochs
        self.batchsize = batchsize
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
                loss = self.opt.compute_loss(pred=y_pred, target=y_batch, weights=model.weights)
                output_grad = self.opt.compute_loss_grad(pred=y_pred, target=y_batch)
                model.backward(output_grad)
                self.opt.step(batch_size=len(X_batch))
                info["training_loss"].append(loss)
                dataloader.update_progress(loss=loss)

            info["training_loss"] = float(np.mean(info["training_loss"]))
            self.opt.update_lr()

            # Validation Set
            logging.info(f"validation set computations in epoch #{epoch + 1}.")
            y_pred = model.forward(X_val, train=False)
            val_loss = self.opt.compute_loss(pred=y_pred, target=y_val, weights=model.weights)
            info["validation_loss"] = float(val_loss)
            dataloader.update_progress(end=True, epoch=epoch, training_loss=info["training_loss"],validation_loss=info["validation_loss"])
            training_info.append(info)
        X_test, y_test = dataloader.test_set()
        y_pred = model.forward(X_test, train=False)
        test_loss = self.opt.compute_loss(pred=y_pred, target=y_test, weights=model.weights)
        return training_info, {"test_loss": test_loss}
