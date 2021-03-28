import numpy as np
from abc import ABC, abstractmethod

from tqdm.notebook import tqdm, trange

class Trainer():
  def __init__(self, opt, loss_function, reg_loss_function = None, epochs=100,
               batchsize=128, shuffle=True, validation_split=0.1):
    self.opt = opt
    self.loss_function = loss
    self.reg_loss_function = reg_loss_function
    self.total_epochs = epochs
    self.batchsize = batchsize
    self.lr = lr
    self.lr_decay = lr_decay
    self.shuffle = True
    self.validation_split = validation_split
    
  
  def train(self, model, dataloader):
    training_info = []
    X_val, y_val = dataloader.validation_set()
    for epoch in range(self.total_epochs):
      info = {"training_loss":[], "validation_loss": []}
      #Train Set
      for X_batch, y_batch in dataloader.batches():
        y_pred = model.forward(X_batch, train=True)
        loss = self.loss_function(y_pred, y_batch)
        reg_loss = self.reg_loss_function(model.params) * (len(X_batch) / len(dataloader))
        reported_loss = loss + reg_loss
        output_grad = self.loss_function.gradient(y_pred=y_pred, y_true=y_batch)
        self.model.backward(output_grad)
        self.opt.step(model.params, reg_loss_function=reg_loss)
        info["training_loss"].append(reported_loss)
      
      #Validation Set
      y_pred = model.forward(X_val, train=True)
      loss = self.loss_function(y_pred, y_val)
      reg_loss = self.reg_loss_function(model.params) * (len(X_val) / len(dataloader))
      reported_loss = loss + reg_loss
      info["validation_loss"] = reported_loss
      training_info.append(info)
    return training_info