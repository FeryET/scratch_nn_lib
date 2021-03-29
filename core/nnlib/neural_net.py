import numpy as np
from abc import ABC, abstractmethod
from core.nnlib import InputLayer
from tabulate import tabulate
# Defining base neural network


class NeuralNet():
    def __init__(self):
        self.layers = []

    def add(self, l):
        if l.name == None:
            count = sum([1 for layer in self.layers if type(layer) == type(l)])
            l.name = f"{str(type(l).__name__)}_{count}"
        self.layers.append(l)

    def forward(self, X, train=True):
        out = X
        for l in self.layers:
            out = l.forward(out, train=train)
            if train:
                l.params["outputs"] = out
        return out

    def backward(self, out_delta):
        next_layer_delta = out_delta
        next_layer_w = None
        for idx in range(len(self.layers), 1, -1):  # We won't reach the input layer
            cur_layer = self.layers[idx]
            prev_layer = self.layers[idx - 1]
            next_layer_delta, next_layer_w = cur_layer.backward(
                prev_layer.params["outputs"],
                next_layer_delta, next_layer_w
            )

    def reset_grad(self):
        for l in self.layes:
            l.reset_grad()

    def predict(self, X):
        return self.forward(X, train=False)

    def __call__(self, X, train=True):
        return self.forward(X, train=True)

    def compile(self):
        if type(self.layers[0]) is not InputLayer:
            raise TypeError("First layer should be an input layer.")
        info = {}
        for l in self.layers:
            info = l.compile(info)
        

    @property
    def params(self):
        _params = {}
        for l in self.layers:
            _params[l.name] = l.params
        return _params


    def __str__(self) -> str:
        dicts_ = [l.to_json() for l in self.layers]
        rows = [d.values() for d in dicts_]
        header = dicts_[0].keys()
        return tabulate(rows, headers=header)