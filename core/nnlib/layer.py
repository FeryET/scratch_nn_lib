from pprint import pprint
import numpy as np
from abc import ABC, abstractmethod
from core.nnlib.weight import NormalWeightInitializer

# Defining base abstract layer class


class Layer(ABC):
    def __init__(self, *args, **kwargs):
        self._name = kwargs.get("name", None)
        self._weight_initializer = kwargs.get("weight_initializer", None)
        self._params = {"weights": None, "bias": None, "outputs": None, "grads": {}}
        pass

    @abstractmethod
    def _initialize_params(self):
        pass

    @abstractmethod
    def forward(self, X, train=True):
        pass

    @abstractmethod
    def backward(self, layer_input, delta_next_layer, w_next_layer):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def reset_grad(self):
        self._params["grads"] = {}
        self._params["outputs"] = None

    @property
    def params(self):
        return self._params

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n):
        self._name = n

    @abstractmethod
    def compile(self, info=None):
        self._initialize_params()

    @abstractmethod
    def to_json(self) -> dict:
        pass


class InputLayer(Layer):
    def __init__(self, input_dim, *args, **kwargs):
        '''
            Be sure that your input should be as shaped as (features, number of samples)
            and your output should be as (output neurons, number of samples)
        '''
        self.input_dim = input_dim
        super().__init__(*args, **kwargs)

    def forward(self, X, train=True):
        if X.shape[1] != self.input_dim:
            raise RuntimeError(
                "Difference between layer dimension and input array dimension.")
        out = X
        return out

    def backward(self, layer_input, delta_next_layer):
        raise NotImplementedError(
            "reset grad is not implemented for an InputLayer instance")

    def reset_grad(self):
        pass

    def compile(self, info=None):
        return {"input_dim": self.input_dim}

    def _initialize_params(self):
        pass

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "dim": f"(n_samples,{self.input_dim})"
        }


class DenseLayer(Layer):
    def __init__(self, dim, activation, *args, **kwargs):
        self.input_dim = None
        self.output_dim = dim
        self.activation = activation
        super().__init__(*args, **kwargs)

    def _initialize_params(self):
        if self._weight_initializer is None:
            self._weight_initializer = NormalWeightInitializer()
        weights = self._weight_initializer(shape=(self.input_dim, self.output_dim))
        bias = self._weight_initializer(shape=(1, self.output_dim))
        self.params["weights"] = weights
        self.params["bias"] = bias

    def forward(self, X, train=True):
        """
          It's important to notice that forward is expecting your matrice 
          to be in form (n_features, n_samples)
        """
        out = self.activation(X @ self.params["weights"]
                              + np.tile(self.params["bias"], (X.shape[0], 1)))
        return out

    def backward(self, layer_input, error):
        delta = error * self.activation.derivative(self.params["outputs"])
        self.params["grads"]["weights"] = layer_input.T @ delta
        self.params["grads"]["bias"] = delta.sum(axis=0).reshape(1, -1)
        return delta @ self.params["weights"].T

    def compile(self, info):
        self.input_dim = info["input_dim"]
        super().compile()
        return {"input_dim": self.output_dim}

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "dim": f"({self.input_dim},{self.output_dim})"
        }
