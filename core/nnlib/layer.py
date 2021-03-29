import numpy as np
from abc import ABC, abstractmethod
from core.nnlib.weight import NormalWeightInitializer

# Defining base abstract layer class


class Layer(ABC):
    def __init__(self, *args, **kwargs):
        self._name = kwargs.get("name", None)
        self._weight_initializer = kwargs.get("weight_initializer", None)
        self._params = {"weights": None, "bias": None, "outputs": None, "grads": None}
        self._initialize_params()
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
        self._params["grads"] = None
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


class InputLayer(Layer):
    def __init__(self, input_dim, *args, **kwargs):
        self.input_dim = input_dim
        super().__init__(*args, **kwargs)

    def forward(self, X, train=True):
        out = X
        return out

    def backward(self, layer_input, delta_next_layer):
        raise NotImplementedError(
            "reset grad is not implemented for an InputLayer instance")

    def reset_grad(self):
        pass


class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, activation, *args, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        super().__init__(*args, **kwargs)

    def _initialize_params(self):
        if self._weight_initializer is None:
            self._weight_initializer = NormalWeightInitializer()
        weights = self._weight_initializer(shape=(self.input_dim, self.output_dim))
        bias = self._weight_initializer(shape=(self.output_dim, 1))
        self.params["weights"] = weights
        self.params["bias"] = bias

    def forward(self, X, train=True):
        """
          It's important to notice that forward is expecting your matrice 
          to be in form (n_features, n_samples)
        """
        out = self.activation(self.params["weights"].T @
                              X + np.tile(self.params["bias"], (1, X.shape[1])))
        return out

    def backward(self, layer_input, delta_next_layer, w_next_layer):
        if w_next_layer is None:
            delta = delta_next_layer * self.activation.derivative(self.params["outputs"])
        else:
            delta = ((w_next_layer).T @
                     delta_next_layer) * self.activation.derivative(self.params["outputs"])
        self.params["grads"]["weights"] = delta @ layer_input
        self.params["grads"]["bias"] = delta
        return delta, self.params["weights"]
