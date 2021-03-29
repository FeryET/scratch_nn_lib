from core.nnlib import *
import unittest
import numpy as np
import pytest
import numpy.testing as ntest
from numpy._pytesttester import PytestTester


class TestActivations:
    def test_leakyrelu_forward(self):
        X = np.array([1, 0, -1], dtype=np.float32)
        leakage = 0.1

        comp = np.array(
            [i if i > 0 else i * leakage for i in X], dtype=np.float32)
        relu = LeakyRelu(leakage_ratio=leakage)
        ntest.assert_array_equal(comp, relu(X))
        X = np.zeros((10, 10), dtype=np.float32)
        ntest.assert_array_equal(np.zeros((10, 10), dtype=np.float32), relu(X))

    def test_leakyrelu_der(self):
        X = np.array([1, 0, -1], dtype=np.float32)
        leakage = 0.1
        relu = LeakyRelu(leakage_ratio=leakage)
        ntest.assert_array_equal(
            np.array([1 if i >= 0 else leakage for i in X],
                     dtype=np.float32),
            relu.derivative(X))
        X = np.ones((10, 10), np.float32)
        ntest.assert_array_equal(X, relu.derivative(X))


class TestWeights:
    def test_abstract_weight_initializer(self):
        ntest.assert_raises(TypeError, WeightInitializer)

    def test_normal_weight_initializer(self):
        shape = (100, 1000)
        std = 0.1
        weights = NormalWeightInitializer(std)(shape)
        ntest.assert_almost_equal(np.std(weights), std, decimal=3)
        ntest.assert_equal(weights.shape, shape)


class TestDecay:
    def test_abstract_decay(self):
        ntest.assert_raises(TypeError, LearningRateDecay)
    
    def test_constant_decay(self):
        decay = ConstantDecay()
        lr = 0.01
        ntest.assert_almost_equal(lr * decay.decay, decay(lr))

class TestLoss:
    def test_abstract_loss(self):
        ntest.assert_raises(TypeError, Loss)
    
    def test_mse_loss(self):
        y1 = np.array([[0, 1, 2, 3, 4]] * 4)
        y2 = np.array([[-1, 0.99, 2.05, 3.05, 10]] * 4)
        mse = np.square(y1 - y2).mean(axis=0)
        

class TestLayers:
    def test_abstract_layer(self):
        ntest.assert_raises(TypeError, Layer)

    def test_dense_layer(self):
        shape = (10, 100)  # 10 features, 1000 elements
        l = DenseLayer(input_dim=shape[0], output_dim=shape[1], activation=LeakyRelu())
        X = np.random.normal(0, 2, (shape[0], 1000))
        # transposing because the feature vector is deemed to be feature first
        ntest.assert_equal(shape[1], l(X).T.shape[1])

    

if __name__ == "__main__":
    pass
