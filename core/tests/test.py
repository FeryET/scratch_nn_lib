import sys
sys.path.append("..")

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
            np.array([i if i >= 0 else leakage for i in X],
                     dtype=np.float32),
            relu.derivative(X))
        X = np.zeros((10, 10), np.float32)
        ntest.assert_array_equal(np.zeros((10, 10)), relu.derivative(X))


class TestWeights:
    def test_abstract_weight_initializer(self):
        ntest.assert_raises(TypeError, WeightInitializer())

    def test_normal_weight_initializer(self):
        shape = (100, 1000)
        std = 0.1
        weights = NormalWeightInitializer(std)(shape)
        ntest.assert_almost_equal(np.std(weights), std, decimal=3)
        ntest.assert_equal(weights.shape, shape)


class TestLayers:
    def test_abstract_layer(self):
        ntest.assert_raises(TypeError, Layer())

    def test_dense_layer(self):
        shape = (10, 100)
        l = DenseLayer(*shape, LeakyRelu())
        X = np.random.normal(0, 2, (100, shape[0]))
        ntest.assert_equal(shape[1], l(X).shape[1])


if __name__ == "__main__":
    # unittest.TestResult
    pass