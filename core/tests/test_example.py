from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
import logging

import numpy as np
from core.nnlib import *
from tqdm import tqdm, trange
from pprint import pprint
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


path = "data/UTKFace"


class TestDataSetLoader(DatasetLoader):
    def __init__(self,
                 batch_size=4,
                 shuffle=True,
                 validation_split=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.validation_split = validation_split
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        X, y = X.astype(np.float128), y.astype(np.float128)
        X = StandardScaler().fit_transform(X)
        y = y[..., np.newaxis]
        y = StandardScaler().fit_transform(y)
        self.X = X
        self.y = y

    def batches(self, *args, **kwargs):
        self.prange = trange(0, self.split_index, self.batch_size)
        for start_idx in self.prange:
            end_idx = start_idx + self.batch_size
            yield self.X[start_idx:end_idx, :], self.y[start_idx:end_idx, :]

    def _getindex(self, index):
        return self.X[index, :], self.y[index, :]

    def __iter__(self):
        self.n = 0
        return self

    def __len__(self):
        return len(self.X)

    def __next__(self):
        prev_n = self.n
        self.n += 1
        return self[prev_n]

    @property
    def split_index(self):
        return int(len(self.X) * (1 - self.validation_split))

    def validation_set(self, batch_size=None):
        return self.X[self.split_index:, :], self.y[self.split_index, :]


dataset = TestDataSetLoader()
model = NeuralNet()

model.add(InputLayer(dataset.X.shape[1]))
model.add(
    DenseLayer(
        dim=50, activation=LeakyRelu(),
        weight_initializer=NormalWeightInitializer(std=0.02)))
model.add(DenseLayer(dim=50, activation=LeakyRelu()))
model.add(DenseLayer(dim=1, activation=LeakyRelu()))
model.compile()
print(str(model))

loss = MSELoss()
reg_loss = L2RegularizationLoss()
opt = SGDOptimizer(loss=loss, lr=10e-3, decay_strategy=ConstantDecay(
    0.98), lambda_reg=10e-5, regularizaton_loss=reg_loss)

opt.parameters(model.params)

trainer = Trainer(opt)

print(dataset.y.shape)

trainer.train(model, dataset)
# X, y = dataset.validation_set(batch_size=100)
# print(X.shape, y.shape)

opt = SGDOptimizer()

