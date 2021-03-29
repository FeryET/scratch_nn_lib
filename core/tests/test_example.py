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
        self.X = np.random.uniform(low=0, high=1, size=(1000, 5))
        self.y = np.random.randint(low=0, high=2, size=(1000, 1))

    def batches(self, *args, **kwargs):
        for start_idx in trange(0, self.split_index, self.batch_size):
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
model.add(DenseLayer(dim=10, activation=LeakyRelu()))
model.add(DenseLayer(dim=1, activation=LeakyRelu()))
model.compile()
print(str(model))
# X, y = dataset.validation_set(batch_size=100)
# print(X.shape, y.shape)
