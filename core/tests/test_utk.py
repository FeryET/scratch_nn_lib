import logging

import numpy as np
from core.nnlib import *
from tqdm import tqdm, trange
from pprint import pprint
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
from sklearn.metrics import mean_absolute_error

path = "data/UTKFace"


path = "data/UTKFace/"
n_samples = 300

dataset = UTKDatasetLoader(path, target_column=0, validation_split=0.002)
dataset.fit_dim_reducer(n_samples)

input_dim, output_dim = dataset.dimensions
model = NeuralNet()
model.add(InputLayer(input_dim))
model.add(DenseLayer(dataset.dim_reducer_size, activation=LeakyRelu(), weight_initializer=ZeroWeightInitializer()))
model.add(DenseLayer(dataset.dim_reducer_size//4, activation=LeakyRelu(), weight_initializer=ZeroWeightInitializer()))
model.add(DenseLayer(output_dim, activation=LeakyRelu(), weight_initializer=ZeroWeightInitializer()))

model.compile()

loss = MSELoss()
reg_loss = L2RegularizationLoss()
opt = SGDOptimizer(loss=loss, lr=10e-3, decay_strategy=ConstantDecay(
    0.98), lambda_reg=10e-5, regularizaton_loss=reg_loss)
opt.parameters(model.params)

trainer = Trainer(opt, epochs=1)

train_info, test_info = trainer.train(model, dataset, mae = mean_absolute_error)

pprint(train_info)
pprint(test_info)
# X_test, y_test = dataset.test_set()
# print(X_test.shape, y_test.shape)
    # pass