from core.nnlib.metrics import SoftmaxAccuracy
import logging

import numpy as np
from core.nnlib import *
from tqdm import tqdm, trange
from pprint import pprint
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
from sklearn.metrics import mean_absolute_error, accuracy_score

path = "data/UTKFace"


path = "data/UTKFace/"
n_samples = 300

dataset = UTKDatasetLoader(path, target_column=2, validation_split=0.002)
dataset.fit_dim_reducer(n_samples)

input_dim, output_dim = dataset.dimensions
loss = CrossEntropyLossWithSoftmax()
reg_loss = L2RegularizationLoss()
lr = 10e-7
lambda_reg = lr * 10e-2
decay = TimeStepDecay(decay=0.5)
weight_initializer = NormalWeightInitializer(std=0.02)
model = NeuralNet()
model.add(InputLayer(input_dim))
model.add(DenseLayer(dataset.dim_reducer_size, activation=LeakyRelu(), weight_initializer=weight_initializer))
model.add(DenseLayer(dataset.dim_reducer_size//4, activation=LeakyRelu(), weight_initializer=weight_initializer))
model.add(DenseLayer(output_dim, activation=SoftmaxWithCrossEntropy(), weight_initializer=weight_initializer))

model.compile()

opt = SGDOptimizer(loss=loss, lr=lr, decay_strategy=decay, lambda_reg=lambda_reg, regularizaton_loss=reg_loss)

opt.parameters(model.params)
trainer = Trainer(opt, epochs=1)

train_info, test_info = trainer.train(model, dataset, acc = SoftmaxAccuracy())

pprint(train_info)
pprint(test_info)
# X_test, y_test = dataset.test_set()
# print(X_test.shape, y_test.shape)
    # pass