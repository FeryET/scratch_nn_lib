from core.nnlib import *
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)



path = "data/UTKFace"

dataset = UTKDatasetLoader(path, target_columns=1, dim_reducer_size=2)
dataset.fit_dim_reducer(n_samples=10)
X, y = dataset.validation_set(batch_size=100)
print(X.shape, y.shape)