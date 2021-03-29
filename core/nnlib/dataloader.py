import os
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from tqdm.auto import trange
from sklearn.base import TransformerMixin
import logging


class UTKDatasetLoader:
    class Reshaper(TransformerMixin):
        def transform(self, X):
            return X.reshape(X.shape[0], -1)

        def fit_transform(self, X, y=None, **fit_params):
            return self.transform(X)

    class Decorators:
        @classmethod
        def target_columns(cls, ufunc):
            def wrapper(self, *args, **kwargs):
                X, y = ufunc(self, *args, **kwargs)
                if self.target_columns is None:
                    return X, y
                else:
                    return X, y[:, self.target_columns]
            return wrapper

        @classmethod
        def floater(cls, ufunc):
            def wrapper(self, *args, **kwargs):
                X, y = ufunc(self, *args, **kwargs)
                return X.astype(np.float32), y
            return wrapper

    def __init__(
        self,
        path,
        batch_size=128,
        shuffle=True,
        validation_split=0.1,
        dim_reducer_size=128,
        target_columns=None,
    ):
        self.files = [
            os.path.join(root, f)
            for root, __, files in os.walk(path)
            for f in files if len(f.split("_")) == 4
        ]
        self.shuffle = True
        if self.shuffle:
            np.random.shuffle(self.files)

        self.validation_split = validation_split
        self.batch_size = batch_size
        self._target_columns = target_columns
        self.dim_reducer_size = dim_reducer_size
        self.dim_reducer = Pipeline(
            [
                ("reshaper", UTKDatasetLoader.Reshaper()),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=dim_reducer_size)),
            ]
        )

    @property
    def target_columns(self):
        return self._target_columns

    @target_columns.setter
    def target_columns(self, *args):
        self._target_columns = args

    def fit_dim_reducer(self, n_samples, shuffle=True):
        logging.info(f"fitting dimension reducer(shuffle={shuffle}).")
        if shuffle:
            sample_files = np.random.choice(self.files, n_samples)
        else:
            sample_files = self.files[:n_samples]
        X = []
        for fpath in sample_files:
            img = np.array(Image.open(fpath))
            X.append(img)
        X = np.array(X)
        self.dim_reducer.fit(X)
        logging.info(f"fitting dimension reducer complete.")

    def __iter__(self):
        self.n = 0
        return self

    def __len__(self):
        return len(self.files)

    def __next__(self):
        prev_n = self.n
        self.n += 1
        return self[prev_n]

    def _getindex(self, index):
        fpath = self.files[index]
        fname = Path(fpath).name
        age, gender, race, _ = fname.split("_")
        img = np.array(Image.open(fpath))
        return img, int(age), int(gender), int(race)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self._getindex(key)
        else:
            raise TypeError('Invalid argument type: {}'.format(type(key)))

    @property
    def split_index(self):
        return int(len(self) * (1 - self.validation_split))

    @Decorators.floater
    @Decorators.target_columns
    def batches(self, progress=True):
        logging.info("loading training batches.")
        train_indices = np.arange(self.split_index)
        if self.shuffle:
            np.random.shuffle(train_indices)
        prange = (
            trange(0, self.split_index, self.batch_size)
            if progress
            else range(0, self.split_index, self.batch_size)
        )

        for start_idx in prange:
            batch_indices = train_indices[
                start_idx: start_idx + self.batch_size
            ]
            X_batch, y_batch = [], []
            for index in batch_indices:
                img, age, gender, race = self[index]
                X_batch.append(img)
                y_batch.append((age, gender, race))
            X_batch = np.array(X_batch)
            X_batch = self.dim_reducer.transform(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch

    @Decorators.floater
    @Decorators.target_columns
    def validation_set(self, batch_size=30):
        X_val = np.zeros((len(self)-self.split_index, self.dim_reducer_size))
        y_val = np.zeros((len(self)-self.split_index, 3))
        print(X_val.shape, y_val.shape)
        logging.info(
            f"validation set loading started.(batch_size={batch_size})")
        #lazily reducing dimensions of validation set
        for start_idx in range(self.split_index, len(self), batch_size):
            curr_idx = start_idx - self.split_index
            total = len(self) - self.split_index
            X_cache, ages, genders, races = list(
                zip(*self[start_idx:start_idx+batch_size]))
            X_cache = np.array(X_cache)
            X_cache = self.dim_reducer.transform(X_cache)
            y_cache = np.array((ages, genders, races)).T
            y_val[curr_idx:curr_idx+batch_size, :] = y_cache
            X_val[curr_idx:curr_idx+batch_size, :] = X_cache
            logging.info(
                "validation set loading: "
                f"step #{1 + curr_idx//batch_size} "
                f"out of total {1 + total//batch_size} steps.")
        return X_val, y_val
