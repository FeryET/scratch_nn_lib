import os
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from tqdm.auto import trange
from sklearn.base import TransformerMixin
import logging
from abc import ABC, abstractmethod
import time


class DatasetLoader(ABC):
    class Reshaper(TransformerMixin):
        def transform(self, X):
            return X.reshape(X.shape[0], -1)

        def fit_transform(self, X, y=None, **fit_params):
            return self.transform(X)

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def _getindex(self, index):
        pass

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self._getindex(key)
        else:
            raise TypeError('Invalid argument type: {}'.format(type(key)))

    @abstractmethod
    def batches(self, *args, **kwargs):
        pass

    @abstractmethod
    def validation_set(self, batch_size):
        pass

    @property
    @abstractmethod
    def test_split_index(self):
        pass

    @property
    @abstractmethod
    def val_split_index(self):
        pass

    @abstractmethod
    def test_set(self, batch_size):
        pass

    def update_progress(self, end=False, **kwargs):
        try:
            if end:
                self.prange.close()
                epoch = kwargs.pop("epoch")
                print(f"Epoch #{epoch+1} ended with " + "\t".join([f"{k}: {float(v):.3f}" for k, v in kwargs.items()]))
            else:
                if "epoch" in kwargs.keys():
                    self.prange.set_description(f"Epoch #{kwargs['epoch'] + 1}")
                    kwargs.pop("epoch")
                self.prange.set_postfix(kwargs)
                self.prange.refresh()
                time.sleep(0.01)    
            
        except Exception as e:
            # logging.debug(e)
            print(e)
            pass


class UTKDatasetLoader(DatasetLoader):
    class Decorators:
        @classmethod
        def target_column(cls, gen_func):
            def wrapper(self, *args, **kwargs):
                for X, y in gen_func(self, *args, **kwargs):
                    if self.target_column is None:
                        pass  # Do Nothing
                    else:
                        y = y[:, self.target_column][..., np.newaxis]
                        y = self.encoders[self.target_column].transform(y)
                    yield X, y
            return wrapper

        @classmethod
        def floater(cls, gen_func):
            def wrapper(self, *args, **kwargs):
                for X, y in gen_func(self, *args, **kwargs):
                    yield X.astype(np.float64), y
            return wrapper

    def __init__(
        self,
        path,
        batch_size=128,
        validation_split=0.1,
        test_split=0.1,
        dim_reducer_size=128,
        target_column=None,
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
        self.test_split = test_split
        self.batch_size = batch_size
        self._target_column = target_column
        self.dim_reducer_size = dim_reducer_size
        self.dim_reducer = Pipeline(
            [
                ("reshaper", DatasetLoader.Reshaper()),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=dim_reducer_size)),
            ]
        )
        self._initalize_label_encoders()

    def _initalize_label_encoders(self):
        ages, genders, races = list(
            zip(*[map(int, Path(f).stem.split("_")[:-1]) for f in self.files]))
        ages, genders, races = map(
            lambda x: np.array(x)[..., np.newaxis],
            (ages, genders, races))
        self.encoders = [MinMaxScaler(), OneHotEncoder(), OneHotEncoder()]
        self.encoders[0].fit_transform(ages)
        self.encoders[1].fit_transform(genders)
        self.encoders[2].fit_transform(races)

    @property
    def target_column(self):
        return self._target_column

    @target_column.setter
    def target_column(self, c):
        self._target_column = c

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

    @property
    def val_split_index(self):
        return int(self.test_split_index * (1 - self.validation_split))

    @property
    def test_split_index(self):
        return int(len(self) * (1 - self.test_split))

    @Decorators.floater
    @Decorators.target_column
    def batches(self, progress=True):
        logging.info("loading training batches.")
        train_indices = np.arange(self.val_split_index)
        if self.shuffle:
            np.random.shuffle(train_indices)
        self.prange = (
            trange(0, len(train_indices), self.batch_size, leave=False)
            if progress
            else range(0, len(train_indices), self.batch_size)
        )

        for start_idx in self.prange:
            batch_indices = list(train_indices[
                start_idx: start_idx + self.batch_size
            ])
            X_batch, y_batch = [], []
            for index in batch_indices:
                img, age, gender, race = self[int(index)]
                X_batch.append(img)
                y_batch.append((age, gender, race))
            X_batch = np.array(X_batch)
            X_batch = self.dim_reducer.transform(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch

    def validation_set(self, batch_size=30):
        val_size = self.test_split_index - self.val_split_index
        X_val = np.zeros((val_size, self.dim_reducer_size))
        y_val = np.zeros((val_size, 3))
        logging.info(
            f"validation set loading started.(batch_size={batch_size}"
            f", validation size={val_size})")
        # lazily reducing dimensions of validation set
        for start_idx in range(self.val_split_index, self.test_split_index, batch_size):
            curr_idx = start_idx - self.val_split_index
            X_cache, ages, genders, races = list(
                zip(*self[start_idx:min(start_idx+batch_size, self.test_split_index)]))
            X_cache = np.array(X_cache)
            X_cache = self.dim_reducer.transform(X_cache)
            y_cache = np.array((ages, genders, races)).T
            y_val[curr_idx:curr_idx+batch_size, :] = y_cache
            X_val[curr_idx:curr_idx+batch_size, :] = X_cache
            logging.info(
                "validation set loading: "
                f"step #{1 + curr_idx//batch_size} "
                f"out of total {1 + val_size//batch_size} steps.")
        return X_val.astype(np.float64), y_val[..., self.target_column][..., np.newaxis]

    def test_set(self, batch_size=30):
        test_size = len(self) - self.test_split_index
        X_test = np.zeros((test_size, self.dim_reducer_size))
        y_test = np.zeros((test_size, 3))
        logging.info(
            f"test set loading started.(batch_size={batch_size})")
        # lazily reducing dimensions of validation set
        for start_idx in range(self.test_split_index, len(self), batch_size):
            curr_idx = start_idx - self.test_split_index
            X_cache, ages, genders, races = list(
                zip(*self[start_idx:start_idx+batch_size]))
            X_cache = np.array(X_cache)
            X_cache = self.dim_reducer.transform(X_cache)
            y_cache = np.array((ages, genders, races)).T
            y_test[curr_idx:curr_idx+batch_size, :] = y_cache
            X_test[curr_idx:curr_idx+batch_size, :] = X_cache
            logging.info(
                "test set loading: "
                f"step #{1 + curr_idx//batch_size} "
                f"out of total {1 + test_size//batch_size} steps.")
        return X_test.astype(np.float64), y_test[..., self.target_column][..., np.newaxis]

    @property
    def dimensions(self):
        dim = [self.dim_reducer_size]
        if self.target_column == 0:
            dim.append(1)
        else:
            dim.np.append(len(self.encoders[self.target_column].get_feature_names()))
        return dim
