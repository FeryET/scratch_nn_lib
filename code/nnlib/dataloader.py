
import os
from PIL import Image
import numpy as np
from pathlib import Path 
from sklearn.decomposition import PCA
from tqdm.notebook import trange 

class UTKDatasetLoader():
  def __init__(self, path, batch_size=128, shuffle=True, validation_split=0.1, dim_reducer_size=128):
    self.files = [os.path.join(root, f) for root, __, files in os.walk(path) for f in files]
    self.shuffle = True
    self.validation_split = validation_split
    self.batch_size = batch_size
    if self.shuffle:
      np.random.shuffle(self.files)
    self.target_columns = None
    self.dim_reducer = PCA(n_components=dim_reducer_size)

  def set_target_columns(self,*args):
    self.target_columns = args

  def train_dim_reducer(self, n_samples, shuffle=True):
    if shuffle:
      sample_files = np.random.choice(self.files, n_samples)
    else:
      sample_files = self.files[:n_samples]
    images = []
    for fpath in sample_files:
      img = np.array(Image.open(fpath))
      images.append(img)
    X = np.array(images).reshape(len(images),-1)
    self.dim_reducer.fit(X)

  def __iter__(self):
    self.n = 0
    return self
  
  def __len__(self):
    return len(self.files)

  def __next__(self):
    prev_n = self.n
    self.n += 1
    return self[prev_n]

  def __getitem__(self, index):
    fpath = self.files[index]
    fname = Path(fpath).name
    age, gender, race, _ = fname.split("_")
    img = np.array(Image.open(self.fpath))
    return img, int(age), int(gender), int(race)
  
  @property
  def split_index(self):
    return int(len(self) * (1 - self.validation_split))

  def batches(self, progress=True):
    train_indices = np.arange(self.split_index)
    if self.shuffle:
      np.random.shuffle(train_indices)
    prange = trange(0, self.split_index, self.batch_size) if progress else \
                                    range(0, self.split_index, self.batch_size)

    for start_idx in prange:
      batch_indices = train_indices[start_idx: start_idx + self.batch_size]
      X_batch = []
      y_batch = []
      for index in batch_indices:
        img, age, gender, race = self[index]
        X_batch.append(img)
        y_batch.append((age, gender, race))
      X_batch = np.array(X_batch).reshape(len(X_batch),-1)
      X_batch = self.dim_reducer.transform(X_batch)
      y_batch = np.array(y_batch)
      if self.target_columns is not None:
        y_batch = y_batch[:, self.target_columns]
      yield X_batch, y_batch
    
  def validation_set(self):
    X_val, y_val = [], []
    for idx in range(self.split_index, len(self)):
      img, age, gender, race = self[idx]
      X_val.append(img)
      y_val.append((age, gender, race))
    if self.target_columns is not None:
      y_val = y_val[:, self.target_columns]
    X_val = np.array(X_val).reshape(len(X_val),1)
    X_val = self.dim_reducer.transform(X_val)
    y_val = np.array(y_val)
    return X_val, y_val