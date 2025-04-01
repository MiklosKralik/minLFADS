"""
Train lfads of pickled macaque reaching data from day 31
** reference:
"""

import os
import pickle
import torch
from torch.utils.data import Dataset

from minlfads.utils import set_seed, get_default_config
from minlfads.model import LFADS
from minlfads.trainer import Trainer


def main():
    cfg = get_default_config()
    cfg.model = LFADS.get_default_config()
    cfg.train = Trainer.get_default_config()

    # Load the data
    data_path = './data/macaque_reaching_day_31.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    position = data['velocity']
    conditions = data['conditions']

    print('X: ', X.shape, 'Y: ', position.shape, 'conditions: ', conditions.shape)
    cfg.model.input_size = X.shape[2] # set input size

    # split the data into training and test sets
    train_size = int(0.8 * len(X))
    X_train = X[:train_size]
    X_test = X[train_size:]
    position_train = position[:train_size]
    position_test = position[train_size:]
    conditions_train = conditions[:train_size]
    conditions_test = conditions[train_size:]

    # normalize the position data
    position_mean = position_train.mean(axis=0)
    position_std = position_train.std(axis=0)
    position_train = (position_train - position_mean) / position_std
    position_test = (position_test - position_mean) / position_std

    # create the dataset
    train_dataset = ReachingDataset(X_train)
    test_dataset = ReachingDataset(X_test)

    model = LFADS(cfg.model)
    trainer = Trainer(cfg.train, model, train_dataset, test_dataset)

    trainer.run()

class ReachingDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).float()

if __name__ == '__main__':
    set_seed(42)
    main()
