from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path('data/')
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

def load_observations(data_path=DATA_PATH, fname=TRAIN_FILE) -> (pd.DataFrame, pd.DataFrame):
    train_data = pd.read_csv(data_path / fname)
    train_labels = train_data.pop('label')
    return (train_data / 255).astype(np.float32), pd.get_dummies(train_labels, prefix="digit", dtype=np.float32)


if __name__=='__main__':
    images, labels = load_observations()
    pass