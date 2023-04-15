import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import numpy as np


if __name__ == '__main__':

    rand_state = 1

    # load breast data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=rand_state)

    # Standard scale the inputs
    X_train_scaler = StandardScaler()
    X_train_scaler.fit(X_train)
    X_train_scaled = X_train_scaler.transform(X_train)
    X_test_scaled = X_train_scaler.transform(X_test) # use train scaler for test
