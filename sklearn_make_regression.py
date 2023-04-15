# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import numpy as np


class CustomDataset(Dataset):

    def __init__(self, x, y):
        assert (isinstance(x, np.ndarray))

        self.y = y.values
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        obs = self.x[idx, :]
        y = self.y[idx]

        return obs, y


class NeuralNetwork(nn.Module):

    def __init__(self, input_size):

        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self, x):
        y= self.linear_relu_stack(x)
        return y


def train_loop(dataloader, model, loss_fn, optimizer, device):

    train_loss = 0
    num_batches = len(dataloader)

    for batch, (X, y) in enumerate(dataloader):

        if device == 'cuda': # move batch to GPU if device is cuda
            X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # increment for this batch
        train_loss += loss

    train_loss /= num_batches
    print(f"Train loss: {train_loss:>7f} ]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:

            if device == 'cuda':  # move batch to GPU if device is cuda
                X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    print(f"Test loss: {test_loss:>8f} \n")


if __name__ == '__main__':

    # generate regression dataset
    n = 5000
    n_features = 5
    n_informative = 5
    n_targets = 1
    rand_state = 1

    data = make_regression(n_samples=n, n_features=n_features, n_informative=n_informative,
                           n_targets=n_targets, random_state=rand_state)
    X = pd.DataFrame(data[0])
    y = pd.DataFrame(data[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=rand_state)

    # Standard scale the inputs
    X_train_scaler = StandardScaler()
    X_train_scaler.fit(X_train)
    X_train_scaled = X_train_scaler.transform(X_train)
    X_test_scaled = X_train_scaler.transform(X_test) # use train scaler for test

    # Log-scale the regression output (y)?

    # Create Torch dataset and dataloaders
    # https://hsf-training.github.io/hsf-training-ml-gpu-webpage/03-usingthegpu/index.html
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using %s device"%device)

    train_data = CustomDataset(X_train_scaled, y_train)
    test_data = CustomDataset(X_test_scaled, y_test)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Create and forward-pass with nueral network
    model = NeuralNetwork(n_features).to(device).to(torch.float64)
    print(model)

    # feed-forward inference example
    train_features, train_labels = next(iter(train_dataloader))
    y_train_pred = model(train_features[0].to(device))

    # Train neural network
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")



