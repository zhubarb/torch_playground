import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
import numpy as np

from sklearn_make_regression import CustomDataset, train_loop, test_loop

class NeuralNetwork_Concatenated_inputs(nn.Module):

    def __init__(self, input_size, input_size1):

        super().__init__()
        # the first 'input_size1' features to propagate from stack 1, and the rest from stack 2
        self.len_features1 = input_size1
        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(input_size1, 32),
            nn.ReLU(),
            nn.Linear(32,16),
        )
        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(input_size-input_size1, 32),
            nn.ReLU(),
            nn.Linear(32,16),
        )
        self.output_layer = nn.Linear(32,1)

    def forward(self, x):
        x1 = self.linear_relu_stack1(x[:, :self.len_features1])
        x2 = self.linear_relu_stack2(x[:, self.len_features1:])
        # concatenate the relu stack 1 and 2
        x_interim = nn.ReLU()(torch.concat([x1,x2], dim=1))
        # feed concat. inputs to the output layer
        y = self.output_layer(x_interim)

        return y


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
    input_size1 = 2  # the first two features to propagate from stack 1, and the rest from stack 2
    model = NeuralNetwork_Concatenated_inputs(n_features, input_size1).to(device).to(torch.float64)  # move model to device
    print(model)

    # feed-forward inference example
    train_features, train_labels = next(iter(train_dataloader))
    expanded_input = train_features[0][None,:].to(device)  # expand from shape (5) to (5,1) to make compatib. w batch processing
    y_train_pred = model(expanded_input)

     # Train model or load existing weights
    model_weights_name = './torch_models/tutorial_w_concat_model_weights.pth'
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 10

    try:  # Loading existing weights
        # When loading existing weights, we need to instantiate the model class first,
        # because the class defines the structure of the network
        model.load_state_dict(torch.load(model_weights_name))
        print('Loaded existing %s'%model_weights_name)
        print('Checking oo-sample errors for %i epochs')
        for epoch in range(epochs):
            test_loop(test_dataloader, model, loss_fn, device)
    except FileNotFoundError as e:  # Train neural network
        print('Training for %s' % model_weights_name)

        # default `log_dir` is "runs" - we'll be more specific here
        # launch on terminal with cmd:  "tensorboard --logdir=runs/tutorial_concat/"
        # double-click on the grey NeuralNetwork mode to see the two parallel Relu stacks
        writer = SummaryWriter('runs/tutorial_concat')
        writer.add_graph(model, train_features.to(device))

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, device, epoch, writer)
            test_loop(test_dataloader, model, loss_fn, device)
        print("Done!")

        # Save model
        torch.save(model.state_dict(), model_weights_name)
        writer.close()
