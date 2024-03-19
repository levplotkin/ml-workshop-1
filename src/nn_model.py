import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder

import torchvision
from torchvision.transforms import v2

import os
import matplotlib.pyplot as plt

import numpy as np

import json

from tqdm import tqdm

from PIL import Image

if __name__ == "__main__":

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Grayscale(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    training = ImageFolder(root='/home/lev/dev/tikal/mll/workshop-01/data/mnist/training', transform=transform)
    testing = ImageFolder('/home/lev/dev/tikal/mll/workshop-01/data/mnist/testing', transform=transform)

    train_data, val_data = random_split(training, [0.8, 0.2])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(testing, batch_size=16, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    input = torch.rand([16, 28 * 28], dtype=torch.float32).to(device)
    out = model(input)

    loss_model = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    """
        learning
            training
                data
                forward propagation
                error calculation
                backward correction
                optimization step
            metrics
            save loss and metrics
            validation
                 data
                 forward propagation
                 error calculation
                 backward correction
            metrics
            save loss and metrics
    """
    EPOCH = 2
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # learning
    for epoch in range(EPOCH):
        # training
        model.train()  # set model in learning mode
        running_train_loss = []  # loss function results per batch
        true_answer = 0  # count of true positive
        train_loop = tqdm(train_loader, leave=False)  # progress bar
        for x, targets in train_loop:
            # (batch_size,1,28,28) ->  (batch_size,28*28)
            x = x.reshape(-1, 28 * 28).to(device)
            # (batch_size,1) ->  (batch_size,10, dtype = int32)
            targets = targets.reshape(-1).to(torch.int32)
            targets = torch.eye(10)[targets].to(device)
            # forward propagation
            pred = model(x)
            # calculate errors
            loss = loss_model(pred, targets)
            # backward
            opt.zero_grad()  # reset gradients
            # calculate new gradients
            loss.backward()
            # optimization step
            opt.step()

            running_train_loss.append(loss.item())
            mean_train_loss = sum(running_train_loss) / len(running_train_loss)

            # pred is tensor (batch_size, 10), index with the max value is the answer of model
            true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            train_loop.set_description(f"Epoch [{epoch + 1}/{EPOCH}, train_loss = {mean_train_loss}]")
        # metric: accuracy true positive/ total
        running_train_acc = true_answer / len(train_data)
        # save loss and metrics
        train_loss.append(mean_train_loss)
        train_acc.append(running_train_acc)
        # validation
        model.eval()  # set model in evaluation mode
        with torch.no_grad():  # stop gradient calculation
            running_val_loss = []  # loss function results per batch
            true_answer = 0  # count of true positive
            for x, targets in val_loader:
                # (batch_size,1,28,28) ->  (batch_size,28*28)
                x = x.reshape(-1, 28 * 28).to(device)
                # (batch_size,1) ->  (batch_size,10, dtype = int32)
                targets = targets.reshape(-1).to(torch.int32)
                targets = torch.eye(10)[targets].to(device)
                # forward propagation
                pred = model(x)
                # calculate errors
                loss = loss_model(pred, targets)

                running_val_loss.append(loss.item())
                mean_val_loss = sum(running_val_loss) / len(running_val_loss)

                # pred is tensor (batch_size, 10), index with the max value is the answer of model
                true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            # metric: accuracy true positive/ total
            running_val_acc = true_answer / len(val_data)
            # save loss and metrics
            val_loss.append(mean_val_loss)
            val_acc.append(running_val_acc)

        train_loop.set_description(
            f"Epoch [{epoch + 1}/{EPOCH}, train_loss = {mean_train_loss}, train_acc={running_train_acc}, val_loss = {mean_val_loss}, val_acc={running_val_acc}]")

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train_loss', 'val_loss'])
    plt.show()

    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['train_acc', 'val_acc'])
    plt.show()

    torch.save(model, "mnist_model.pt")

