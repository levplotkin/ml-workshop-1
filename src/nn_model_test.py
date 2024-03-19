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

    testing = ImageFolder('/home/lev/dev/tikal/mll/workshop-01/data/mnist/testing', transform=transform)

    test_loader = DataLoader(testing, batch_size=16, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.load('/home/lev/dev/tikal/mll/workshop-01/src/mnist_model.pt')

    model.eval()  # set model in evaluation mode

    input = torch.rand([16, 28 * 28], dtype=torch.float32).to(device)
    out = model(input)

    loss_model = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    test_loop = tqdm(test_loader, leave=False)  # progress bar
    with torch.no_grad():  # stop gradient calculation
        true_answer = 0  # count of true positive
        for x, targets in test_loop:
            # (batch_size,1,28,28) ->  (batch_size,28*28)
            x = x.reshape(-1, 28 * 28).to(device)
            # (batch_size,1) ->  (batch_size,10, dtype = int32)
            targets = targets.reshape(-1).to(torch.int32)
            targets = torch.eye(10)[targets].to(device)
            # forward propagation
            pred = model(x)

            # pred is tensor (batch_size, 10), index with the max value is the answer of model
            true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

            test_loop.set_description(f"true_answers = {true_answer}]")

        print(f"true_answers = {true_answer} from 10000")

