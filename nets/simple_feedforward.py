"""Simple feedforward neural network."""

import torch.nn as nn


# Fully connected neural network with one hidden layer
class FeedFowardNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedFowardNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out
