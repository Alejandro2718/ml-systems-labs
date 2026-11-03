import numpy as np
import torch
from torch import nn

from value import Value


class ScalarModel:
    def __init__(self, weights):
        self.w1 = [[Value(x) for x in row] for row in weights['w1']]
        self.b1 = [Value(x) for x in weights['b1']]
        self.w2 = [Value(x) for x in weights['w2'][0]]
        self.b2 = Value(weights['b2'][0])

    def __call__(self, row):
        hidden = [(sum(w * x for w, x in zip(weights, row)) + bias).relu()
                  for weights, bias in zip(self.w1, self.b1)]
        return (sum(w * h for w, h in zip(self.w2, hidden)) + self.b2).sigmoid()

    def parameters(self):
        return [p for row in self.w1 for p in row] + self.b1 + self.w2 + [self.b2]

    def weights(self):
        return {'w1': np.array([[p.data for p in row] for row in self.w1]),
                'b1': np.array([p.data for p in self.b1]),
                'w2': np.array([[p.data for p in self.w2]]),
                'b2': np.array([self.b2.data])}


class TorchModel(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.hidden = nn.Linear(2, 8, dtype=torch.float64)
        self.output = nn.Linear(8, 1, dtype=torch.float64)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.from_numpy(weights['w1']))
            self.hidden.bias.copy_(torch.from_numpy(weights['b1']))
            self.output.weight.copy_(torch.from_numpy(weights['w2']))
            self.output.bias.copy_(torch.from_numpy(weights['b2']))

    def forward(self, x):
        return self.output(self.hidden(x).relu()).sigmoid()

    def weights(self):
        return {'w1': self.hidden.weight.detach().numpy().copy(),
                'b1': self.hidden.bias.detach().numpy().copy(),
                'w2': self.output.weight.detach().numpy().copy(),
                'b2': self.output.bias.detach().numpy().copy()}


def predict(weights, X):
    hidden = np.maximum(0, X @ weights['w1'].T + weights['b1'])
    return (1 / (1 + np.exp(-(hidden @ weights['w2'].T + weights['b2'])))).ravel()
