import torch
import torch.nn as nn
import numpy as np


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class PythonNet(nn.Module):
    def __init__(self, in_shape, n_out, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        in_ch = in_shape[0]
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        ).to(self.device)
        _dummy_in = torch.zeros(in_shape).to(self.device)
        _dummy_out = self.body(_dummy_in.unsqueeze(0))
        out_shape = _dummy_out.shape
        out_node = np.prod(np.array(out_shape[1:]))
        self.fc = nn.Sequential(
            nn.Linear(out_node, 512),
            nn.ReLU(),
            nn.Linear(512, n_out),
        ).to(self.device)

        self.body.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.shape[0], -1)
        return self.fc(out)
