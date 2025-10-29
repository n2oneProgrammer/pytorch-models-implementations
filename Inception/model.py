from typing import List

import numpy as np
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, output, in_channel=1, width=24, height=24, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=11, stride=5, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=0, stride=1)
        )
        self.sequential2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=0, stride=1)
        )
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=self._calc_output_size(width) * self._calc_output_size(height) * 128,
                                 out_features=1024)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=output)

    def _calc_output_size(self, size):
        size = np.floor((size + 2 * 0 - (11 - 1) - 1) / 5 + 1)
        size = np.floor((size + 2 * 0 - (3 - 1) - 1) / 1 + 1)
        size = np.floor((size + 2 * 2 - (5 - 1) - 1) / 1 + 1)
        size = np.floor((size + 2 * 0 - (3 - 1) - 1) / 1 + 1)
        size = np.floor((size + 2 * 1 - (2 - 1) - 1) / 1 + 1)
        # size = np.floor((size + 2 * 1 - (3 - 1) - 1) / 1 + 1)
        size = np.floor((size + 2 * 1 - (2 - 1) - 1) / 1 + 1)
        size = np.floor((size + 2 * 0 - (2 - 1) - 1) / 1 + 1)
        return int(size)

    def forward(self, x):
        x = self.sequential(x)
        x = self.sequential2(x)
        x = self.conv1(x)
        x = self.relu(x)

        # x = self.conv2(x)
        # x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.max_pool(x)
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
