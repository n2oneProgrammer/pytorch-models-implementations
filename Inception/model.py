from typing import List

import numpy as np
import torch
from torch import nn


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int,
                 ch1x1: int,
                 ch3x3red: int,
                 ch3x3: int,
                 ch5x5red: int,
                 ch5x5: int,
                 pool_proj: int):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Conv2d(in_channels=in_channels, out_channels=ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1),
            nn.Conv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1),
            nn.Conv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class Inception(nn.Module):
    def __init__(self, output, in_channel=1, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=7 // 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=3 // 2),
            nn.LocalResponseNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=3 // 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=3 // 2),
            nn.LocalResponseNorm(192),
            nn.ReLU(),
            InceptionModule(192, 64, 96, 128, 16, 32, 32),
            InceptionModule(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=3 // 2),
            InceptionModule(480, 192, 96, 208, 16, 48, 64),
            InceptionModule(512, 160, 112, 224, 24, 64, 64),
            InceptionModule(512, 128, 128, 256, 24, 64, 64),
            InceptionModule(512, 112, 144, 288, 32, 64, 64),
            InceptionModule(528, 256, 160, 320, 32, 128, 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=3 // 2),
            InceptionModule(832, 256, 160, 320, 32, 128, 128),
            InceptionModule(832, 384, 192, 384, 48, 128, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_features=1024, out_features=output),
        )
        self.fc3 = nn.Linear(in_features=512, out_features=output)

    def forward(self, x):
        x = self.sequence(x)
        return x
