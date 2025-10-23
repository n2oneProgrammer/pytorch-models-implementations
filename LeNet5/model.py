from typing import List

from torch import nn


class LeNet5(nn.Module):
    def __init__(self, output, **kwargs):
        super(LeNet5, self).__init__(**kwargs)

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # 14x14x6
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, padding=0, stride=2)  # 10x10x6
        )
        self.sequential2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # 5x5x16
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, padding=0, stride=2))  # 5x5x16

        self.fc1 = nn.Linear(in_features=5 * 5 * 16, out_features=120)
        self.signmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=output)

    def forward(self, x):
        x = self.sequential(x)
        x = self.sequential2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.signmoid(x)
        x = self.fc2(x)
        x = self.signmoid(x)
        x = self.fc3(x)
        return x
