from typing import List

from torch import nn


class MultilayerPerceptron(nn.Module):
    def __init__(self, input: int, hidden: List[int], output: int, activation_default_function,
                 activation_output_function, **kwargs):
        super(MultilayerPerceptron, self).__init__(**kwargs)

        self.fc1 = nn.Linear(input, hidden[0] if len(hidden) > 0 else output)
        self.hidden = []

        for h in range(1, len(hidden)):
            self.hidden.append(nn.Linear(hidden[h - 1], hidden[h]))
        self.hidden = nn.ModuleList(self.hidden)
        self.fc2 = nn.Linear(hidden[-1] if len(hidden) > 0 else output, output)
        self.activation_default_function = activation_default_function
        self.activation_output_function = activation_output_function

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_default_function(x)
        for h in self.hidden:
            x = h(x)
            x = self.activation_default_function(x)
        x = self.fc2(x)
        x = self.activation_output_function(x)
        return x
