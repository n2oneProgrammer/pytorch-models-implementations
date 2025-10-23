from torch import nn


class Perceptron(nn.Module):
    def __init__(self, input: int,output:int, activation_function):
        super(Perceptron, self).__init__()

        self.fc1 = nn.Linear(input, output)
        self.activation = activation_function

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        return x
