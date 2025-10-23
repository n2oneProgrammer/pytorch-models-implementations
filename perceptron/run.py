import os

import numpy as np
import torch
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from perceptron.model import Perceptron
from multiprocessing import Pool

batch_size = 128
n_epochs = 100
learning_rate = 0.01
train_dataset = torchvision.datasets.MNIST(root='../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def show_predictions_grid(path_save, model, test_loader, class_names):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:16], labels[:16]

    with torch.no_grad():
        images_flatten = torch.flatten(images, 1)
        outputs = model(images_flatten)
        preds = outputs.argmax(1)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(16):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]

        color = 'green' if preds[i] == labels[i] else 'red'
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=9)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(path_save)
    plt.close()


def train(name: str, activation_function):
    model = Perceptron(784, 10, activation_function)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    val_losses = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            model.train()
            inputs, labels = data
            inputs = torch.flatten(inputs, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = torch.flatten(x, 1)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
        losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(test_loader))
        print('[{}] Epoch [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}'.format(name, epoch + 1, n_epochs,
                                                                          running_loss / len(train_loader),
                                                                          val_loss / len(test_loader)))
    os.makedirs(f'results/{name}', exist_ok=True)
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"results/{name}/losses.png")
    plt.close()

    plt.plot(val_losses)
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.savefig(f"results/{name}/validation_losses.png")
    plt.close()
    show_predictions_grid(f"results/{name}/predict.png", model, test_loader, test_dataset.classes)

def train_runner(p):
    train(p[0],p[1])
if __name__ == '__main__':
    with Pool(5) as p:
        p.map(train_runner, [
            ("Relu", torch.nn.ReLU()),
            ("LeakyRelu", torch.nn.LeakyReLU()),
            ("Sigmoid", torch.nn.Sigmoid()),
            ("Tanh", torch.nn.Tanh()),
            ("Softmax", torch.nn.Softmax()),
        ])
