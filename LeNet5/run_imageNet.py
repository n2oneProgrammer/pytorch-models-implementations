import os

import numpy as np
import torch
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from multiprocessing import Pool

from LeNet5.model import LeNet5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
batch_size = 128
n_epochs = 100
learning_rate = 0.001
train_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../data/',
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
        images_flatten = images.to(device)
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


def train(name: str):
    print(name, "Training...")
    model = LeNet5(10, in_channel=3, width=32, height=32)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    accuracy = []
    losses = []
    val_losses = []
    val_accuracy = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        correct = 0.0
        for i, data in enumerate(train_loader, 0):
            model.train()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).float().sum().item() / batch_size

        model.eval()
        val_loss = 0.0
        val_correct = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
                preds = preds.argmax(1)
                val_correct += (preds == y).float().sum().item() / batch_size
        losses.append(running_loss / len(train_loader))
        accuracy.append(correct / len(train_loader))
        val_losses.append(val_loss / len(test_loader))
        val_accuracy.append(val_correct / len(test_loader))
        scheduler.step(val_loss / len(test_loader))
        print('[{}] Epoch [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}'.format(name, epoch + 1, n_epochs,
                                                                          running_loss / len(train_loader),
                                                                          val_loss / len(test_loader)))
    os.makedirs(f'results/{name}', exist_ok=True)
    plt.plot(losses, label='Training Loss')
    plt.plot(accuracy, label='Training Accuracy')
    plt.legend()
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(f"results/{name}/losses.png")
    plt.close()

    plt.plot(val_losses, label='Validation Loss')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(f"results/{name}/validation_losses.png")
    plt.close()
    show_predictions_grid(f"results/{name}/predict.png", model, test_loader, test_dataset.classes)


def train_runner(p):
    torch.cuda.init()
    train(p)


if __name__ == '__main__':
    train_runner("ImageNet")
