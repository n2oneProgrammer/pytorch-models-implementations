import os

import numpy as np
import torch
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt

from AlexNet.model import AlexNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

classes = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]

batch_size = 64
n_epochs = 200
learning_rate = 0.001

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.Flowers102(root='../data/',
                                                split="train",
                                                transform=train_transforms,
                                                download=True)

test_dataset = torchvision.datasets.Flowers102(root='../data/',
                                               split="test",
                                               transform=val_transforms)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True, num_workers=4)


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
    model = AlexNet(102, in_channel=3, width=224, height=224)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
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
        print('[{}] Epoch [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}, Acc: {:.4f}, Val Acc: {:.4f}'
              .format(name, epoch + 1,
                      n_epochs,
                      running_loss / len(train_loader),
                      val_loss / len(test_loader),
                      correct / len(train_loader),
                      val_correct / len(test_loader)))
    os.makedirs(f'results/{name}', exist_ok=True)
    plt.plot(losses, label='Training Loss')
    plt.plot(accuracy, label='Training Accuracy')
    plt.legend()
    plt.ylim([0, 1])
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(f"results/{name}/losses.png")
    plt.close()

    plt.plot(val_losses, label='Validation Loss')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.ylim([0, 1])
    plt.title("Validation Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(f"results/{name}/validation_losses.png")
    plt.close()
    show_predictions_grid(f"results/{name}/predict.png", model, test_loader, classes)


def train_runner(p):
    torch.cuda.init()
    train(p)


if __name__ == '__main__':
    train_runner("AlexNet")
