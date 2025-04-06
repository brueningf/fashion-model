import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    plt.subplot(rows, cols, i)

    sample_idx = torch.randint(len(training_data), (1,)).item()
    image, label = training_data[sample_idx]

    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(image.squeeze(), cmap="gray")
plt.show()

