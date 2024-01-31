from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST, FashionMNIST, CIFAR100, CIFAR10
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms as T

from torchhd import functional
from torchhd import embeddings

device = "cuda"

BATCH_SIZE = 32
DIMENSIONS = 10000
NUM_LEVELS = 256

# dataset = "MNIST"
# dataset = "FashionMNIST"
dataset = "CIFAR100"
# dataset = "CIFAR10"




if dataset == "MNIST":
    transform = torchvision.transforms.ToTensor()

    train_ds = MNIST("../data", train=True, transform=transform, download=True)
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    test_ds = MNIST("../data", train=False, transform=transform, download=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

if dataset == "FashionMNIST":
    transform = torchvision.transforms.ToTensor()

    train_ds = FashionMNIST("../data", train=True, transform=transform, download=True)
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    test_ds = FashionMNIST("../data", train=False, transform=transform, download=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)


if dataset == "CIFAR100":
    transform = T.Compose([
        T.Grayscale(), 
        T.ToTensor()
    ])
    train_ds = CIFAR100("../data", train=True, transform=transform, download=True)
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    test_ds = CIFAR100("../data", train=False, transform=transform, download=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(train_ds.classes)


if dataset == "CIFAR10":
    transform = T.Compose([
        T.Grayscale(), 
        T.ToTensor()
    ])
    train_ds = CIFAR10("../data", train=True, transform=transform, download=True)
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    test_ds = CIFAR10("../data", train=False, transform=transform, download=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    print(train_ds.classes)


for x, l in test_ds:
    print(x.shape)
    IMG_SIZE = x.shape[-1]
    print(f"image size = {IMG_SIZE}")
    # exit()
    break


class Model(nn.Module):
    def __init__(self, num_classes, size):
        super(Model, self).__init__()

        self.flatten = torch.nn.Flatten()
        self.size = size

        # self.position = embeddings.Random(size * size, DIMENSIONS).weight
        self.position_x = embeddings.Random(size,DIMENSIONS)
        self.position_y = embeddings.Random(size,DIMENSIONS)

        px = self.position_x(torch.arange(size).repeat(size))
        py = self.position_y(torch.arange(size).repeat_interleave(size))

        self.position = functional.hard_quantize(functional.bundle(px.to(device), py.to(device)))

        self.value = embeddings.Level(NUM_LEVELS, DIMENSIONS)

        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

    def encode(self, x):
        x = self.flatten(x)
        sample_hv = functional.bind(self.position, self.value(x))
        sample_hv = functional.multiset(sample_hv)
        return functional.hard_quantize(sample_hv)

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit

with torch.set_grad_enabled(False):
    model = Model(len(train_ds.classes), IMG_SIZE)

    model = model.to(device)

    for samples, labels in tqdm(train_ld):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = model.encode(samples)

        model.classify.weight[labels] = functional.bundle(
            model.classify.weight[labels], samples_hv
        )

    model.classify.weight[:] = functional.hard_quantize(model.classify.weight)

    train_accuracy = MulticlassAccuracy(num_classes=len(train_ds.classes))
    for samples, labels in tqdm(train_ld):
        samples = samples.to(device)

        outputs = model(samples)
        predictions = torch.argmax(outputs, dim=-1)
        train_accuracy.update(predictions.cpu(), labels)

    print(train_accuracy.compute())

    test_accuracy = MulticlassAccuracy(num_classes=len(train_ds.classes))

    for samples, labels in tqdm(test_ld):
        samples = samples.to(device)

        outputs = model(samples)
        predictions = torch.argmax(outputs, dim=-1)
        test_accuracy.update(predictions.cpu(), labels)

    print(test_accuracy.compute())
