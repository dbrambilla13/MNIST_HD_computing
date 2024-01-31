import torch
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

DIMENSIONS = 10000

import torchvision.models as models

# cnn_encoder = models.resnet50()
# model = models.alexnet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
# densenet = models.densenet161()
# inception = models.inception_v3()
# googlenet = models.googlenet()
# shufflenet = models.shufflenet_v2_x1_0()
# mobilenet_v2 = models.mobilenet_v2()
# mobilenet_v3_large = models.mobilenet_v3_large()
# mobilenet_v3_small = models.mobilenet_v3_small()
# resnext50_32x4d = models.resnext50_32x4d()
# wide_resnet50_2 = models.wide_resnet50_2()
# mnasnet = models.mnasnet1_0()


device = "cuda"

BATCH_SIZE = 32
DIMENSIONS = 10000


normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = T.Compose([T.ToTensor(), normalize])
train_ds = CIFAR100("../data", train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = CIFAR100("../data", train=False, transform=transform, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
N_CLASSES = len(train_ds.classes)


for x, l in test_ld:
    print(x.shape)
    IMG_SIZE = x.shape[-1]
    print(f"image size = {IMG_SIZE}")
    break


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.value = embeddings.Level(NUM_LEVELS, DIMENSIONS)

        self.classify = nn.Linear(DIMENSIONS, num_classes, bias=False)
        self.classify.weight.data.fill_(0.0)

        self.cnn_encoder = models.resnet50()
        for param in self.cnn_encoder.parameters():
            param.requires_grad = False

        num_features = self.cnn_encoder.fc.in_features
        self.cnn_encoder.fc = torch.nn.Linear(
            in_features=num_features, out_features=DIMENSIONS, bias=False
        )
        # self.cnn_encoder.fc = torch.nn.Identity()

    def encode(self, x):
        # extract features from images with CNN and create HD vector
        x = self.cnn_encoder(x)
        # apply sigmoid and quantization to get bipolar vector
        # x = torch.sigmoid(x)
        x = functional.hard_quantize(x)
        return x

    def forward(self, x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit


with torch.set_grad_enabled(False):
    model = Model(num_classes=N_CLASSES)

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
