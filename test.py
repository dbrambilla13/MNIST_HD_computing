from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy

from torchhd import functional
from torchhd import embeddings
from torchhd.utils import plot_similarity
import matplotlib.pyplot as plt

BATCH_SIZE = 32
device = 'cuda'
DIMENSIONS = 10000
NUM_LEVELS = 10
IMG_SIZE = 28

embs = embeddings.Random(10,DIMENSIONS)
ms =  functional.multiset(embs.weight)

plot_similarity(ms, embs.weight)
plot_similarity(functional.hard_quantize(ms), embs.weight)
plt.savefig('test.png')
