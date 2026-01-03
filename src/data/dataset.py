from torchvision import datasets
from transforms import train_transform, test_transform

train_dataset = datasets.ImageFolder(
    root='data/processed/train',
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root='data/processed/test',
    transform=test_transform
)