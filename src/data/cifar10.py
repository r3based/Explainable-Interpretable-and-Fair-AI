"""CIFAR-10 data loading and preprocessing for ViT-B/16."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ViT-B/16 pretrained on ImageNet expects 224×224 with these stats
VIT_MEAN = (0.5, 0.5, 0.5)
VIT_STD  = (0.5, 0.5, 0.5)

VIT_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
])

VIT_TRANSFORM_UNNORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])


def get_cifar10(root: str = "./data", train: bool = False) -> datasets.CIFAR10:
    return datasets.CIFAR10(root=root, train=train, download=True, transform=VIT_TRANSFORM)


def get_loader(
    root: str = "./data",
    train: bool = False,
    batch_size: int = 32,
    num_workers: int = 2,
    shuffle: bool = False,
    subset_size: int | None = None,
    seed: int = 42,
) -> DataLoader:
    dataset = get_cifar10(root=root, train=train)
    if subset_size is not None:
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=rng)[:subset_size].tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ViT normalization, return values in [0, 1]."""
    mean = torch.tensor(VIT_MEAN, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(VIT_STD,  device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)
