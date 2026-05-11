"""DataLoader helpers for CIFAR-10 training scripts."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.data import VIT_TRAIN_TRANSFORM, VIT_TRANSFORM, get_cifar10


def create_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    eval_batch_size: int | None = None,
    num_workers: int = 4,
    device: str | torch.device = "cpu",
) -> tuple[DataLoader, DataLoader]:
    device = torch.device(device)
    train_dataset = get_cifar10(root=data_dir, train=True, transform=VIT_TRAIN_TRANSFORM)
    test_dataset = get_cifar10(root=data_dir, train=False, transform=VIT_TRANSFORM)
    pin_memory = device.type == "cuda"

    common_kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": pin_memory,
    }
    if int(num_workers) > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(eval_batch_size or batch_size),
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, test_loader
