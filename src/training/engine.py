"""Training and evaluation loops for CIFAR-10 ViT experiments."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .attacks import AdversarialAttackConfig, generate_adversarial_examples


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def _autocast_enabled(device: torch.device, enabled: bool) -> bool:
    return bool(enabled and device.type == "cuda")


def _autocast_device_type(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = True,
    attack_config: AdversarialAttackConfig | None = None,
    adversarial_weight: float = 1.0,
    max_batches: int | None = None,
    log_every: int = 0,
) -> dict[str, Any]:
    """Run one training epoch, optionally with adversarial examples."""
    device = torch.device(device)
    model.train()
    started = time.perf_counter()

    total = 0
    clean_correct = 0.0
    adv_correct = 0.0
    loss_sum = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(_autocast_device_type(device), enabled=_autocast_enabled(device, amp)):
            clean_logits = model(images)
            clean_loss = F.cross_entropy(clean_logits, labels)
            loss = clean_loss

        adv_logits = None
        if attack_config is not None:
            with torch.enable_grad():
                adv_images = generate_adversarial_examples(model, images, labels, attack_config)
            with torch.amp.autocast(_autocast_device_type(device), enabled=_autocast_enabled(device, amp)):
                adv_logits = model(adv_images)
                adv_loss = F.cross_entropy(adv_logits, labels)
                loss = clean_loss + float(adversarial_weight) * adv_loss

        if scaler is not None and _autocast_enabled(device, amp):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = int(labels.shape[0])
        total += batch_size
        loss_sum += float(loss.detach().item()) * batch_size
        clean_correct += float((clean_logits.detach().argmax(dim=1) == labels).sum().item())
        if adv_logits is not None:
            adv_correct += float((adv_logits.detach().argmax(dim=1) == labels).sum().item())

        if log_every > 0 and (batch_idx + 1) % int(log_every) == 0:
            processed = batch_idx + 1
            total_batches = len(loader) if max_batches is None else min(int(max_batches), len(loader))
            print(
                f"epoch={epoch:03d} batch={processed:04d}/{total_batches:04d} "
                f"loss={loss_sum / max(total, 1):.4f} "
                f"clean_acc={clean_correct / max(total, 1):.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    return {
        "epoch": int(epoch),
        "loss": loss_sum / max(total, 1),
        "clean_accuracy": clean_correct / max(total, 1),
        "adversarial_accuracy": None if attack_config is None else adv_correct / max(total, 1),
        "num_examples": total,
        "seconds": elapsed,
        "attack": None if attack_config is None else asdict(attack_config),
    }


@torch.no_grad()
def evaluate_clean(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    amp: bool = True,
    max_batches: int | None = None,
    log_every: int = 0,
    phase_name: str = "clean_eval",
) -> dict[str, Any]:
    device = torch.device(device)
    model.eval()
    started = time.perf_counter()

    total = 0
    correct = 0.0
    loss_sum = 0.0
    confidence_sum = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(_autocast_device_type(device), enabled=_autocast_enabled(device, amp)):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        probs = logits.softmax(dim=1)
        confs, preds = probs.max(dim=1)
        batch_size = int(labels.shape[0])
        total += batch_size
        correct += float((preds == labels).sum().item())
        loss_sum += float(loss.item()) * batch_size
        confidence_sum += float(confs.sum().item())
        if log_every > 0 and (batch_idx + 1) % int(log_every) == 0:
            processed = batch_idx + 1
            total_batches = len(loader) if max_batches is None else min(int(max_batches), len(loader))
            print(
                f"{phase_name} batch={processed:04d}/{total_batches:04d} "
                f"loss={loss_sum / max(total, 1):.4f} "
                f"acc={correct / max(total, 1):.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "mean_confidence": confidence_sum / max(total, 1),
        "num_examples": total,
        "seconds": elapsed,
    }


def evaluate_under_attack(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    attack_config: AdversarialAttackConfig,
    amp: bool = True,
    max_batches: int | None = None,
    log_every: int = 0,
    phase_name: str | None = None,
) -> dict[str, Any]:
    device = torch.device(device)
    model.eval()
    started = time.perf_counter()

    total = 0
    clean_correct = 0.0
    robust_correct = 0.0
    clean_confidence_sum = 0.0
    robust_confidence_sum = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast(_autocast_device_type(device), enabled=_autocast_enabled(device, amp)):
            clean_logits = model(images)
            clean_probs = clean_logits.softmax(dim=1)
            clean_confs, clean_preds = clean_probs.max(dim=1)

        adv_images = generate_adversarial_examples(model, images, labels, attack_config)

        with torch.no_grad(), torch.amp.autocast(_autocast_device_type(device), enabled=_autocast_enabled(device, amp)):
            robust_logits = model(adv_images)
            robust_probs = robust_logits.softmax(dim=1)
            robust_confs, robust_preds = robust_probs.max(dim=1)

        batch_size = int(labels.shape[0])
        total += batch_size
        clean_correct += float((clean_preds == labels).sum().item())
        robust_correct += float((robust_preds == labels).sum().item())
        clean_confidence_sum += float(clean_confs.sum().item())
        robust_confidence_sum += float(robust_confs.sum().item())
        if log_every > 0 and (batch_idx + 1) % int(log_every) == 0:
            processed = batch_idx + 1
            total_batches = len(loader) if max_batches is None else min(int(max_batches), len(loader))
            label = phase_name or f"{attack_config.method}_eval"
            print(
                f"{label} batch={processed:04d}/{total_batches:04d} "
                f"clean_acc={clean_correct / max(total, 1):.4f} "
                f"robust_acc={robust_correct / max(total, 1):.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    return {
        "clean_accuracy": clean_correct / max(total, 1),
        "robust_accuracy": robust_correct / max(total, 1),
        "attack_success_rate": 1.0 - robust_correct / max(total, 1),
        "clean_mean_confidence": clean_confidence_sum / max(total, 1),
        "robust_mean_confidence": robust_confidence_sum / max(total, 1),
        "num_examples": total,
        "seconds": elapsed,
        "attack": asdict(attack_config),
    }
