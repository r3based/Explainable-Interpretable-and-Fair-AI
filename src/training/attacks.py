"""Adversarial perturbations used for robustness training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


AttackName = Literal["fgsm", "pgd"]


@dataclass
class AdversarialAttackConfig:
    method: AttackName = "fgsm"
    epsilon: float = 0.031
    step_size: float = 0.008
    steps: int = 4
    random_start: bool = True
    clamp_min: float = -1.0
    clamp_max: float = 1.0


def fgsm_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.031,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """Single-step adversarial perturbation in normalized image space."""
    x_adv = images.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
    x_adv = x_adv + float(epsilon) * grad.sign()
    return x_adv.clamp(float(clamp_min), float(clamp_max)).detach()


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.031,
    step_size: float = 0.008,
    steps: int = 4,
    random_start: bool = True,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """Projected gradient descent adversarial perturbation in normalized image space."""
    x_orig = images.detach()
    if random_start:
        noise = torch.empty_like(x_orig).uniform_(-float(epsilon), float(epsilon))
        x_adv = (x_orig + noise).clamp(float(clamp_min), float(clamp_max)).detach()
    else:
        x_adv = x_orig.clone().detach()

    for _ in range(int(steps)):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + float(step_size) * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-float(epsilon), max=float(epsilon))
        x_adv = (x_orig + delta).clamp(float(clamp_min), float(clamp_max)).detach()

    return x_adv


def generate_adversarial_examples(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: AdversarialAttackConfig,
) -> torch.Tensor:
    if config.method == "fgsm":
        return fgsm_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=config.epsilon,
            clamp_min=config.clamp_min,
            clamp_max=config.clamp_max,
        )
    if config.method == "pgd":
        return pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=config.epsilon,
            step_size=config.step_size,
            steps=config.steps,
            random_start=config.random_start,
            clamp_min=config.clamp_min,
            clamp_max=config.clamp_max,
        )
    raise ValueError("Unsupported attack method. Use 'fgsm' or 'pgd'.")
