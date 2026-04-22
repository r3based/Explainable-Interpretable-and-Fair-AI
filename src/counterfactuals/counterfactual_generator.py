from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class CounterfactualConfig:
    steps: int = 300
    step_size: float = 1e-2
    lambda_l2: float = 1e-2
    lambda_tv: float = 1e-4
    confidence_margin: float = 0.0
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    target_mode: str = "untargeted"
    log_every: int = 25


@dataclass
class CounterfactualResult:
    original_class: int
    final_class: int
    success: bool
    steps_run: int
    original_confidence: float
    final_confidence: float
    perturbation_l2: float
    perturbation_linf: float
    config: Dict
    original_image: torch.Tensor
    counterfactual_image: torch.Tensor
    perturbation: torch.Tensor
    history: Dict[str, list]


class GradientCounterfactualGenerator:
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(image.to(self.device))
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        return preds, confs

    def generate(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        config: Optional[CounterfactualConfig] = None,
    ) -> CounterfactualResult:
        config = config or CounterfactualConfig()
        if image.dim() != 4 or image.size(0) != 1:
            raise ValueError("Expected image shape [1, C, H, W].")

        x0 = image.detach().to(self.device)
        x_cf = x0.clone().detach().requires_grad_(True)

        with torch.no_grad():
            logits0 = self.model(x0)
            probs0 = F.softmax(logits0, dim=1)
            original_class = int(probs0.argmax(dim=1).item())
            original_confidence = float(probs0[0, original_class].item())

        if config.target_mode not in {"untargeted", "targeted"}:
            raise ValueError("target_mode must be 'untargeted' or 'targeted'.")
        if config.target_mode == "targeted" and target_class is None:
            raise ValueError("target_class must be provided for targeted mode.")
        if config.target_mode == "untargeted":
            target_class = None

        optimizer = torch.optim.Adam([x_cf], lr=config.step_size)
        history = {"total_loss": [], "attack_loss": [], "l2": [], "tv": [], "pred": [], "conf": []}

        best_success = False
        best_image = x0.clone().detach()
        best_pred = original_class
        best_conf = original_confidence
        best_l2 = float("inf")
        best_linf = 0.0
        steps_run = 0

        for step in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(x_cf)
            probs = F.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            pred_conf = float(probs[0, pred].item())

            attack_loss = self._attack_objective(
                logits=logits,
                original_class=original_class,
                target_class=target_class,
                mode=config.target_mode,
                confidence_margin=config.confidence_margin,
            )
            l2 = torch.norm((x_cf - x0).reshape(1, -1), p=2)
            tv = total_variation(x_cf)
            total_loss = attack_loss + config.lambda_l2 * l2 + config.lambda_tv * tv
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_cf.clamp_(config.clamp_min, config.clamp_max)

            history["total_loss"].append(float(total_loss.item()))
            history["attack_loss"].append(float(attack_loss.item()))
            history["l2"].append(float(l2.item()))
            history["tv"].append(float(tv.item()))
            history["pred"].append(pred)
            history["conf"].append(pred_conf)

            success = self._is_success(pred, original_class, target_class, config.target_mode)
            if success:
                with torch.no_grad():
                    delta = x_cf - x0
                    current_l2 = float(torch.norm(delta.reshape(1, -1), p=2).item())
                    current_linf = float(delta.abs().max().item())
                if current_l2 < best_l2:
                    best_success = True
                    best_image = x_cf.detach().clone()
                    best_pred = pred
                    best_conf = pred_conf
                    best_l2 = current_l2
                    best_linf = current_linf
                    if step > 10:
                        steps_run = step + 1
                        break
            steps_run = step + 1

        if not best_success:
            with torch.no_grad():
                logits_final = self.model(x_cf)
                probs_final = F.softmax(logits_final, dim=1)
                best_pred = int(probs_final.argmax(dim=1).item())
                best_conf = float(probs_final[0, best_pred].item())
                best_image = x_cf.detach().clone()
                delta = best_image - x0
                best_l2 = float(torch.norm(delta.reshape(1, -1), p=2).item())
                best_linf = float(delta.abs().max().item())

        perturbation = best_image - x0
        return CounterfactualResult(
            original_class=original_class,
            final_class=best_pred,
            success=best_success,
            steps_run=steps_run,
            original_confidence=original_confidence,
            final_confidence=best_conf,
            perturbation_l2=best_l2,
            perturbation_linf=best_linf,
            config=asdict(config),
            original_image=x0.detach().cpu(),
            counterfactual_image=best_image.detach().cpu(),
            perturbation=perturbation.detach().cpu(),
            history=history,
        )

    @staticmethod
    def _attack_objective(
        logits: torch.Tensor,
        original_class: int,
        target_class: Optional[int],
        mode: str,
        confidence_margin: float,
    ) -> torch.Tensor:
        if mode == "untargeted":
            target_logit = logits[:, original_class]
            other_logits = logits.clone()
            other_logits[:, original_class] = -1e9
            best_other = other_logits.max(dim=1).values
            return F.relu(target_logit - best_other + confidence_margin).mean()

        assert target_class is not None
        target_logit = logits[:, target_class]
        other_logits = logits.clone()
        other_logits[:, target_class] = -1e9
        best_other = other_logits.max(dim=1).values
        return F.relu(best_other - target_logit + confidence_margin).mean()

    @staticmethod
    def _is_success(pred: int, original_class: int, target_class: Optional[int], mode: str) -> bool:
        if mode == "untargeted":
            return pred != original_class
        return pred == target_class


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


def denormalize_image(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.view(1, -1, 1, 1).to(x.device)
    std = std.view(1, -1, 1, 1).to(x.device)
    return x * std + mean


def normalize_image(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.view(1, -1, 1, 1).to(x.device)
    std = std.view(1, -1, 1, 1).to(x.device)
    return (x - mean) / std


def generate_counterfactual_for_normalized_input(
    model: torch.nn.Module,
    normalized_image: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_class: Optional[int] = None,
    config: Optional[CounterfactualConfig] = None,
) -> CounterfactualResult:
    pixel_image = denormalize_image(normalized_image, mean, std).clamp(0.0, 1.0)

    class WrappedModel(torch.nn.Module):
        def __init__(self, base_model: torch.nn.Module):
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_norm = normalize_image(x, mean=mean, std=std)
            return self.base_model(x_norm)

    wrapped = WrappedModel(model)
    generator = GradientCounterfactualGenerator(wrapped)
    return generator.generate(pixel_image, target_class=target_class, config=config)
