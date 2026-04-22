import torch
import torch.nn as nn

from .vit import ViTWrapper


class ViTCounterfactualAdapter(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.vit = ViTWrapper(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Возвращает дифференцируемые logits только для 10 CIFAR-10 anchor-классов.
        НИКАКОГО no_grad и НИКАКОГО softmax/log(prob) здесь не делаем.
        """
        logits_1000 = self.vit(x)  # forward() у wrapper дифференцируемый
        logits_10 = logits_1000[:, self.vit._anchor_idx]
        return logits_10