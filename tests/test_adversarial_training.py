from __future__ import annotations

import torch

from src.training import AdversarialAttackConfig, evaluate_under_attack, generate_adversarial_examples


class TinyImageClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = x.mean(dim=(1, 2, 3))
        return torch.stack([score, -score], dim=1)


def test_fgsm_attack_respects_epsilon_and_clamp() -> None:
    model = TinyImageClassifier().eval()
    images = torch.zeros(2, 3, 4, 4)
    labels = torch.zeros(2, dtype=torch.long)
    config = AdversarialAttackConfig(method="fgsm", epsilon=0.1, clamp_min=-1.0, clamp_max=1.0)

    adversarial = generate_adversarial_examples(model, images, labels, config)
    delta = adversarial - images

    assert adversarial.shape == images.shape
    assert float(delta.abs().max()) <= 0.100001
    assert float(adversarial.max()) <= 1.0
    assert float(adversarial.min()) >= -1.0


def test_attack_evaluation_returns_robust_metrics() -> None:
    model = TinyImageClassifier().eval()
    dataset = [(torch.zeros(3, 4, 4), 0), (torch.ones(3, 4, 4), 0)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    config = AdversarialAttackConfig(method="pgd", epsilon=0.1, step_size=0.05, steps=2)

    metrics = evaluate_under_attack(model, loader, device="cpu", attack_config=config, amp=False)

    assert metrics["num_examples"] == 2
    assert 0.0 <= metrics["robust_accuracy"] <= 1.0
    assert 0.0 <= metrics["attack_success_rate"] <= 1.0
