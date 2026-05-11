from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import bootstrap  # noqa: F401

from src.data import VIT_TRANSFORM, get_cifar10
from src.model import load_project_model
from src.training import AdversarialAttackConfig, evaluate_clean, evaluate_under_attack
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clean and adversarial CIFAR-10 accuracy.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="artifacts/eval/classifier")
    parser.add_argument("--model-kind", type=str, default="anchor", choices=["anchor", "finetuned", "robust"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--attacks", nargs="*", default=["fgsm", "pgd"], choices=["fgsm", "pgd"])
    parser.add_argument("--epsilon", type=float, default=0.031)
    parser.add_argument("--attack-step-size", type=float, default=0.008)
    parser.add_argument("--attack-steps", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_cifar10(root=args.data_dir, train=False, transform=VIT_TRANSFORM)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_project_model(
        model_kind=args.model_kind,
        checkpoint=args.checkpoint,
        device=device,
        model_name=args.model_name,
        require_checkpoint=args.model_kind != "anchor" and not args.allow_random_init,
    )
    classifier = model.as_cifar10_classifier()
    classifier.eval()

    amp = not args.no_amp
    print(
        f"Evaluating {args.model_kind} on {device}; "
        f"batch_size={args.batch_size}, max_batches={args.max_batches}, attacks={args.attacks}",
        flush=True,
    )
    print("Starting clean evaluation", flush=True)
    clean_metrics = evaluate_clean(
        model=classifier,
        loader=loader,
        device=device,
        amp=amp,
        max_batches=args.max_batches,
        log_every=args.log_every,
        phase_name="clean_eval",
    )
    print(
        f"Finished clean evaluation: accuracy={clean_metrics['accuracy']:.4f}, "
        f"loss={clean_metrics['loss']:.4f}",
        flush=True,
    )

    attack_metrics = {}
    for attack_name in args.attacks:
        print(
            f"Starting {attack_name} evaluation "
            f"(epsilon={args.epsilon}, steps={args.attack_steps}, step_size={args.attack_step_size})",
            flush=True,
        )
        attack_config = AdversarialAttackConfig(
            method=attack_name,
            epsilon=args.epsilon,
            step_size=args.attack_step_size,
            steps=args.attack_steps,
        )
        attack_metrics[attack_name] = evaluate_under_attack(
            model=classifier,
            loader=loader,
            device=device,
            attack_config=attack_config,
            amp=amp,
            max_batches=args.max_batches,
            log_every=args.log_every,
            phase_name=f"{attack_name}_eval",
        )
        print(
            f"Finished {attack_name}: robust_accuracy={attack_metrics[attack_name]['robust_accuracy']:.4f}, "
            f"attack_success_rate={attack_metrics[attack_name]['attack_success_rate']:.4f}",
            flush=True,
        )

    payload = {
        "args": vars(args),
        "device": str(device),
        "clean": clean_metrics,
        "attacks": attack_metrics,
    }
    output_path = output_dir / f"{args.model_kind}_classifier_metrics.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved classifier metrics to {output_path}", flush=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
