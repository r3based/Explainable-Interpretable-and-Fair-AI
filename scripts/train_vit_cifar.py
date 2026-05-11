from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import bootstrap  # noqa: F401

from src.model import ViTCIFAR10Classifier
from src.training import create_cifar10_loaders, evaluate_clean, train_one_epoch
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a ViT CIFAR-10 classifier.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-train-augment", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="artifacts/training")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/model_weights")
    parser.add_argument("--run-name", type=str, default="finetuned_vit")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    if device.type == "cuda" and not args.deterministic:
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = create_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
        train_augment=not args.no_train_augment,
    )

    model = ViTCIFAR10Classifier(
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        device=device,
    )
    trainable_params = model.trainable_parameter_count()
    total_params = model.total_parameter_count()
    print(f"Device: {device}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    amp = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and amp)
    best_accuracy = -1.0
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            amp=amp,
            max_batches=args.max_train_batches,
            log_every=args.log_every,
        )
        val_metrics = evaluate_clean(
            model=model,
            loader=val_loader,
            device=device,
            amp=amp,
            max_batches=args.max_val_batches,
        )
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} "
            f"loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['clean_accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        latest_path = checkpoint_dir / f"{args.run_name}_latest.pt"
        model.save_checkpoint(
            latest_path,
            extra={"args": vars(args), "history": history, "best_accuracy": best_accuracy},
        )
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = float(val_metrics["accuracy"])
            best_path = checkpoint_dir / f"{args.run_name}_best.pt"
            model.save_checkpoint(
                best_path,
                extra={"args": vars(args), "history": history, "best_accuracy": best_accuracy},
            )

        with open(output_dir / f"{args.run_name}_history.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "args": vars(args),
                    "device": str(device),
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "best_accuracy": best_accuracy,
                    "history": history,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
