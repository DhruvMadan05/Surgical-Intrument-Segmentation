"""Fine-tunes a U-Net (ResNet34 encoder) for binary instrument segmentation.

Trains on the standard EndoVis 2017 train split (instrument_dataset_1..8,
first 225 frames per sequence) with a Dice loss, which optimizes mask
overlap directly and is therefore robust to the instrument/background
class imbalance without needing manually tuned class weights. Saves a
checkpoint, a per-epoch loss log, and qualitative sanity-check overlays
on a handful of training frames so it's visually obvious whether the
model learned something, not just that the loss went down.

Full test-set IoU/Dice evaluation (to compare against the classical
baseline in results/baseline_threshold.json) is left to a follow-up
evaluate.py script; this script's job is to get the model fine-tuned and
producing masks on training data.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from segmentation.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    InstrumentSegDataset,
    train_test_split,
)


def get_device() -> torch.device:
    """Selects MPS (Apple Silicon GPU) if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Soft Dice loss from raw (pre-sigmoid) logits."""
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    loss = 1 - (2 * intersection + eps) / (union + eps)
    return loss.mean()


def save_sanity_overlays(
    model: torch.nn.Module,
    dataset: InstrumentSegDataset,
    device: torch.device,
    out_dir: Path,
    num_examples: int = 6,
) -> None:
    """Saves input/ground-truth/prediction overlays for a few frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(min(num_examples, len(dataset))):
            image, mask = dataset[i]
            logits = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float()
            pred = pred.cpu().squeeze().numpy()

            denorm = image.numpy().transpose(1, 2, 0)
            denorm = np.clip(denorm * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(denorm)
            axes[0].set_title("input")
            axes[1].imshow(mask.squeeze().numpy(), cmap="gray")
            axes[1].set_title("ground truth")
            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title("prediction")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"train_frame_{i}.png")
            plt.close(fig)
    model.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset" / "training",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Train on only the first N frames, for a quick overfit "
            "sanity check of the training loop"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "checkpoints" / "unet_resnet34.pt",
    )
    parser.add_argument(
        "--log", type=Path, default=repo_root / "results" / "train_log.csv"
    )
    parser.add_argument(
        "--overlays-dir",
        type=Path,
        default=repo_root / "results" / "sanity_overlays",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_pairs, _ = train_test_split(args.dataset_root)
    if args.limit:
        train_pairs = train_pairs[: args.limit]
    print(f"Training on {len(train_pairs)} frames")

    dataset = InstrumentSegDataset(train_pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "mean_loss"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_losses = []
            for images, masks in loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = dice_loss(logits, masks)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses))
            print(
                f"Epoch {epoch}/{args.epochs}: mean dice loss "
                f"= {mean_loss:.4f}"
            )
            writer.writerow([epoch, mean_loss])
            log_file.flush()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")

    save_sanity_overlays(model, dataset, device, args.overlays_dir)
    print(f"Saved sanity overlays to {args.overlays_dir}")


if __name__ == "__main__":
    main()
