"""Evaluates a trained checkpoint on the official EndoVis 2017 test split.

Loads a U-Net checkpoint (see scripts/train_model.py) and runs it on the
test-split frames (instrument_dataset_1..8, last 75 frames per
sequence), scored against the official BinarySegmentation masks (see
scripts/download_test_masks.py) using the same segmentation.metrics
code as the classical baseline, so the two are directly comparable.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image

from segmentation.crop import crop_camera_view
from segmentation.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    InstrumentSegDataset,
)
from segmentation.dataset import test_pairs as load_test_pairs
from segmentation.metrics import evaluate_predictions
from segmentation.visualize import save_prediction_overlays


def get_device() -> torch.device:
    """Selects MPS (Apple Silicon GPU) if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Builds a U-Net (ResNet34 encoder) and loads trained weights.

    encoder_weights=None skips the ImageNet-pretrained download, since
    load_state_dict immediately overwrites every weight anyway.
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def make_predict_fn(model, device, size):
    """Builds a predict_fn(frame_path) -> mask at native crop resolution.

    The model operates at `size` (e.g. 320x256); the prediction is
    resized back up to the cropped frame's native resolution so it can
    be compared pixel-for-pixel against the full-resolution ground
    truth, the same way the classical baseline's predictions are.
    """

    def predict(frame_path: Path) -> np.ndarray:
        full = crop_camera_view(Image.open(frame_path).convert("RGB"))
        resized = np.array(full.resize(size)).astype(np.float32)
        resized = (resized / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))
        pred = (probs > 0.5).float().cpu().squeeze().numpy()

        pred_img = Image.fromarray((pred * 255).astype(np.uint8))
        pred_img = pred_img.resize(full.size, Image.NEAREST)
        return np.array(pred_img) > 127

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset" / "training",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=None,
        help=(
            "Directory with official test masks (default: a 'test' "
            "sibling of --dataset-root)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "checkpoints" / "unet_resnet34.pt",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test frames (quick sanity check)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "results" / "model_eval.json",
    )
    parser.add_argument(
        "--overlays-dir",
        type=Path,
        default=repo_root / "results" / "model_eval_overlays",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    pairs = load_test_pairs(args.dataset_root, args.test_root)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Evaluating on {len(pairs)} test frames")

    model = load_model(args.checkpoint, device)
    predict_fn = make_predict_fn(
        model, device, InstrumentSegDataset.DEFAULT_SIZE
    )

    results = evaluate_predictions(pairs, predict_fn)
    print(
        f"Evaluated {results['num_frames']} test frames: "
        f"mean IoU={results['mean_iou']:.4f}, "
        f"mean Dice={results['mean_dice']:.4f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "method": "unet_resnet34",
                "checkpoint": str(args.checkpoint),
                "mean_iou": results["mean_iou"],
                "mean_dice": results["mean_dice"],
                "num_frames": results["num_frames"],
            },
            f,
            indent=2,
        )
    print(f"Saved results to {args.out}")

    save_prediction_overlays(pairs, predict_fn, args.overlays_dir)
    print(f"Saved overlays to {args.overlays_dir}")


if __name__ == "__main__":
    main()
