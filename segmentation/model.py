"""U-Net construction, device selection and inference helpers.

Shared by training (scripts/training/train_model.py) and evaluation
(scripts/evaluation/evaluate_model.py) so both build the exact same
architecture and preprocess frames identically.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image

from segmentation.crop import crop_camera_view
from segmentation.dataset import IMAGENET_MEAN, IMAGENET_STD


def get_device() -> torch.device:
    """Selects MPS (Apple Silicon GPU) if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_unet(pretrained_encoder: bool) -> torch.nn.Module:
    """Builds the U-Net (ResNet34 encoder) with a single logit output.

    Args:
        pretrained_encoder: If True, initialize the encoder from
            ImageNet weights (used for fresh training). If False, skip
            the download -- appropriate when a checkpoint will
            immediately overwrite every weight anyway.
    """
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained_encoder else None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Builds a U-Net, loads trained weights and switches to eval mode."""
    model = build_unet(pretrained_encoder=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def make_predict_fn(
    model: torch.nn.Module,
    device: torch.device,
    size: Tuple[int, int],
    threshold: float = 0.5,
) -> Callable[[Path], np.ndarray]:
    """Builds a predict_fn(frame_path) -> boolean mask at native resolution.

    The model operates at `size` (e.g. 320x256); the prediction is
    resized back up to the cropped frame's native resolution so it can
    be compared pixel-for-pixel against the full-resolution ground
    truth, the same way the classical baseline's predictions are.

    Args:
        model: Trained network in eval mode.
        device: Device the model lives on.
        size: (width, height) the network input is resized to.
        threshold: Probability above which a pixel is foreground.
    """

    def predict(frame_path: Path) -> np.ndarray:
        full = crop_camera_view(Image.open(frame_path).convert("RGB"))
        resized = np.array(full.resize(size)).astype(np.float32)
        resized = (resized / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))
        pred = (probs > threshold).float().cpu().squeeze().numpy()

        pred_img = Image.fromarray((pred * 255).astype(np.uint8))
        pred_img = pred_img.resize(full.size, Image.NEAREST)
        return np.array(pred_img) > 127

    return predict
