"""Frame/mask pairing and train/test split for EndoVis 2017 sequences.

Sequences instrument_dataset_1..8 are split per-frame-index (first 225
frames train, last 75 test), following the standard EndoVis 2017
protocol. Sequences 9 and 10 are held out entirely for the stretch-goal
full-sequence evaluation and are not touched here.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from segmentation.crop import crop_camera_view

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# instrument_dataset_1..8; 9 and 10 are reserved for the stretch-goal
# held-out full-sequence evaluation.
TRAIN_SEQUENCES = range(1, 9)

FramePair = Tuple[Path, Path]


def list_frame_pairs(dataset_dir: Path) -> List[FramePair]:
    """Pairs each left frame with its binary mask for one sequence.

    Args:
        dataset_dir: Path to an instrument_dataset_N directory containing
            left_frames/ and binary_masks/ subdirectories.

    Returns:
        A list of (frame_path, mask_path) tuples, sorted by frame name.

    Raises:
        FileNotFoundError: If a frame's binary mask is missing.
    """
    frames_dir = dataset_dir / "left_frames"
    masks_dir = dataset_dir / "binary_masks"
    pairs = []
    for frame_path in sorted(frames_dir.glob("frame*.png")):
        mask_path = masks_dir / frame_path.name
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing binary mask for {frame_path}; run "
                "scripts/make_binary_masks.py first."
            )
        pairs.append((frame_path, mask_path))
    return pairs


def train_test_split(
    dataset_root: Path, train_frames: int = 225
) -> Tuple[List[FramePair], List[FramePair]]:
    """Splits instrument_dataset_1..8 into train/test frame pairs.

    Each sequence's own frames are split by index: the first
    `train_frames` frames go to train, the remainder go to test. This
    matches the standard EndoVis 2017 protocol. Sequences 9 and 10 are
    not included.

    Args:
        dataset_root: Directory containing instrument_dataset_N folders
            (e.g. dataset/training).
        train_frames: Number of leading frames per sequence used for
            training; the rest are used for testing.

    Returns:
        (train_pairs, test_pairs), each a flat list of (frame_path,
        mask_path) tuples across all 8 sequences.
    """
    train_pairs: List[FramePair] = []
    test_pairs: List[FramePair] = []
    for n in TRAIN_SEQUENCES:
        dataset_dir = dataset_root / f"instrument_dataset_{n}"
        pairs = list_frame_pairs(dataset_dir)
        train_pairs.extend(pairs[:train_frames])
        test_pairs.extend(pairs[train_frames:])
    return train_pairs, test_pairs


def load_image(path: Path, size: Tuple[int, int]) -> np.ndarray:
    """Loads an RGB frame, cropped to the camera view and resized."""
    image = crop_camera_view(Image.open(path).convert("RGB"))
    return np.array(image.resize(size))


def load_mask(path: Path, size: Tuple[int, int]) -> np.ndarray:
    """Loads a mask, cropped to the camera view and resized, as {0, 1}."""
    mask = crop_camera_view(Image.open(path).convert("L"))
    mask = mask.resize(size, Image.NEAREST)
    return (np.array(mask) > 127).astype(np.uint8)


class InstrumentSegDataset(Dataset):
    """Binary instrument segmentation dataset over (frame, mask) pairs."""

    # (width, height); divisible by 32 for the ResNet34 encoder's 5
    # downsampling stages, and the 5:4 aspect ratio of the cropped view.
    DEFAULT_SIZE = (320, 256)

    def __init__(
        self,
        pairs: List[FramePair],
        size: Tuple[int, int] = DEFAULT_SIZE,
    ):
        """
        Args:
            pairs: List of (frame_path, mask_path) tuples.
            size: (width, height) to resize images/masks to.
        """
        self.pairs = pairs
        self.size = size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_path, mask_path = self.pairs[idx]

        image = load_image(frame_path, self.size).astype(np.float32)
        image = (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        mask = load_mask(mask_path, self.size).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor
