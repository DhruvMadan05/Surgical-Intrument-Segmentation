"""Frame/mask pairing and train/test split for EndoVis 2017 sequences.

Sequences instrument_dataset_1..8 are split per-frame-index (first 225
frames train, last 75 test), following the standard EndoVis 2017
protocol. Sequences 9 and 10 are held out entirely for the stretch-goal
full-sequence evaluation and are not touched here.

Train and test pairs come from different mask sources:
  - Train frames are scored against our own self-computed binary_masks/
    (scripts/make_binary_masks.py), since no official ground truth
    exists for them.
  - Test frames are scored against the official EndoVis 2017
    BinarySegmentation masks (scripts/download_test_masks.py), so
    results are comparable with published benchmarks that were also
    scored against the official test masks. This matters: spot-checks
    showed our self-computed masks agree with the official ones on
    30/32 sampled frames, but diverge on a few frames in sequences 1-2
    where the official release appears to omit a visible instrument
    near the frame edge -- using the official masks for evaluation
    keeps us consistent with how published numbers were produced, even
    though our own masks are arguably more complete on those frames.
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


def list_frame_pairs(dataset_dir: Path, limit: int = None) -> List[FramePair]:
    """Pairs left frames with self-computed binary masks for one sequence.

    Args:
        dataset_dir: Path to an instrument_dataset_N directory containing
            left_frames/ and binary_masks/ subdirectories.
        limit: If given, only pair the first `limit` frames (by name),
            so masks beyond that point don't need to exist.

    Returns:
        A list of (frame_path, mask_path) tuples, sorted by frame name.

    Raises:
        FileNotFoundError: If a frame's binary mask is missing.
    """
    frames_dir = dataset_dir / "left_frames"
    masks_dir = dataset_dir / "binary_masks"
    frame_paths = sorted(frames_dir.glob("frame*.png"))
    if limit is not None:
        frame_paths = frame_paths[:limit]

    pairs = []
    for frame_path in frame_paths:
        mask_path = masks_dir / frame_path.name
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing binary mask for {frame_path}; run "
                "scripts/make_binary_masks.py first."
            )
        pairs.append((frame_path, mask_path))
    return pairs


def official_mask_path(
    test_root: Path, sequence: int, frame_name: str
) -> Path:
    """Resolves the official BinarySegmentation mask path for one frame.

    instrument_dataset_1's test masks sit directly under
    instrument_dataset_1/BinarySegmentation/, while every other sequence
    nests them under .../ground_truth/BinarySegmentation/ -- the HF
    mirror's folder layout is inconsistent between the two.
    """
    seq_dir = test_root / f"instrument_dataset_{sequence}"
    direct = seq_dir / "BinarySegmentation" / frame_name
    if direct.exists():
        return direct
    return seq_dir / "ground_truth" / "BinarySegmentation" / frame_name


def train_pairs(
    dataset_root: Path, train_frames: int = 225
) -> List[FramePair]:
    """Training pairs for instrument_dataset_1..8.

    Uses each sequence's first `train_frames` frames, scored against our
    self-computed binary_masks/.

    Args:
        dataset_root: Directory containing instrument_dataset_N folders
            (e.g. dataset/training).
        train_frames: Number of leading frames per sequence to use.

    Returns:
        A flat list of (frame_path, mask_path) tuples across all 8
        sequences.
    """
    pairs: List[FramePair] = []
    for n in TRAIN_SEQUENCES:
        dataset_dir = dataset_root / f"instrument_dataset_{n}"
        pairs.extend(list_frame_pairs(dataset_dir, limit=train_frames))
    return pairs


def test_pairs(
    dataset_root: Path,
    test_root: Path = None,
    train_frames: int = 225,
) -> List[FramePair]:
    """Official test pairs for instrument_dataset_1..8.

    Uses each sequence's frames from `train_frames` onward (the last 75
    of each 300-frame sequence), scored against the official
    BinarySegmentation masks downloaded by
    scripts/download_test_masks.py. Frame images are read from
    `dataset_root` rather than `test_root`, since the HF mirror's test/
    images are byte-identical to the ones already under
    dataset/training/ -- only test/'s ground truth is new.

    Args:
        dataset_root: Directory containing instrument_dataset_N folders
            (e.g. dataset/training).
        test_root: Directory containing the downloaded official test
            masks (e.g. dataset/test). Defaults to a `test` sibling of
            `dataset_root`.
        train_frames: Number of leading frames per sequence reserved
            for training; frames from this index onward are test.

    Returns:
        A flat list of (frame_path, mask_path) tuples across all 8
        sequences.

    Raises:
        FileNotFoundError: If an official test mask is missing.
    """
    if test_root is None:
        test_root = dataset_root.parent / "test"

    pairs: List[FramePair] = []
    for n in TRAIN_SEQUENCES:
        frames_dir = dataset_root / f"instrument_dataset_{n}" / "left_frames"
        frame_paths = sorted(frames_dir.glob("frame*.png"))[train_frames:]
        for frame_path in frame_paths:
            mask_path = official_mask_path(test_root, n, frame_path.name)
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing official test mask for {frame_path}; run "
                    "scripts/download_test_masks.py first."
                )
            pairs.append((frame_path, mask_path))
    return pairs


def train_test_split(
    dataset_root: Path,
    test_root: Path = None,
    train_frames: int = 225,
) -> Tuple[List[FramePair], List[FramePair]]:
    """Convenience wrapper combining train_pairs() and test_pairs()."""
    return (
        train_pairs(dataset_root, train_frames),
        test_pairs(dataset_root, test_root, train_frames),
    )


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
