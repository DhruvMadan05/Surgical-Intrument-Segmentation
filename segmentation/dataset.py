"""Frame/mask pairing and train/test split for EndoVis 2017 sequences.

The test set for each sequence is whatever frame names are actually
present in the downloaded official test masks (dataset/test/, see
scripts/download_test_masks.py) -- not an assumed index cutoff. For
instrument_dataset_1..8 that happens to be frame225-frame299 (75
frames) and for 9/10 it's frame000-frame299 (all 300), verified against
the real files, but the split is derived from the files themselves so
it can't silently drift out of sync with them. Training uses every
frame in instrument_dataset_1..8 NOT in that sequence's test set.
Sequences 9 and 10 are never used for training at all; they're held
out entirely as a stronger generalization check.

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
from typing import Dict, List, Tuple

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

ALL_SEQUENCES = range(1, 11)

# 9 and 10 have no training portion at all in this project -- they're
# evaluated on all 300 frames each, not just a 75-frame tail.
HELD_OUT_SEQUENCES = (9, 10)

FramePair = Tuple[Path, Path]


def official_mask_dir(test_root: Path, sequence: int) -> Path:
    """Resolves the official BinarySegmentation mask directory for a sequence.

    instrument_dataset_1's test masks sit directly under
    instrument_dataset_1/BinarySegmentation/, while every other sequence
    nests them under .../ground_truth/BinarySegmentation/ -- the HF
    mirror's folder layout is inconsistent between the two.
    """
    seq_dir = test_root / f"instrument_dataset_{sequence}"
    direct = seq_dir / "BinarySegmentation"
    if direct.is_dir():
        return direct
    return seq_dir / "ground_truth" / "BinarySegmentation"


def official_mask_path(
    test_root: Path, sequence: int, frame_name: str
) -> Path:
    """Resolves the official BinarySegmentation mask path for one frame."""
    return official_mask_dir(test_root, sequence) / frame_name


def test_frame_names(test_root: Path, sequence: int) -> List[str]:
    """Lists frame filenames actually present in a sequence's test masks.

    This is the source of truth for the train/test split: whatever
    frame names exist here are test frames, and every other frame in
    that sequence's left_frames/ is a train frame. Verified against the
    real downloaded files to be frame225-frame299 for
    instrument_dataset_1..8 and frame000-frame299 (all 300) for 9/10,
    but reading the directory directly means the split can't silently
    drift out of sync if the dataset ever changes.

    Raises:
        FileNotFoundError: If the sequence's official test mask
            directory doesn't exist (download_test_masks.py not run).
    """
    mask_dir = official_mask_dir(test_root, sequence)
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Missing official test masks for instrument_dataset_"
            f"{sequence} at {mask_dir}; run "
            "scripts/download_test_masks.py first."
        )
    return sorted(p.name for p in mask_dir.glob("frame*.png"))


def train_pairs(dataset_root: Path, test_root: Path = None) -> List[FramePair]:
    """Training pairs for instrument_dataset_1..8.

    Uses every frame in each sequence NOT present in that sequence's
    official test set (see test_frame_names()), scored against our
    self-computed binary_masks/. Sequences 9 and 10 are never included.

    Args:
        dataset_root: Directory containing instrument_dataset_N folders
            (e.g. dataset/training).
        test_root: Directory containing the downloaded official test
            masks (e.g. dataset/test). Defaults to a `test` sibling of
            `dataset_root`. Only used to determine which frame names to
            exclude -- mask pixel content from test_root is never read
            here.

    Returns:
        A flat list of (frame_path, mask_path) tuples across all 8
        sequences.

    Raises:
        FileNotFoundError: If a train frame's binary mask is missing,
            or a sequence's official test masks haven't been
            downloaded (needed to know what to exclude).
    """
    if test_root is None:
        test_root = dataset_root.parent / "test"

    pairs: List[FramePair] = []
    for n in TRAIN_SEQUENCES:
        dataset_dir = dataset_root / f"instrument_dataset_{n}"
        excluded = set(test_frame_names(test_root, n))
        frames_dir = dataset_dir / "left_frames"
        masks_dir = dataset_dir / "binary_masks"
        for frame_path in sorted(frames_dir.glob("frame*.png")):
            if frame_path.name in excluded:
                continue
            mask_path = masks_dir / frame_path.name
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing binary mask for {frame_path}; run "
                    "scripts/make_binary_masks.py first."
                )
            pairs.append((frame_path, mask_path))
    return pairs


def test_pairs(dataset_root: Path, test_root: Path = None) -> List[FramePair]:
    """Official test pairs for instrument_dataset_1..8.

    Uses exactly the frame names present in each sequence's official
    test masks (see test_frame_names()), scored against those masks.
    Frame images are read from `dataset_root` rather than `test_root`,
    since the HF mirror's test/ images are byte-identical to the ones
    already under dataset/training/ -- only test/'s ground truth is
    new.

    Args:
        dataset_root: Directory containing instrument_dataset_N folders
            (e.g. dataset/training).
        test_root: Directory containing the downloaded official test
            masks (e.g. dataset/test). Defaults to a `test` sibling of
            `dataset_root`.

    Returns:
        A flat list of (frame_path, mask_path) tuples across all 8
        sequences.

    Raises:
        FileNotFoundError: If a test frame's image or official mask is
            missing.
    """
    if test_root is None:
        test_root = dataset_root.parent / "test"

    pairs: List[FramePair] = []
    for n in TRAIN_SEQUENCES:
        pairs.extend(_sequence_test_pairs(dataset_root, test_root, n))
    return pairs


def all_sequences_test_pairs(
    dataset_root: Path, test_root: Path = None
) -> Dict[int, List[FramePair]]:
    """Official test pairs for all 10 sequences, grouped by sequence.

    Sequences 1..8 use exactly the frame names in their official test
    masks (the same frames test_pairs() returns, just grouped per
    sequence instead of flattened). Sequences 9 and 10 have no training
    portion in this project, so all of their frames are test frames:
    evaluating on them is a stronger generalization check, since the
    model has never seen any frame from those two videos.

    This is independent of test_pairs(): it doesn't call it, so
    requesting only sequences 1..8 elsewhere never requires sequences
    9/10's masks to exist.

    Args:
        dataset_root: Directory containing instrument_dataset_N folders
            (e.g. dataset/training).
        test_root: Directory containing the downloaded official test
            masks (e.g. dataset/test). Defaults to a `test` sibling of
            `dataset_root`.

    Returns:
        Dict mapping sequence number (1-10) to its list of
        (frame_path, mask_path) tuples.

    Raises:
        FileNotFoundError: If a test frame's image or official mask is
            missing.
    """
    if test_root is None:
        test_root = dataset_root.parent / "test"

    return {
        n: _sequence_test_pairs(dataset_root, test_root, n)
        for n in ALL_SEQUENCES
    }


def _sequence_test_pairs(
    dataset_root: Path, test_root: Path, sequence: int
) -> List[FramePair]:
    """Test pairs for one sequence, driven by test_frame_names()."""
    frames_dir = (
        dataset_root / f"instrument_dataset_{sequence}" / "left_frames"
    )
    pairs: List[FramePair] = []
    for name in test_frame_names(test_root, sequence):
        frame_path = frames_dir / name
        if not frame_path.exists():
            raise FileNotFoundError(
                f"Missing frame image {frame_path} for a frame listed "
                f"in instrument_dataset_{sequence}'s official test set"
            )
        pairs.append(
            (frame_path, official_mask_path(test_root, sequence, name))
        )
    return pairs


def train_test_split(
    dataset_root: Path, test_root: Path = None
) -> Tuple[List[FramePair], List[FramePair]]:
    """Convenience wrapper combining train_pairs() and test_pairs()."""
    return (
        train_pairs(dataset_root, test_root),
        test_pairs(dataset_root, test_root),
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
