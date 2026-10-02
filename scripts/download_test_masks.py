"""Downloads the official EndoVis 2017 test-split binary masks from HF.

The HF mirror's `training/` folder already contains all 300 frames per
sequence (images are byte-identical to `test/left_frames`/`right_frames`,
verified by comparing blob hashes), so only `test/` is worth pulling
down: the pre-merged BinarySegmentation ground truth, which does not
exist anywhere under `training/` (that folder only has raw per-class
label folders). This script downloads just those mask files
(~40 MB total), skipping the ~4.2 GB of duplicate frame images.

Masks land at dataset/test/instrument_dataset_N/BinarySegmentation/ (or
.../ground_truth/BinarySegmentation/ for most sequences; the repo's
folder layout is inconsistent between instrument_dataset_1 and the
rest), mirroring the HF repo's own structure.
"""

import argparse

from huggingface_hub import snapshot_download

REPO_ID = "maxhallan7/robotic-instrument-segmentation-miccai-2017"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        default="dataset",
        help="Directory to download into (default: dataset/)",
    )
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=[
            "test/*/BinarySegmentation/*",
            "test/*/ground_truth/BinarySegmentation/*",
        ],
    )
    print(f"Downloaded test-split binary masks to {path}")


if __name__ == "__main__":
    main()
