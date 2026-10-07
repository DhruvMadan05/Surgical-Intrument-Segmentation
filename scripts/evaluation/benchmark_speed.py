"""Benchmarks per-frame processing time of the baseline and the U-Net.

Times how long each method takes to turn one frame into a full-resolution
(1280x1024) binary mask and reports the mean latency and the equivalent
processing rate in Hz (frames per second = 1 / mean latency). Results are
saved to results/metrics/speed_benchmark.json; make the table and figure
with scripts/analysis/plot_speed_benchmark.py.

Benchmarks (each is timed on the same evenly spaced sample of test
frames):
  - otsu_baseline_cpu: the classical baseline. It is CPU-only.
  - unet_<device>: the U-Net end to end, on every available device (cpu,
    plus mps/cuda if present). Includes everything evaluation does:
    decoding the 1920x1080 PNG, cropping, resizing, normalizing, the
    network forward pass, and upsampling the mask back to 1280x1024.
  - unet_<device>_forward: only the network forward pass on an already
    preprocessed tensor, to show how much of the time is the network
    versus image I/O and pre/post-processing.

Method: each benchmark first runs a few untimed warmup calls (the first
GPU calls compile kernels and are much slower), then times each frame
individually with time.perf_counter(). Frame files are read once before
timing so disk-cache state doesn't skew the first measurements. GPU
timings synchronize the device before stopping the clock, since GPU work
is asynchronous.

For a stable measurement, close other heavy programs while this runs.
"""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

from segmentation import paths
from segmentation.baseline import predict_mask
from segmentation.dataset import InstrumentSegDataset, all_sequences_test_pairs
from segmentation.model import frame_to_tensor, load_model, make_predict_fn


def available_devices() -> List[torch.device]:
    """CPU plus any accelerator PyTorch can use on this machine."""
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def synchronize(device: torch.device) -> None:
    """Blocks until all queued work on `device` has finished.

    GPU kernels launch asynchronously, so without this the clock would
    stop before the work is actually done.
    """
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def sample_frames(
    pairs_by_sequence: Dict[int, list], num_frames: int
) -> List[Path]:
    """Picks `num_frames` evenly spaced frame paths across all test frames.

    Evenly spaced (rather than the first N) so the sample covers every
    sequence and its lighting/instrument variety. Deterministic.
    """
    frames = [
        frame
        for n in sorted(pairs_by_sequence)
        for frame, _ in pairs_by_sequence[n]
    ]
    step = max(1, len(frames) // num_frames)
    return frames[::step][:num_frames]


def time_calls(
    fn: Callable, items: list, warmup: int, sync_device=None
) -> np.ndarray:
    """Times fn(item) for each item; returns seconds per call.

    Args:
        fn: The function being benchmarked.
        items: One input per timed call.
        warmup: Number of untimed calls (on the first items) made first.
        sync_device: If given, synchronized before each clock reading.
    """

    def sync() -> None:
        if sync_device is not None:
            synchronize(sync_device)

    for item in items[:warmup]:
        fn(item)
    sync()

    seconds = []
    for item in items:
        sync()
        start = time.perf_counter()
        fn(item)
        sync()
        seconds.append(time.perf_counter() - start)
    return np.array(seconds)


def summarize(seconds: np.ndarray) -> Dict[str, float]:
    """Latency statistics (ms) and processing rate (Hz) for timed calls.

    The rate is 1 / mean latency, i.e. sustained throughput when frames
    are processed back to back, rather than the mean of per-frame rates
    (which would overweight unusually fast frames).
    """
    ms = seconds * 1000.0
    return {
        "mean_ms": float(ms.mean()),
        "std_ms": float(ms.std()),
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "hz": float(1.0 / seconds.mean()),
        "num_frames": int(len(seconds)),
    }


def run_benchmark(
    name: str,
    description: str,
    device: str,
    fn: Callable,
    items: list,
    warmup: int,
    sync_device=None,
) -> Dict:
    """Runs one benchmark, prints a one-line summary, returns its record."""
    stats = summarize(time_calls(fn, items, warmup, sync_device))
    print(
        f"{name:<24} {stats['mean_ms']:8.1f} ms/frame "
        f"(median {stats['median_ms']:.1f}, p95 {stats['p95_ms']:.1f})  "
        f"-> {stats['hz']:.2f} Hz"
    )
    return {"description": description, "device": device, **stats}


def main() -> None:
    """Parses CLI arguments, runs all benchmarks and saves the JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=paths.DATASET_ROOT
    )
    parser.add_argument("--test-root", type=Path, default=None)
    parser.add_argument(
        "--checkpoint", type=Path, default=paths.DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of test frames to time per benchmark (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Untimed warmup calls before timing (default: 10)",
    )
    parser.add_argument("--out", type=Path, default=paths.SPEED_METRICS)
    args = parser.parse_args()

    frames = sample_frames(
        all_sequences_test_pairs(args.dataset_root, args.test_root),
        args.num_frames,
    )
    # Pull every file into the OS page cache so the first timed frames
    # aren't penalized by cold disk reads.
    for frame in frames:
        frame.read_bytes()
    print(f"Timing {len(frames)} frames, {args.warmup} warmup calls each\n")

    size = InstrumentSegDataset.DEFAULT_SIZE
    benchmarks: Dict[str, Dict] = {}

    benchmarks["otsu_baseline_cpu"] = run_benchmark(
        "otsu_baseline_cpu",
        "Otsu saturation threshold, end to end (PNG decode to 1280x1024 mask)",
        "cpu",
        predict_mask,
        frames,
        args.warmup,
    )

    for device in available_devices():
        model = load_model(args.checkpoint, device)

        name = f"unet_{device.type}"
        benchmarks[name] = run_benchmark(
            name,
            "U-Net end to end (PNG decode, preprocess, forward, upsample)",
            device.type,
            make_predict_fn(model, device, size),
            frames,
            args.warmup,
            sync_device=device,
        )

        # Forward pass alone: preprocess outside the timed region.
        inputs = [frame_to_tensor(f, size)[1].to(device) for f in frames]

        def forward(tensor: torch.Tensor) -> torch.Tensor:
            """One network forward pass, no gradient tracking."""
            with torch.no_grad():
                return model(tensor)

        name = f"unet_{device.type}_forward"
        benchmarks[name] = run_benchmark(
            name,
            f"U-Net forward pass only, {size[0]}x{size[1]} input",
            device.type,
            forward,
            inputs,
            args.warmup,
            sync_device=device,
        )

    summary = {
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "settings": {
            "num_frames": len(frames),
            "warmup": args.warmup,
            "model_input_size": list(size),
            "checkpoint": str(args.checkpoint),
        },
        "benchmarks": benchmarks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
