"""
data_processing.py
------------------
tf.data pipeline utilities for the Medical MNIST dataset.

All image loading, preprocessing, and dataset construction
functions live here to keep the notebook clean and modular.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants (override via config dict if needed)
# ---------------------------------------------------------------------------
IMG_SIZE: int = 64
BATCH_SIZE: int = 256
AUTOTUNE: int = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def load_preprocess(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decode a JPEG/PNG file and normalise pixel values to [0, 1].

    Args:
        path:  Scalar string tensor — absolute path to the image file.
        label: Integer label tensor (unused for reconstruction; kept for API
               compatibility with ``from_tensor_slices``).

    Returns:
        A tuple ``(image, label)`` where *image* has shape
        ``(IMG_SIZE, IMG_SIZE, 1)`` and dtype ``float32``.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def get_paths(data_dir: str, region: str) -> List[str]:
    """Collect all image file paths for a single anatomical region.

    Args:
        data_dir: Root directory that contains per-region sub-folders.
        region:   Name of the anatomical region (sub-folder name).

    Returns:
        Sorted list of absolute path strings for every JPEG/PNG file found.
    """
    paths: List[str] = []
    for ext in ("*.jpeg", "*.jpg", "*.png"):
        paths += [str(p) for p in Path(data_dir, region).glob(ext)]
    return sorted(paths)


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def make_region_ds(
    paths: List[str],
    training: bool = True,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """Build a ``tf.data.Dataset`` that yields ``(image, image)`` pairs.

    Suitable for autoencoder training where the target equals the input.

    Args:
        paths:      List of image file paths for the region.
        training:   If ``True``, shuffle the dataset before batching.
        batch_size: Number of samples per batch.

    Returns:
        A batched, prefetched ``tf.data.Dataset``.
    """
    labels = [0] * len(paths)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img, _: (img, img))
    if training:
        ds = ds.shuffle(buffer_size=4096)
    ds = ds.cache().batch(batch_size).prefetch(AUTOTUNE)
    return ds


def make_vis_ds(
    paths: List[str],
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """Build a dataset that yields raw images (no reconstruction target).

    Used for latent-space visualisation where labels are not required.

    Args:
        paths:      List of image file paths.
        batch_size: Number of samples per batch.

    Returns:
        A batched, prefetched ``tf.data.Dataset`` yielding ``(image, label)``
        where *label* is always ``0``.
    """
    labels = [0] * len(paths)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.cache().batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Region split helper
# ---------------------------------------------------------------------------

def build_region_splits(
    data_dir: str,
    regions: List[str],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """Split each region's images into train/val subsets.

    Args:
        data_dir:    Root directory containing per-region sub-folders.
        regions:     Ordered list of region names to process.
        train_ratio: Fraction of images used for training (default 0.9).
        seed:        Random seed for reproducibility.

    Returns:
        A dict mapping ``region_name → {"train": [...], "val": [...]}``.
    """
    rng = np.random.RandomState(seed)
    splits: Dict[str, Dict[str, List[str]]] = {}

    for region in regions:
        all_paths = get_paths(data_dir, region)
        rng.shuffle(all_paths)
        cut = int(train_ratio * len(all_paths))
        splits[region] = {"train": all_paths[:cut], "val": all_paths[cut:]}

    return splits
