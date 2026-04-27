"""
test_data_processing.py
-----------------------
Unit tests for src/data_processing.py.
Run with: pytest tests/test_data_processing.py
"""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.data_processing import (
    build_region_splits,
    get_paths,
    make_region_ds,
    make_vis_ds,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Create a temporary Medical MNIST-style directory with dummy images."""
    region = "AbdomenCT"
    region_dir = tmp_path / region
    region_dir.mkdir()

    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
        img.save(str(region_dir / f"img_{i:04d}.jpg"))

    return str(tmp_path), region


# ---------------------------------------------------------------------------
# get_paths
# ---------------------------------------------------------------------------

def test_get_paths_returns_list(tmp_data_dir):
    """get_paths should return a non-empty list of strings."""
    data_dir, region = tmp_data_dir
    paths = get_paths(data_dir, region)
    assert isinstance(paths, list)
    assert len(paths) == 10
    assert all(isinstance(p, str) for p in paths)


# ---------------------------------------------------------------------------
# make_region_ds
# ---------------------------------------------------------------------------

def test_make_region_ds_output_shape(tmp_data_dir):
    """Dataset batches should have shape (batch, 64, 64, 1)."""
    data_dir, region = tmp_data_dir
    paths = get_paths(data_dir, region)
    ds = make_region_ds(paths, training=False, batch_size=4)
    batch_x, batch_y = next(iter(ds))
    assert batch_x.shape[1:] == (64, 64, 1)
    assert batch_y.shape[1:] == (64, 64, 1)


def test_make_region_ds_target_equals_input(tmp_data_dir):
    """Reconstruction dataset: target must equal input pixel-for-pixel."""
    data_dir, region = tmp_data_dir
    paths = get_paths(data_dir, region)
    ds = make_region_ds(paths, training=False, batch_size=4)
    x, y = next(iter(ds))
    np.testing.assert_array_equal(x.numpy(), y.numpy())


def test_make_region_ds_pixel_range(tmp_data_dir):
    """Pixel values must lie in [0, 1] after preprocessing."""
    data_dir, region = tmp_data_dir
    paths = get_paths(data_dir, region)
    ds = make_region_ds(paths, training=False, batch_size=4)
    x, _ = next(iter(ds))
    assert float(tf.reduce_min(x)) >= 0.0
    assert float(tf.reduce_max(x)) <= 1.0


# ---------------------------------------------------------------------------
# make_vis_ds
# ---------------------------------------------------------------------------

def test_make_vis_ds_output_shape(tmp_data_dir):
    """Vis dataset should yield images of shape (batch, 64, 64, 1)."""
    data_dir, region = tmp_data_dir
    paths = get_paths(data_dir, region)
    ds = make_vis_ds(paths, batch_size=4)
    imgs, _ = next(iter(ds))
    assert imgs.shape[1:] == (64, 64, 1)


# ---------------------------------------------------------------------------
# build_region_splits
# ---------------------------------------------------------------------------

def test_build_region_splits_ratio(tmp_data_dir):
    """Train split should contain ~90 % of images, val the remainder."""
    data_dir, region = tmp_data_dir
    splits = build_region_splits(data_dir, [region], train_ratio=0.9)
    total = len(splits[region]["train"]) + len(splits[region]["val"])
    assert total == 10
    assert len(splits[region]["train"]) == 9
    assert len(splits[region]["val"]) == 1


def test_build_region_splits_no_overlap(tmp_data_dir):
    """Train and validation splits must be disjoint."""
    data_dir, region = tmp_data_dir
    splits = build_region_splits(data_dir, [region])
    train_set = set(splits[region]["train"])
    val_set   = set(splits[region]["val"])
    assert train_set.isdisjoint(val_set)
