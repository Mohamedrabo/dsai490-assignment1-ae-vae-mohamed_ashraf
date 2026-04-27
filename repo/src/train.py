"""
train.py
--------
Training pipeline for AE and VAE models over all anatomical regions.

Run this script directly to train all models and save them to ``models/``.
Alternatively, import ``train_all_regions`` for use inside a notebook.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import tensorflow as tf

from src.data_processing import build_region_splits, make_region_ds
from src.model import build_ae, build_vae

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------
EPOCHS: int = 15
PATIENCE: int = 3
MODELS_DIR: str = "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_callbacks(patience: int = PATIENCE) -> List[tf.keras.callbacks.Callback]:
    """Return a list of standard training callbacks.

    Args:
        patience: Number of epochs with no improvement before early stopping.

    Returns:
        A list containing an ``EarlyStopping`` callback.
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        )
    ]


def _save_models(region: str, ae: tf.keras.Model, vae: tf.keras.Model, out_dir: str = MODELS_DIR) -> None:
    """Persist trained AE and VAE weights to disk.

    Args:
        region:  Anatomical region name used to construct the file name.
        ae:      Trained AE model.
        vae:     Trained VAE model.
        out_dir: Directory where ``.keras`` files will be written.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ae.save(os.path.join(out_dir, f"ae_{region}_v1.keras"))
    vae.save(os.path.join(out_dir, f"vae_{region}_v1.keras"))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_all_regions(
    data_dir: str,
    regions: List[str],
    epochs: int = EPOCHS,
    save: bool = True,
) -> Dict[str, dict]:
    """Train one AE and one VAE for every anatomical region.

    Args:
        data_dir: Root directory containing per-region image sub-folders.
        regions:  Ordered list of region names to train on.
        epochs:   Maximum number of training epochs (early stopping may halt sooner).
        save:     If ``True``, save trained models to ``models/``.

    Returns:
        A dict mapping ``region_name`` to a sub-dict with keys:
        ``ae``, ``vae``, ``ae_enc``, ``vae_enc``, ``vae_dec``,
        ``ae_hist``, ``vae_hist``, ``val_paths``.
    """
    splits  = build_region_splits(data_dir, regions)
    results = {}
    cbs     = _get_callbacks()

    for region in regions:
        print(f"\n{'='*55}")
        print(f"  Region: {region}")
        print(f"{'='*55}")

        tr_ds = make_region_ds(splits[region]["train"], training=True)
        va_ds = make_region_ds(splits[region]["val"],   training=False)

        # ── Autoencoder ──────────────────────────────────────────
        print("  [AE] Training …")
        ae, ae_enc, ae_dec = build_ae(region)
        ae_hist = ae.fit(
            tr_ds,
            validation_data=va_ds,
            epochs=epochs,
            callbacks=cbs,
            verbose=1,
        )

        # ── Variational Autoencoder ───────────────────────────────
        print("  [VAE] Training …")
        vae, vae_enc, vae_dec = build_vae(region)
        vae_hist = vae.fit(
            tr_ds,
            validation_data=va_ds,
            epochs=epochs,
            callbacks=cbs,
            verbose=1,
        )

        if save:
            _save_models(region, ae, vae)

        results[region] = dict(
            ae=ae,       vae=vae,
            ae_enc=ae_enc, vae_enc=vae_enc, vae_dec=vae_dec,
            ae_hist=ae_hist, vae_hist=vae_hist,
            val_paths=splits[region]["val"],
        )

    print("\n✓ All regions trained.")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train AE + VAE on Medical MNIST")
    parser.add_argument("--data_dir", required=True, help="Path to Medical MNIST root directory")
    parser.add_argument("--epochs",   type=int, default=EPOCHS, help="Max training epochs")
    parser.add_argument("--no_save",  action="store_true", help="Skip saving models to disk")
    args = parser.parse_args()

    region_names = sorted([
        e.name for e in os.scandir(args.data_dir) if e.is_dir()
    ])
    print("Regions found:", region_names)

    train_all_regions(
        data_dir=args.data_dir,
        regions=region_names,
        epochs=args.epochs,
        save=not args.no_save,
    )
