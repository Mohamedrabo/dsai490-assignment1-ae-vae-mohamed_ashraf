"""
test_model.py
-------------
Unit tests for src/model.py.
Run with: pytest tests/test_model.py
"""

import numpy as np
import pytest
import tensorflow as tf

from src.model import VAE, Sampling, build_ae, build_decoder, build_encoder, build_vae

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE   = 64
LATENT_DIM = 16
BATCH      = 4


# ---------------------------------------------------------------------------
# build_encoder
# ---------------------------------------------------------------------------

def test_encoder_ae_output_shape():
    """AE encoder must produce a single latent vector of shape (B, LATENT_DIM)."""
    enc = build_encoder(IMG_SIZE, LATENT_DIM, variational=False, name_prefix="test")
    x   = tf.zeros((BATCH, IMG_SIZE, IMG_SIZE, 1))
    z   = enc(x)
    assert z.shape == (BATCH, LATENT_DIM)


def test_encoder_vae_output_shapes():
    """VAE encoder must produce [z_mean, z_log_var] both of shape (B, LATENT_DIM)."""
    enc         = build_encoder(IMG_SIZE, LATENT_DIM, variational=True, name_prefix="test")
    x           = tf.zeros((BATCH, IMG_SIZE, IMG_SIZE, 1))
    z_mean, z_lv = enc(x)
    assert z_mean.shape == (BATCH, LATENT_DIM)
    assert z_lv.shape   == (BATCH, LATENT_DIM)


# ---------------------------------------------------------------------------
# build_decoder
# ---------------------------------------------------------------------------

def test_decoder_output_shape():
    """Decoder must reconstruct images of shape (B, IMG_SIZE, IMG_SIZE, 1)."""
    dec = build_decoder(IMG_SIZE, LATENT_DIM, name_prefix="test")
    z   = tf.zeros((BATCH, LATENT_DIM))
    out = dec(z)
    assert out.shape == (BATCH, IMG_SIZE, IMG_SIZE, 1)


def test_decoder_output_range():
    """Decoder output must lie in [0, 1] (sigmoid activation)."""
    dec = build_decoder(IMG_SIZE, LATENT_DIM, name_prefix="test")
    z   = tf.random.normal((BATCH, LATENT_DIM))
    out = dec(z)
    assert float(tf.reduce_min(out)) >= 0.0
    assert float(tf.reduce_max(out)) <= 1.0


# ---------------------------------------------------------------------------
# Sampling layer
# ---------------------------------------------------------------------------

def test_sampling_output_shape():
    """Sampling layer must return tensor with same shape as z_mean."""
    layer  = Sampling()
    mu     = tf.zeros((BATCH, LATENT_DIM))
    log_var = tf.zeros((BATCH, LATENT_DIM))
    z      = layer([mu, log_var])
    assert z.shape == (BATCH, LATENT_DIM)


def test_sampling_is_stochastic():
    """Two separate samples from the same (mu, log_var) must differ."""
    layer   = Sampling()
    mu      = tf.zeros((BATCH, LATENT_DIM))
    log_var = tf.zeros((BATCH, LATENT_DIM))
    z1 = layer([mu, log_var]).numpy()
    z2 = layer([mu, log_var]).numpy()
    assert not np.allclose(z1, z2)


# ---------------------------------------------------------------------------
# build_ae
# ---------------------------------------------------------------------------

def test_build_ae_forward_pass():
    """AE end-to-end forward pass must produce correct output shape."""
    ae, _, _ = build_ae("test", IMG_SIZE, LATENT_DIM)
    x   = tf.zeros((BATCH, IMG_SIZE, IMG_SIZE, 1))
    out = ae(x)
    assert out.shape == (BATCH, IMG_SIZE, IMG_SIZE, 1)


def test_build_ae_is_compiled():
    """AE must have an optimizer attached after build_ae()."""
    ae, _, _ = build_ae("test", IMG_SIZE, LATENT_DIM)
    assert ae.optimizer is not None


# ---------------------------------------------------------------------------
# build_vae
# ---------------------------------------------------------------------------

def test_build_vae_forward_pass():
    """VAE end-to-end forward pass must produce correct output shape."""
    vae, _, _ = build_vae("test", IMG_SIZE, LATENT_DIM)
    x   = tf.zeros((BATCH, IMG_SIZE, IMG_SIZE, 1))
    out = vae(x)
    assert out.shape == (BATCH, IMG_SIZE, IMG_SIZE, 1)


def test_build_vae_is_compiled():
    """VAE must have an optimizer attached after build_vae()."""
    vae, _, _ = build_vae("test", IMG_SIZE, LATENT_DIM)
    assert vae.optimizer is not None


def test_vae_train_step_returns_losses():
    """A single VAE train step must return total_loss, recon_loss, and kl_loss."""
    vae, _, _ = build_vae("test", IMG_SIZE, LATENT_DIM)
    x_batch   = tf.random.uniform((BATCH, IMG_SIZE, IMG_SIZE, 1))
    losses    = vae.train_step((x_batch, x_batch))
    for key in ("total_loss", "recon_loss", "kl_loss"):
        assert key in losses
        assert losses[key].numpy() >= 0.0
