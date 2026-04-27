"""
model.py
--------
AE and VAE model definitions for the Medical MNIST assignment.

Provides factory functions ``build_ae`` and ``build_vae`` that return
fully-compiled Keras models ready for training.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

# ---------------------------------------------------------------------------
# Hyper-parameters (can be overridden by passing explicit arguments)
# ---------------------------------------------------------------------------
IMG_SIZE: int = 64
LATENT_DIM: int = 16


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def build_encoder(
    img_size: int = IMG_SIZE,
    latent_dim: int = LATENT_DIM,
    variational: bool = False,
    name_prefix: str = "",
) -> tf.keras.Model:
    """Convolutional encoder shared by both AE and VAE.

    Args:
        img_size:    Spatial size of the square input image (pixels).
        latent_dim:  Dimensionality of the latent code.
        variational: If ``True`` the encoder outputs ``(z_mean, z_log_var)``
                     for the VAE reparameterisation trick; otherwise it
                     outputs a single deterministic ``z`` vector.
        name_prefix: String prefix added to the Keras model name.

    Returns:
        A ``tf.keras.Model`` with input shape ``(img_size, img_size, 1)``.
    """
    inp = tf.keras.Input(shape=(img_size, img_size, 1))
    x = tf.keras.layers.Conv2D(32,  3, strides=2, activation="relu", padding="same")(inp)
    x = tf.keras.layers.Conv2D(64,  3, strides=2, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    if variational:
        z_mean    = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
        return tf.keras.Model(inp, [z_mean, z_log_var], name=f"{name_prefix}_vae_enc")

    z = tf.keras.layers.Dense(latent_dim, name="z")(x)
    return tf.keras.Model(inp, z, name=f"{name_prefix}_ae_enc")


def build_decoder(
    img_size: int = IMG_SIZE,
    latent_dim: int = LATENT_DIM,
    name_prefix: str = "",
) -> tf.keras.Model:
    """Convolutional decoder shared by both AE and VAE.

    Args:
        img_size:    Spatial size of the reconstructed output image (pixels).
        latent_dim:  Dimensionality of the latent code.
        name_prefix: String prefix added to the Keras model name.

    Returns:
        A ``tf.keras.Model`` mapping ``(latent_dim,)`` → ``(img_size, img_size, 1)``.
    """
    s = img_size // 8
    inp = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(s * s * 128, activation="relu")(inp)
    x = tf.keras.layers.Reshape((s, s, 128))(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(64,  3, strides=2, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32,  3, strides=2, activation="relu", padding="same")(x)
    out = tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
    return tf.keras.Model(inp, out, name=f"{name_prefix}_dec")


# ---------------------------------------------------------------------------
# Sampling layer
# ---------------------------------------------------------------------------

class Sampling(tf.keras.layers.Layer):
    """Reparameterisation trick: sample z ~ N(mu, exp(0.5 * log_var)).

    Inputs:
        A list ``[z_mean, z_log_var]``, both of shape ``(batch, latent_dim)``.

    Returns:
        Sampled latent vector of shape ``(batch, latent_dim)``.
    """

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:  # noqa: D102
        mu, log_var = inputs
        return mu + tf.exp(0.5 * log_var) * tf.random.normal(tf.shape(mu))


# ---------------------------------------------------------------------------
# VAE model class
# ---------------------------------------------------------------------------

class VAE(tf.keras.Model):
    """Variational Autoencoder with combined reconstruction + KL loss.

    Args:
        enc:  Encoder model that outputs ``[z_mean, z_log_var]``.
        dec:  Decoder model that reconstructs the input image.
        **kw: Additional keyword arguments forwarded to ``tf.keras.Model``.
    """

    def __init__(self, enc: tf.keras.Model, dec: tf.keras.Model, **kw) -> None:
        super().__init__(**kw)
        self.enc    = enc
        self.dec    = dec
        self.sample = Sampling()
        self.t_loss = tf.keras.metrics.Mean("total_loss")
        self.r_loss = tf.keras.metrics.Mean("recon_loss")
        self.k_loss = tf.keras.metrics.Mean("kl_loss")

    @property
    def metrics(self) -> list:  # noqa: D102
        return [self.t_loss, self.r_loss, self.k_loss]

    def call(self, x: tf.Tensor) -> tf.Tensor:  # noqa: D102
        mu, log_var = self.enc(x)
        return self.dec(self.sample([mu, log_var]))

    def _compute(
        self, x: tf.Tensor, training: bool
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass + loss computation.

        Returns:
            Tuple of ``(total_loss, reconstruction_loss, kl_loss)``.
        """
        mu, log_var = self.enc(x, training=training)
        x_hat = self.dec(self.sample([mu, log_var]), training=training)
        recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=[1, 2, 3]))
        kl    = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
        return recon + kl, recon, kl

    def train_step(self, data):  # noqa: D102
        x, _ = data
        with tf.GradientTape() as tape:
            loss, recon, kl = self._compute(x, training=True)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.t_loss.update_state(loss)
        self.r_loss.update_state(recon)
        self.k_loss.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):  # noqa: D102
        x, _ = data
        loss, recon, kl = self._compute(x, training=False)
        self.t_loss.update_state(loss)
        self.r_loss.update_state(recon)
        self.k_loss.update_state(kl)
        return {m.name: m.result() for m in self.metrics}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_ae(
    region: str,
    img_size: int = IMG_SIZE,
    latent_dim: int = LATENT_DIM,
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Construct and compile a standard Autoencoder.

    Args:
        region:     Anatomical region name used as a naming prefix.
        img_size:   Spatial size of the square input image.
        latent_dim: Dimensionality of the bottleneck layer.

    Returns:
        Tuple ``(ae_model, encoder, decoder)`` — all compiled Keras models.
    """
    enc = build_encoder(img_size, latent_dim, variational=False, name_prefix=region)
    dec = build_decoder(img_size, latent_dim, name_prefix=region)
    inp = tf.keras.Input(shape=(img_size, img_size, 1))
    ae  = tf.keras.Model(inp, dec(enc(inp)), name=f"AE_{region}")
    ae.compile(optimizer="adam", loss="mse")
    return ae, enc, dec


def build_vae(
    region: str,
    img_size: int = IMG_SIZE,
    latent_dim: int = LATENT_DIM,
) -> Tuple[VAE, tf.keras.Model, tf.keras.Model]:
    """Construct and compile a Variational Autoencoder.

    Args:
        region:     Anatomical region name used as a naming prefix.
        img_size:   Spatial size of the square input image.
        latent_dim: Dimensionality of the latent distribution.

    Returns:
        Tuple ``(vae_model, encoder, decoder)`` — all compiled Keras models.
    """
    enc = build_encoder(img_size, latent_dim, variational=True, name_prefix=region)
    dec = build_decoder(img_size, latent_dim, name_prefix=region)
    vae = VAE(enc, dec, name=f"VAE_{region}")
    vae.compile(optimizer="adam")
    return vae, enc, dec
