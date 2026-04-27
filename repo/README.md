# DSAI 490 — Assignment 1: Representation Learning with Autoencoders (AE & VAE)

## Overview

This project implements **Autoencoder (AE)** and **Variational Autoencoder (VAE)** models trained on the **Medical MNIST** dataset. One AE and one VAE are trained per anatomical region. The project covers data reconstruction, latent space visualization, sample generation, and denoising.

---

## Project Structure

```
├── data/
│   ├── raw/           # Original dataset (upload to Google Drive)
│   └── processed/     # Processed/cached data
├── models/            # Saved trained models (.keras files)
├── notebooks/
│   └── Mohamed_Ashraf_assignment1_GAI.ipynb   # Main experiment notebook
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # tf.data pipeline utilities
│   ├── model.py             # AE and VAE model definitions
│   └── train.py             # Training loop and callbacks
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
├── README.md
└── requirements.txt
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/dsai490-assignment1-ae-vae.git
cd dsai490-assignment1-ae-vae
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

- Upload the **Medical MNIST** dataset folder to your **Google Drive** at:
  `MyDrive/medical_mnist/archive`
- The dataset must **not** be in `.npz` or `.csv` format — use the original image folder structure.

### 5. Run the Notebook

Open `notebooks/Mohamed_Ashraf_assignment1_GAI.ipynb` in **Google Colab** (recommended for GPU access) or locally:

```bash
jupyter notebook notebooks/Mohamed_Ashraf_assignment1_GAI.ipynb
```

---

## Models

| Model | Architecture | Latent Dim | Loss |
|-------|-------------|------------|------|
| AE    | Conv Encoder + Conv Decoder | 16 | MSE |
| VAE   | Conv Encoder (μ, σ) + Sampling + Conv Decoder | 16 | MSE + KL Divergence |

---

## Key Results

- Reconstruction quality compared between AE and VAE per anatomical region
- Latent space visualized using PCA and t-SNE
- VAE generates new samples by sampling from the learned latent distribution
- Both models tested for denoising capability

---

## Requirements

See `requirements.txt` for all dependencies.

---

## Author

**Mohamed Ashraf**
DSAI 490 — Generative AI
