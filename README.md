
# Variational Autoencoder (VAE) for Image Classification

## Introduction

This project focuses on learning useful representations from *unlabelled* data for downstream tasks, specifically categorizing images into one of N categories using Variational Autoencoders (VAE). The goal is to extract meaningful representations from unlabelled images and then use these representations for classification.

## Problem Description

In this assignment, you will:
1. Train a Variational Autoencoder (VAE) to extract useful features from unlabelled images.
2. Use these features to classify images of digits (1, 4, or 8) with minimal examples using a Gaussian Mixture Model (GMM).

## Implementation Details

### Component I: Training Variational Autoencoder

#### VAE Basics
- VAEs impose a normal distribution on the latent space, enabling smooth interpolation and new sample generation.
- The VAE loss includes a reconstruction loss term and a KL divergence term, ensuring the output image resembles the input and the latent distribution is regularized.

#### Architecture
- Use a Multilayer Perceptron (MLP) for the encoder and decoder.
- The encoder maps a $28 \times 28$ image to a 2-dimensional latent vector.

![image](https://github.com/user-attachments/assets/53c9ff16-141a-41bc-9e9d-f3e5b54ead9b)


#### Loss Function
```python
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

#### Evaluating Image Reconstruction
Visualize reconstructed images using:
```python
def show_reconstruction(model, val_loader, n=10):
    # Code to visualize reconstructed images
```

#### Generating New Images
Generate new images by sampling from the latent space:
```python
def plot_2d_manifold(vae, latent_dim=2, n=20, digit_size=28, device='cuda'):
    # Code to plot generated images
```

#### Visualization of Latent Space
Visualize the latent vectors in a 2D scatter plot to observe clustering patterns.

### Component II: Classification using Learnt Embedding Features

#### GMM Model
- Extract the latent vectors for each image.
- Implement a GMM to cluster these vectors into three groups representing different digit classes.
- Initialize cluster means using the validation dataset and tune hyperparameters.

#### Visualizing the GMM Model
Plot each Gaussian distribution as an ellipse to assess the separation of clusters.

#### Evaluation of the Learnt GMM
Evaluate the GMM performance using accuracy, precision, recall, and F1 score.

## Training and Evaluation Process

### Dataset
- Use the MNIST dataset focusing on digits 1, 4, and 8.
- The dataset is provided as NumPy arrays.

### Training Code
Run the VAE training and testing using the following commands:
```bash
# Training
python vae.py path_to_train_dataset path_to_val_dataset train vae.pth gmm_params.pkl

# Image reconstruction
python vae.py path_to_test_dataset_recon test_reconstruction vae.pth

# Classification
python vae.py path_to_test_dataset test_classifier vae.pth gmm_params.pkl
```

### Evaluation
- **Reconstruction:** Evaluate the VAE using Mean Square Error and Structural Similarity Index (SSIM).
- **Classification:** Evaluate using 'Macro' and 'Micro' F1 scores.


## References
- [Jeremy Jordan's VAE Tutorial](https://www.jeremyjordan.me/variational-autoencoders/)
- [Understanding VAEs](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)
- [Original VAE Paper](https://arxiv.org/abs/1312.6114)

