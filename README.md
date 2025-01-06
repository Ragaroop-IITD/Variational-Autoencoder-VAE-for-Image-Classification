# Variational Autoencoder (VAE) for Image Classification

## Introduction

This project focuses on learning useful representations from *unlabelled* data for downstream tasks, specifically categorizing images into one of N categories using Variational Autoencoders (VAE). The goal is to extract meaningful representations from unlabelled images and then use these representations for classification.

In this assignment, we explore how to learn meaningful representations from unlabelled image data, which can then be applied to classify images with minimal labelled examples. Our main focus is on using a Variational Autoencoder (VAE) to learn a structured latent space and a Gaussian Mixture Model (GMM) for clustering and classification.

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

![image](https://github.com/user-attachments/assets/020adc52-d452-4d94-ba9a-f84bb28fe541)

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

## Validation Dataset Reconstruction
The images reconstructed for the validation set are shown below along with the SSIM score.
- The VAE shows good reconstruction for simpler digits like "1," with high SSIM scores (up to 0.923), but struggles with more complex digits like "4" and "8," where SSIM scores drop to as low as 0.360.
- The average SSIM across all images is 0.694, indicating decent but imperfect reconstruction quality.

![image](https://github.com/user-attachments/assets/d97566a0-eb87-4d6e-b854-00eeeffb0aeb)


## Generative Performance of the VAE
This visualization shows a 2D manifold learned by the VAE, mapping digits "1," "4," and "8" across a continuous latent space. The image illustrates smooth transitions between similar digits and some degree of blending between different classes.
- The arrangement highlights the VAE's ability to capture features that connect these digits, suggesting that the model can represent distinct digit styles on a continuum.
- However, there’s some distortion in transitioning between distinct shapes, indicating that a 2D latent space might be limiting for capturing complex variations within each digit. Increasing the latent dimensionality could potentially improve this representation by allowing for a more nuanced encoding of individual digit characteristics.

![image](https://github.com/user-attachments/assets/5148127c-b904-48fd-b564-0eafa0d97374)

## Visualization of the Latent Space
- **Effectiveness of VAE:** The presence of distinct clusters suggests that the VAE has successfully organized the latent space in a way that captures the underlying structure of the data. Each digit class occupies a distinct region in the latent space, which is a desirable property for classification tasks.
- **Quality of Clustering:** While the clusters are not perfectly separated, the separation is good enough to indicate that the VAE has learned meaningful representations of the digits. The overlap between clusters can be due to the inherent variability in the data or the complexity of the digit classes.

![image](https://github.com/user-attachments/assets/f7e76888-00ee-47d3-afd4-235fcabe2fd2)

## Visualization of the GMM
- **Effectiveness of GMM:** The presence of distinct ellipses for each digit class suggests that the GMM has effectively captured the underlying structure of the latent space. Each digit class occupies a distinct region in the latent space, which is a desirable property for classification tasks.
- **Separation of Clusters:** The ellipses are relatively well-separated, indicating that the GMM has learned to distinguish between the different digit classes in the latent space. However, there is some overlap between the clusters, particularly between the teal and yellow clusters (digits 4 and 8), which suggests that there might be some misclassification in these regions.

![image](https://github.com/user-attachments/assets/5a16ca47-e71b-4b5d-a37b-c4559b20ec27)



## References
- [Jeremy Jordan's VAE Tutorial](https://www.jeremyjordan.me/variational-autoencoders/)
- [Understanding VAEs](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)
- [Original VAE Paper](https://arxiv.org/abs/1312.6114)
