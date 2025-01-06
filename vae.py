import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm



# Set random seed
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetMNIST(Dataset):
    def __init__(self, dataset, keep_labels=[1, 4, 8]):
        self.data = []
        self.labels = []
        
        if isinstance(dataset, torch.Tensor):
            dataset = dataset.numpy()
        
        if isinstance(dataset, tuple):
            data, labels = dataset
            if labels is not None:
                for i, label in enumerate(labels):
                    if label in keep_labels:
                        self.data.append(data[i])
                        self.labels.append(label)
            else:
                self.data = data
                self.labels = [-1] * len(data)
        else:
            self.data = dataset
            self.labels = [-1] * len(dataset)
            
        self.data = torch.FloatTensor(self.data)
        if self.data.max() > 1.0:
            self.data = self.data / 255.0
            
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[2048, 1024, 512], latent_dim=2):
        super(VAE, self).__init__()
        
        # Enhanced Encoder
        encoder_layers = []
        in_features = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_features = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Enhanced Decoder
        decoder_layers = []
        hidden_dims = hidden_dims[::-1]  # Reverse for decoder
        
        in_features = latent_dim
        for hidden_dim in hidden_dims:
            decoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_features = hidden_dim
            
        decoder_layers.extend([
            nn.Linear(hidden_dims[-1], input_dim),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        # if self.training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        # return mu
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    

class GMM:
    def __init__(self, n_components=3, n_features=2):
        self.n_components = n_components
        self.n_features = n_features
        self.means = None
        self.covs = None
        self.weights = None
        self.label_map = {}
        
    def initialize_params(self, initial_means, labels):
        self.means = initial_means
        self.covs = np.array([np.eye(self.n_features) for _ in range(self.n_components)])
        self.weights = np.ones(self.n_components) / self.n_components
        unique_labels = sorted(list(set(labels)))
        self.label_map = {i: label for i, label in enumerate(unique_labels)}
        
    def gaussian_pdf(self, x, mean, cov):
        n = x.shape[1]
        diff = x - mean
        cov = cov + np.eye(n) * 1e-6
        return np.exp(-0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)) / \
               (np.sqrt((2 * np.pi) ** n * np.linalg.det(cov)))
    
    def expectation_step(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self.gaussian_pdf(X, self.means[k], self.covs[k])
        responsibilities += 1e-10
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def maximization_step(self, X, responsibilities):
        N = responsibilities.sum(axis=0)
        self.means = (responsibilities.T @ X) / N[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covs[k] = (responsibilities[:, k:k+1] * diff).T @ diff / N[k]
        self.weights = N / N.sum()
    
    def fit(self, X, n_iterations=100, tol=1e-6):
        prev_log_likelihood = float('-inf')
        for _ in range(n_iterations):
            responsibilities = self.expectation_step(X)
            self.maximization_step(X, responsibilities)
            log_likelihood = self.compute_log_likelihood(X)
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood
    
    def compute_log_likelihood(self, X):
        likelihoods = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights[k] * self.gaussian_pdf(X, self.means[k], self.covs[k])
        return np.sum(np.log(np.sum(likelihoods, axis=1)))
    
    def predict(self, X):
        responsibilities = self.expectation_step(X)
        cluster_labels = np.argmax(responsibilities, axis=1)
        return np.array([self.label_map[label] for label in cluster_labels])

def loss_function(recon_x, x, mu, logvar, kld_weight=0.1):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + kld_weight * KLD

def train_vae(train_loader, val_loader, model, optimizer, epochs=1000, arg4='vae.pth'):
    best_val_loss = float('inf')
    best_state_dict = None
    patience = 900
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                val_loss += loss_function(recon_batch, data, mu, logvar).item()
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader.dataset):.4f}, Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()
            torch.save(model.state_dict(), arg4)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

def extract_features(loader, model):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            _, mu, _ = model(data)
            features.append(mu.cpu().numpy())
            labels.append(label.numpy())
    return np.vstack(features), np.concatenate(labels)

def show_reconstruction(model, val_loader, n=15):
    model.eval()
    data, labels = next(iter(val_loader))
    data = data.to(device)
    recon_data, _, _ = model(data)
    
    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        axes[0, i].imshow(data[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_data[i].cpu().view(28, 28).detach().numpy(), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

def plot_2d_manifold(vae, latent_dim=2, n=20, digit_size=28, device='cuda'):
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    
    vae.eval()
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], device=device).float()
                decoded = vae.decode(z_sample)
                digit = decoded.view(digit_size, digit_size).cpu().numpy()
                figure[i * digit_size: (i + 1) * digit_size,
                      j * digit_size: (j + 1) * digit_size] = digit
    
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gnuplot2')
    plt.title('2D Manifold of VAE Latent Space')
    plt.axis('off')
    plt.show()





arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3] if len(sys.argv) > 3 else None
arg4 = sys.argv[4] if len(sys.argv) > 4 else None
arg5 = sys.argv[5] if len(sys.argv) > 5 else None

if arg3=='train':
    # Load datasets
    train_data = np.load(arg1)
    val_data = np.load(arg2)

    # Extract images and labels
    train_images = train_data['data']
    train_labels = train_data['labels']
    val_images = val_data['data']
    val_labels = val_data['labels']

    # Create datasets
    train_dataset = SubsetMNIST((train_images, train_labels) if train_labels is not None else train_images)
    val_dataset = SubsetMNIST((val_images, val_labels) if val_labels is not None else val_images)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize and train VAE
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Print model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")

    # Train VAE
    train_vae(train_loader, val_loader, model, optimizer,arg4=arg4)

    # Save VAE model
    torch.save(model.state_dict(), arg4)


    # Extract features for GMM
    val_features, val_labels = extract_features(val_loader, model)

    # Initialize and train GMM
    gmm = GMM(n_components=3, n_features=2)
    initial_means = []
    labels_for_mapping = []
    for label in [1, 4, 8]:
        mask = val_labels == label
        if np.any(mask):
            initial_means.append(val_features[mask].mean(axis=0))
            labels_for_mapping.append(label)

    gmm.initialize_params(np.array(initial_means), labels_for_mapping)
    train_features, _ = extract_features(train_loader, model)
    gmm.fit(train_features)

    # Save GMM
    with open(arg5, 'wb') as f:
        pickle.dump(gmm, f)


elif arg2=="test_reconstruction":
    # Load VAE and GMM models
    model = VAE().to(device)
    model.load_state_dict(torch.load(arg3))


    print("Model loaded successfully")

    val_data = np.load(arg1)

    val_images = val_data['data']
    val_labels = val_data['labels']

    # Create dataset
    val_dataset = SubsetMNIST((val_images, val_labels) if val_labels is not None else val_images)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    print("Reconstructing validation images....")

    # show_reconstruction(model, val_loader)

    model.eval()
    loader = iter(val_loader)

    recon_data = []

    for i in range(len(loader)):
        data, labels = next(loader)

        #     data = data.to(device)
        data = data.view(data.size(0), 784).to(device)
        recon_imgs, _, _ = model(data)

        bs = len(data)
        for i in range(bs):
            recon_img = recon_imgs[i].cpu().view(28, 28).detach().numpy()
            recon_data.append(recon_img)

    save_file = "vae_reconstructed.npz"
    np.savez_compressed(save_file, data=recon_data)




elif arg2=="test_classifier":
    # Load test data
    test_data = np.load(arg1)
    test_images = test_data['data']
    test_dataset = SubsetMNIST(test_images)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load VAE and GMM models
    model = VAE().to(device)
    model.load_state_dict(torch.load(arg3))

    with open(arg4, 'rb') as f:
        gmm = pickle.load(f)

    # Extract features and predict
    features, _ = extract_features(test_loader, model)
    predictions = gmm.predict(features)

    # Save predictions
    np.savetxt('vae.csv', predictions, header='Predicted_Label', delimiter=',', fmt='%d')


elif arg3=="plot_latent":
    # Load VAE and GMM models
    model = VAE().to(device)
    model.load_state_dict(torch.load(arg4))

    # Load datasets
    train_data = np.load(arg1)
    val_data = np.load(arg2)

    # Extract images and labels
    train_images = train_data['data']
    train_labels = train_data['labels']
    val_images = val_data['data']
    val_labels = val_data['labels']

    # Create datasets
    train_dataset = SubsetMNIST((train_images, train_labels) if train_labels is not None else train_images)
    val_dataset = SubsetMNIST((val_images, val_labels) if val_labels is not None else val_images)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    def plot_latent_space(loader, model, title="VAE Latent Space"):
        # Extract features and labels
        model.eval()
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for data, label in loader:
                data = data.to(device)
                # Get the mean vector (mu) from the VAE's encoder
                mu, _ = model.encode(data.view(-1, 784))
                latent_vectors.append(mu.cpu().numpy())
                labels.append(label.numpy())
        
        # Combine all batches
        latent_vectors = np.vstack(latent_vectors)
        labels = np.concatenate(labels)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6)
        
        # Add legend
        legend1 = plt.legend(*scatter.legend_elements(),
                            loc="upper right", title="Digits")
        # plt.add_artist(legend1)
        
        plt.title(title)
        plt.xlabel("First Latent Dimension")
        plt.ylabel("Second Latent Dimension")
        plt.grid(True, alpha=0.3)
        plt.show()

        return latent_vectors, labels

    # Add these lines after training the VAE and before the GMM part:
    print("Visualizing training data in latent space...")
    train_latent, train_labels = plot_latent_space(train_loader, model, "Training Data Latent Space")

    print("Visualizing validation data in latent space...")
    val_latent, val_labels = plot_latent_space(val_loader, model, "Validation Data Latent Space")




elif arg3=="vizualize_gmm":
    import matplotlib.patches as patches
    from matplotlib.patches import Ellipse
    import scipy

     # Load VAE and GMM models
    model = VAE().to(device)
    model.load_state_dict(torch.load(arg4))

    # Load datasets
    train_data = np.load(arg1)
    val_data = np.load(arg2)

    # Extract images and labels
    train_images = train_data['data']
    train_labels = train_data['labels']
    val_images = val_data['data']
    val_labels = val_data['labels']

    # Create datasets
    train_dataset = SubsetMNIST((train_images, train_labels) if train_labels is not None else train_images)
    val_dataset = SubsetMNIST((val_images, val_labels) if val_labels is not None else val_images)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    with open(arg5, 'rb') as f:
        gmm = pickle.load(f)

    def plot_latent_space_with_gmm(loader, model, gmm, title="VAE Latent Space with GMM"):
        # Extract features and labels
        model.eval()
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for data, label in loader:
                data = data.to(device)
                mu, _ = model.encode(data.view(-1, 784))
                latent_vectors.append(mu.cpu().numpy())
                labels.append(label.numpy())
        
        # Combine all batches
        latent_vectors = np.vstack(latent_vectors)
        labels = np.concatenate(labels)
        
        # Create the main scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6, label='Data points')
        
        # Plot GMM components
        colors = ['r', 'g', 'b']  # Different colors for each Gaussian
        for i in range(gmm.n_components):
            # Get the mean and covariance of the current component
            mean = gmm.means[i]
            covar = gmm.covs[i]
            
            # Calculate eigenvalues and eigenvectors of the covariance matrix
            eigenvals, eigenvecs = np.linalg.eigh(covar)
            
            # Calculate angle of the ellipse
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            
            # Calculate the width and height of the ellipse (using 2 standard deviations)
            width, height = 2 * np.sqrt(2) * np.sqrt(eigenvals)
            
            # Create confidence ellipses for 1, 2, and 3 standard deviations
            for n_std in [1]:
                ellipse = Ellipse(xy=mean, 
                                width=width*n_std/2, 
                                height=height*n_std/2,
                                angle=angle,
                                facecolor='none',
                                edgecolor=colors[i],
                                alpha=0.3,
                                linestyle='--',
                                label=f'Component {gmm.label_map[i]} - {n_std}σ' if n_std == 2 else "")
                plt.gca().add_patch(ellipse)
            
            # Plot the mean of each component
            plt.plot(mean[0], mean[1], colors[i]+'*', markersize=15, 
                    label=f'Mean of component {gmm.label_map[i]}')
        
        # Add legend and labels
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title)
        plt.xlabel("First Latent Dimension")
        plt.ylabel("Second Latent Dimension")
        plt.grid(True, alpha=0.3)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        plt.show()
        
        return latent_vectors, labels

    # Add these lines after training both VAE and GMM:
    print("Visualizing training data with GMM...")
    train_latent_gmm, train_labels_gmm = plot_latent_space_with_gmm(
        train_loader, model, gmm, "Training Data Latent Space with GMM Components")

    print("Visualizing validation data with GMM...")
    val_latent_gmm, val_labels_gmm = plot_latent_space_with_gmm(
        val_loader, model, gmm, "Validation Data Latent Space with GMM Components")
    

elif arg3=="performance":
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pandas as pd

    # Load VAE and GMM models
    model = VAE().to(device)
    model.load_state_dict(torch.load(arg4))

    # Load datasets
    train_data = np.load(arg1)
    val_data = np.load(arg2)

    # Extract images and labels
    train_images = train_data['data']
    train_labels = train_data['labels']
    val_images = val_data['data']
    val_labels = val_data['labels']

    # Create datasets
    train_dataset = SubsetMNIST((train_images, train_labels) if train_labels is not None else train_images)
    val_dataset = SubsetMNIST((val_images, val_labels) if val_labels is not None else val_images)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    with open(arg5, 'rb') as f:
        gmm = pickle.load(f)

    def evaluate_gmm_performance(labels_true, labels_pred):
        """
        Evaluate GMM clustering performance using various metrics
        """
        accuracy = accuracy_score(labels_true, labels_pred)
        precision_macro = precision_score(labels_true, labels_pred, average='macro')
        recall_macro = recall_score(labels_true, labels_pred, average='macro')
        f1_macro = f1_score(labels_true, labels_pred, average='macro')
        
        # Calculate per-class metrics
        precision_per_class = precision_score(labels_true, labels_pred, average=None)
        recall_per_class = recall_score(labels_true, labels_pred, average=None)
        f1_per_class = f1_score(labels_true, labels_pred, average=None)
        
        # Create detailed metrics dictionary
        metrics = {
            'Overall Metrics': {
                'Accuracy': accuracy,
                'Macro Precision': precision_macro,
                'Macro Recall': recall_macro,
                'Macro F1': f1_macro
            },
            'Per-Class Metrics': {
                'Class': [1, 4, 8],
                'Precision': precision_per_class,
                'Recall': recall_per_class,
                'F1-Score': f1_per_class
            }
        }
        
        return metrics

    def print_metrics(metrics, dataset_name):
        """
        Pretty print the evaluation metrics
        """
        print(f"\n=== {dataset_name} Performance Metrics ===")
        print("\nOverall Metrics:")
        for metric, value in metrics['Overall Metrics'].items():
            print(f"{metric}: {value:.4f}")
        
        print("\nPer-Class Metrics:")
        df = pd.DataFrame({
            'Class': metrics['Per-Class Metrics']['Class'],
            'Precision': metrics['Per-Class Metrics']['Precision'],
            'Recall': metrics['Per-Class Metrics']['Recall'],
            'F1-Score': metrics['Per-Class Metrics']['F1-Score']
        })
        print(df.to_string(index=False))

    # Add these lines after training the GMM model:

    # Evaluate on training data
    train_features, train_labels = extract_features(train_loader, model)
    train_predictions = gmm.predict(train_features)
    train_metrics = evaluate_gmm_performance(train_labels, train_predictions)
    print_metrics(train_metrics, "Training Data")

    # Evaluate on validation data
    val_features, val_labels = extract_features(val_loader, model)
    val_predictions = gmm.predict(val_features)
    val_metrics = evaluate_gmm_performance(val_labels, val_predictions)
    print_metrics(val_metrics, "Validation Data")

    # Create confusion matrices
    def plot_confusion_matrices(train_labels, train_predictions, val_labels, val_predictions):
        """
        Plot confusion matrices for both training and validation data
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training confusion matrix
        train_cm = confusion_matrix(train_labels, train_predictions)
        sns.heatmap(train_cm, annot=True, fmt='d', ax=ax1, 
                    xticklabels=[1, 4, 8], yticklabels=[1, 4, 8])
        ax1.set_title('Training Data Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Validation confusion matrix
        val_cm = confusion_matrix(val_labels, val_predictions)
        sns.heatmap(val_cm, annot=True, fmt='d', ax=ax2,
                    xticklabels=[1, 4, 8], yticklabels=[1, 4, 8])
        ax2.set_title('Validation Data Confusion Matrix')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.show()

    # Plot confusion matrices
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    plot_confusion_matrices(train_labels, train_predictions, val_labels, val_predictions)

    # Save metrics to file
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }

    with open('gmm_evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    from skimage.metrics import structural_similarity as ssim
    import numpy as np
    import torch

    def show_reconstruction_with_ssim(model, val_loader, n=15):
        """
        Show original and reconstructed images with their SSIM scores
        """
        model.eval()
        data, labels = next(iter(val_loader))
        data = data.to(device)
        recon_data, _, _ = model(data)
        
        fig, axes = plt.subplots(2, n, figsize=(15, 4))
        ssim_scores = []
        
        # Calculate SSIM for each image pair
        for i in range(n):
            # Get original and reconstructed images
            orig_img = data[i].cpu().numpy().squeeze()
            recon_img = recon_data[i].cpu().view(28, 28).detach().numpy()
            
            # Calculate SSIM
            ssim_score = ssim(orig_img, recon_img, data_range=1.0)
            ssim_scores.append(ssim_score)
            
            # Plot original image
            axes[0, i].imshow(orig_img, cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Original\nLabel: {labels[i].item()}')
            
            # Plot reconstructed image with SSIM score
            axes[1, i].imshow(recon_img, cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'Recon\nSSIM: {ssim_score:.3f}')
        
        plt.tight_layout()
        
        # Calculate and display average SSIM
        avg_ssim = np.mean(ssim_scores)
        plt.suptitle(f'Average SSIM: {avg_ssim:.3f}', y=1.05)
        plt.show()
        
        return avg_ssim, ssim_scores

    # Use this function instead of the original show_reconstruction
    print("Showing reconstructions with SSIM scores...")
    avg_ssim, individual_ssims = show_reconstruction_with_ssim(model, val_loader)

    # Print detailed SSIM statistics
    print(f"\nSSIM Statistics:")
    print(f"Average SSIM: {avg_ssim:.3f}")
    print(f"Min SSIM: {min(individual_ssims):.3f}")
    print(f"Max SSIM: {max(individual_ssims):.3f}")
    print(f"Std SSIM: {np.std(individual_ssims):.3f}")












