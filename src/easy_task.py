import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from vae_model import VAE


df = pd.read_csv("data/Final_Refined_Dataset.csv")
mfcc_cols = [f'mfcc_{i}' for i in range(13)] 
features = df[mfcc_cols].values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
X_tensor = torch.FloatTensor(features_scaled)


input_dim = 13
hidden_dim = 32
latent_dim = 2 
vae = VAE(input_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

print("Training VAE...")
for epoch in range(100):
    recon_batch, mu, logvar = vae(X_tensor)
    recon_loss = nn.functional.mse_loss(recon_batch, X_tensor)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + 0.01 * kld_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


with torch.no_grad():
    vae_latent, _ = vae.encoder(X_tensor)
    vae_latent = vae_latent.numpy()

pca = PCA(n_components=2)
pca_features = pca.fit_transform(features_scaled)


wcss = []
k_range = range(1, 15)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(vae_latent)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, 'bx-')
plt.title('Elbow Method for Optimal k')
plt.show()


optimal_k = 6 


vae_kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(vae_latent)
pca_kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(pca_features)


vae_silhouette = silhouette_score(vae_latent, vae_kmeans.labels_)
vae_ch_index = calinski_harabasz_score(vae_latent, vae_kmeans.labels_)

pca_silhouette = silhouette_score(pca_features, pca_kmeans.labels_)
pca_ch_index = calinski_harabasz_score(pca_features, pca_kmeans.labels_)

print(f"\nVAE Metrics: Silhouette={vae_silhouette:.3f}, CH Index={vae_ch_index:.1f}")
print(f"PCA Metrics: Silhouette={pca_silhouette:.3f}, CH Index={pca_ch_index:.1f}")


tsne = TSNE(n_components=2, random_state=42)
vae_tsne = tsne.fit_transform(vae_latent)

plt.figure(figsize=(10, 7))
plt.scatter(vae_tsne[:, 0], vae_tsne[:, 1], c=vae_kmeans.labels_, cmap='viridis')
plt.title(f'VAE Latent Space Clusters (t-SNE) | k={optimal_k}')
plt.colorbar(label='Cluster ID')
plt.show()