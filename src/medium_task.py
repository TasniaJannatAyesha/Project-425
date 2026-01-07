import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from medium_models import HybridConvVAE

# 1. Load Data
df = pd.read_csv("data/Final_Refined_Dataset.csv")
audio_cols = [f'mfcc_{i}' for i in range(20)] + ['spectral_centroid', 'energy_rms']
audio_features = StandardScaler().fit_transform(df[audio_cols].values)

# 2. Hybrid Feature Representation (Audio + Lyrics)
print("Encoding lyrics...")
lyric_model = SentenceTransformer('all-MiniLM-L6-v2')
lyric_embeddings = lyric_model.encode(df['lyrics'].tolist())

# Convert to Tensors
audio_tensor = torch.FloatTensor(audio_features)
lyric_tensor = torch.FloatTensor(lyric_embeddings)

# 3. Train Hybrid ConvVAE
model = HybridConvVAE(audio_dim=len(audio_cols), lyric_dim=lyric_embeddings.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training Hybrid VAE...")
for epoch in range(50):
    recon, mu, logvar = model(audio_tensor, lyric_tensor)
    # Combine original inputs for loss
    target = torch.cat((audio_tensor, lyric_tensor), dim=1)
    loss = torch.nn.functional.mse_loss(recon, target) + 0.001 * torch.sum(mu**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. Extract Latent Representations
with torch.no_grad():
    _, latent_features, _ = model(audio_tensor, lyric_tensor)
    latent_features = latent_features.numpy()

# 5. Experiment with 3 Clustering Algorithms
print("\n--- Clustering Experiments ---")
n_clusters = 6 # From your Easy Task Elbow Method

# Algorithm 1: K-Means
kmeans_labels = KMeans(n_clusters=n_clusters).fit_predict(latent_features)

# Algorithm 2: Agglomerative (Hierarchical)
agglo_labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(latent_features)

# Algorithm 3: DBSCAN (Density-based)
dbscan_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(latent_features)

# 6. Evaluation Metrics
# We use genre labels for ARI (Adjusted Rand Index)
true_labels = LabelEncoder().fit_transform(df['genre'])

def evaluate(labels, name):
    # DBSCAN often creates -1 labels (noise); we filter them for metric calculation
    mask = labels != -1
    if len(set(labels[mask])) < 2: return # Skip if only 1 cluster found
    
    sil = silhouette_score(latent_features[mask], labels[mask])
    db = davies_bouldin_score(latent_features[mask], labels[mask])
    ari = adjusted_rand_score(true_labels[mask], labels[mask])
    print(f"{name}: Silhouette={sil:.3f}, DB Index={db:.3f}, ARI={ari:.3f}")

evaluate(kmeans_labels, "K-Means")
evaluate(agglo_labels, "Agglomerative")
evaluate(dbscan_labels, "DBSCAN")