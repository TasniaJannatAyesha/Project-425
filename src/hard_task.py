import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sentence_transformers import SentenceTransformer
from hard_models import BetaVAE, SimpleAutoencoder

# 1. Load and Combine Data (Multi-modal)
df = pd.read_csv("data/Final_Refined_Dataset.csv")
audio_cols = [f'mfcc_{i}' for i in range(20)] + ['spectral_centroid', 'energy_rms'] + [f'chroma_{i}' for i in range(12)]
audio_feat = StandardScaler().fit_transform(df[audio_cols].values)

print("Encoding lyrics for Hard Task...")
lyric_model = SentenceTransformer('all-MiniLM-L6-v2')
lyric_embeddings = lyric_model.encode(df['lyrics'].tolist())

# Combine Audio + Lyrics into one big feature vector
X = np.hstack((audio_feat, lyric_embeddings))
X_tensor = torch.FloatTensor(X)
true_labels = LabelEncoder().fit_transform(df['genre'])

# 2. Train Beta-VAE (The Champion)
beta = 4.0 # Beta > 1 encourages disentanglement
model = BetaVAE(input_dim=X.shape[1], latent_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training Beta-VAE...")
for epoch in range(60):
    recon, mu, logvar = model(X_tensor)
    recon_loss = torch.nn.functional.mse_loss(recon, X_tensor)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + (beta * 0.001) * kld_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 3. Baseline 1: Autoencoder
ae = SimpleAutoencoder(input_dim=X.shape[1], latent_dim=32)
ae_opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
for _ in range(60):
    recon, _ = ae(X_tensor)
    loss = torch.nn.functional.mse_loss(recon, X_tensor)
    ae_opt.zero_grad(); loss.backward(); ae_opt.step()

# 4. Extract all latent features for comparison
with torch.no_grad():
    _, beta_latent, _ = model(X_tensor)
    _, ae_latent = ae(X_tensor)
pca_latent = PCA(n_components=32).fit_transform(X)

# 5. Clustering and Evaluation Function
def get_purity(labels, true_labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, labels)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

def run_eval(features, name):
    labels = KMeans(n_clusters=len(np.unique(true_labels)), random_state=42).fit_predict(features)
    sil = silhouette_score(features, labels)
    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)
    pur = get_purity(labels, true_labels)
    print(f"{name:15} | Sil: {sil:.3f} | ARI: {ari:.3f} | NMI: {nmi:.3f} | Purity: {pur:.3f}")

print("\n--- HARD TASK RESULTS (Comparison) ---")
run_eval(X, "Direct Spectral")
run_eval(pca_latent, "PCA + K-Means")
run_eval(ae_latent.numpy(), "AE + K-Means")
run_eval(beta_latent.numpy(), "Beta-VAE")

# 6. VISUALIZATION: Cluster Distribution over Genres (For Report)
beta_labels = KMeans(n_clusters=6, random_state=42).fit_predict(beta_latent.numpy())
df['cluster'] = beta_labels
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='cluster', hue='genre')
plt.title("Cluster Distribution over Genres (Hard Task)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("results/cluster_genre_dist.png")
print("\nVisualization saved to results/cluster_genre_dist.png")