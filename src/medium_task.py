import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from medium_models import HybridConvVAE


df = pd.read_csv("data/Final_Refined_Dataset.csv")
audio_cols = [f'mfcc_{i}' for i in range(20)] + ['spectral_centroid', 'energy_rms']
audio_features = StandardScaler().fit_transform(df[audio_cols].values)


print("Encoding lyrics...")
lyric_model = SentenceTransformer('all-MiniLM-L6-v2')
lyric_embeddings = lyric_model.encode(df['lyrics'].tolist())


audio_tensor = torch.FloatTensor(audio_features)
lyric_tensor = torch.FloatTensor(lyric_embeddings)


model = HybridConvVAE(audio_dim=len(audio_cols), lyric_dim=lyric_embeddings.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training Hybrid VAE...")
for epoch in range(50):
    recon, mu, logvar = model(audio_tensor, lyric_tensor)
    
    target = torch.cat((audio_tensor, lyric_tensor), dim=1)
    loss = torch.nn.functional.mse_loss(recon, target) + 0.001 * torch.sum(mu**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


with torch.no_grad():
    _, latent_features, _ = model(audio_tensor, lyric_tensor)
    latent_features = latent_features.numpy()


print("\n--- Clustering Experiments ---")
n_clusters = 6 


kmeans_labels = KMeans(n_clusters=n_clusters).fit_predict(latent_features)


agglo_labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(latent_features)


dbscan_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(latent_features)


true_labels = LabelEncoder().fit_transform(df['genre'])

def evaluate(labels, name):
    
    mask = labels != -1
    if len(set(labels[mask])) < 2: return 
    
    sil = silhouette_score(latent_features[mask], labels[mask])
    db = davies_bouldin_score(latent_features[mask], labels[mask])
    ari = adjusted_rand_score(true_labels[mask], labels[mask])
    print(f"{name}: Silhouette={sil:.3f}, DB Index={db:.3f}, ARI={ari:.3f}")

evaluate(kmeans_labels, "K-Means")
evaluate(agglo_labels, "Agglomerative")
evaluate(dbscan_labels, "DBSCAN")