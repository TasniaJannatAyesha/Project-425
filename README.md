# Unsupervised Music Clustering using Multi-Modal VAEs

This project implements an end-to-end Machine Learning pipeline to analyze and cluster music genres in an unsupervised manner. It progresses through three levels of complexity (**Easy**, **Medium**, **Hard**), comparing linear baselines (PCA) against generative deep learning models (**Variational Autoencoders**, **Hybrid Conv-VAEs**, and **Beta-VAEs**).

A key contribution of this project is the construction of a custom **Multi-Modal Dataset**, where raw audio from the GTZAN collection is augmented with lyrics transcribed via **OpenAI's Whisper model**.

## 📂 Repository Structure

The source code is located in the `src/` folder. Below is a breakdown of the pipeline workflow:

### 1. Data Pipeline (Run in Order)
* **`src/dataset.py`**
    * **Function:** Initial Data Ingestion & Transcription.
    * **Key Operations:** * Iterates through raw audio files from the GTZAN dataset.
        * Uses **OpenAI Whisper (Tiny)** to transcribe lyrics from every track.
        * Extracts basic 13-coefficient MFCCs.
        * **Output:** Saves `data/Processed_Music_Dataset.csv` (Intermediate raw dataset).

* **`src/Preprocessing.py`**
    * **Function:** Data Refinement & Advanced Feature Engineering.
    * **Key Operations:**
        * **Cleaning:** Filters out instrumental tracks ("No vocals detected") and short/noisy transcriptions.
        * **Feature Extraction:** Re-processes audio to extract **Advanced Features**: 20 MFCCs, Spectral Centroid, Chroma Vectors, and RMS Energy.
        * **Output:** Saves `data/Final_Refined_Dataset.csv` (The final clean dataset used for training).

### 2. Model Architectures
* **`src/vae_model.py`**
    * **Function:** Defines the core `BasicVAE` class used in the Easy Task.
    * **Architecture:** A lightweight fully connected encoder-decoder designed for low-dimensional spectral data.

* **`src/medium_models.py`**
    * **Function:** Defines the **Hybrid Convolutional VAE**.
    * **Architecture:** Features a **1D-CNN branch** for temporal audio features and a **Dense branch** for Semantic Lyric Embeddings, fused into a shared latent space.

* **`src/hard_models.py`**
    * **Function:** Defines the **Beta-VAE ($\beta$-VAE)**.
    * **Architecture:** A deep fully connected network with a modified loss function ($\beta=4.0$) to enforce **disentanglement**, prioritizing statistically independent factors of variation.

### 3. Experiments & Analysis
* **`src/easy_task.py`**
    * **Goal:** Baseline Comparison.
    * **Experiment:** Compares **PCA** vs. **Basic VAE** on standard MFCCs.

* **`src/medium_task.py`**
    * **Goal:** High-Dimensional Hybrid Clustering.
    * **Experiment:** Trains the Hybrid Conv-VAE and tests clustering algorithms (K-Means, Agglomerative, DBSCAN).

* **`src/hard_task.py`**
    * **Goal:** Disentangled Representation Learning.
    * **Experiment:** Trains the Beta-VAE to find latent musical structures. Generates **t-SNE latent space plots** and **Cluster Composition Bar Charts**.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have the following installed:
* Python 3.8+
* `torch` (PyTorch)
* `librosa` (Audio Processing)
* `openai-whisper` (Transcription)
* `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

### Installation
To install all required dependencies, run:

```bash
pip install torch torchvision torchaudio librosa openai-whisper scikit-learn pandas numpy matplotlib seaborn

Usage
Run the pipeline in the following order from your terminal:

1. Data Ingestion (Step 1): Runs Whisper to transcribe lyrics (Warning: This is computationally expensive).

Bash

python src/dataset.py
2. Data Refinement (Step 2): Cleans the data and extracts advanced audio features.

Bash

python src/Preprocessing.py
3. Run Experiments: Execute the tasks sequentially to reproduce the report results.

Bash

# Easy Task: Baseline Comparison (PCA vs VAE)
python src/easy_task.py

# Medium Task: Hybrid Audio-Lyric Clustering
python src/medium_task.py

# Hard Task: Disentangled Beta-VAE & t-SNE Visualization
python src/hard_task.py
📊 Key Results
Linearity of Audio: For simple spectral features (MFCCs), linear PCA outperforms VAEs, suggesting the manifold of standard audio features is largely linear.

The Semantic Gap: Including lyrics improves semantic clustering but introduces sparsity that challenges density-based algorithms like DBSCAN.

Disentanglement: The Beta-VAE successfully discovers continuous musical factors (like "Acoustic Energy" or "Lyrical Density") rather than simply memorizing genre labels, creating novel groupings (e.g., grouping Rock and Blues together based on spectral similarity).
