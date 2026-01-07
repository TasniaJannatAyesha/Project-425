# Analysis of Model Performance across Varying Task Complexities

## Project Overview
This repository contains a Machine Learning pipeline designed to evaluate neural network performance across three distinct levels of task difficulty: **Easy**, **Medium**, and **Hard**. Additionally, it implements a **Variational Autoencoder (VAE)** to analyze feature extraction and generative capabilities.

The goal is to demonstrate how model architecture and complexity must scale to meet increasing data challenges.

## Repository Structure & File Descriptions

The source code is located in the `src/` folder. Below is a detailed breakdown of what each file is responsible for:

### 1. Data Preparation
* **`src/Preprocessing.py`**
    * **Function:** This is the entry point for the data pipeline. It handles the ingestion of raw data.
    * **Key Operations:**
        * Loads the dataset.
        * Performs normalization (scaling inputs to a standard range).
        * Splits data into training and testing sets to prepare it for the models.

### 2. Generative Modeling
* **`src/vae_model.py`**
    * **Function:** Contains the class definition for the Variational Autoencoder (VAE).
    * **Key Operations:** Implements the Encoder (compression), the Reparameterization trick (sampling from latent space), and the Decoder (reconstruction). This is used to understand the underlying structure of the data.

### 3. Model Architectures
These files define the neural networks used for the specific difficulty tiers.
* **`src/medium_models.py`**
    * **Function:** Defines the neural network architecture suited for intermediate complexity. Likely includes increased hidden layers compared to the baseline.
* **`src/hard_models.py`**
    * **Function:** Defines the most complex model architectures. These models are designed to handle the "Hard" task, potentially utilizing deeper layers, dropout, or batch normalization to prevent overfitting on complex patterns.

### 4. Task Execution (Experiments)
These are the main executable scripts. Running these triggers the training and evaluation loops.
* **`src/easy_task.py`**
    * **Function:** Runs the baseline experiment. It trains a simple model on the easiest version of the task to establish a performance benchmark.
* **`src/medium_task.py`**
    * **Function:** Imports models from `medium_models.py` and trains them on the intermediate task, logging accuracy and loss.
* **`src/hard_task.py`**
    * **Function:** Imports models from `hard_models.py` and runs the stress-test experiment on the most difficult data partition.

## Getting Started

### Prerequisites
* Python 3.x
* Required libraries: `numpy`, `pandas`, `tensorflow` (or `torch`), `matplotlib`.


Usage
To run the experiments, execute the task files from the terminal:

1. Prepare the Data:

python src/Preprocessing.py

2. Run the Tasks:

python src/easy_task.py
python src/medium_task.py
python src/hard_task.py