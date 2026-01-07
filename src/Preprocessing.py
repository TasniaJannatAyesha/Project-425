import os
import pandas as pd
import librosa
import numpy as np
import warnings

# Ignore librosa warnings
warnings.filterwarnings("ignore")

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
# This is the file you ALREADY generated with Whisper
INPUT_PATH = os.path.join(BASE_DIR, "data", "Processed_Music_Dataset.csv") 
# This will be the high-quality final version
FINAL_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "Final_Refined_Dataset.csv")

def refine_dataset():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Could not find {INPUT_PATH}. Please run your transcription script first.")
        return

    # 1. Load the dataset
    df = pd.read_csv(INPUT_PATH)
    print(f"Initial songs loaded: {len(df)}")

    # 2. Cleanup: Remove rows where lyrics are missing or failed
    # We remove "Transcription Failed" and "No vocals detected" or very short text
    invalid_tags = ["Transcription Failed", "No vocals detected"]
    df = df[~df['lyrics'].isin(invalid_tags)]
    df = df[df['lyrics'].str.len() > 20] # Remove songs with almost no text
    
    print(f"Songs remaining after cleaning lyrics: {len(df)}")

    refined_data = []

    # 3. Extract Advanced Audio Features for the cleaned rows
    print("--- Extracting Advanced Audio Features ---")
    for index, row in df.iterrows():
        file_path = os.path.join(AUDIO_DIR, row['filename'])
        
        if os.path.exists(file_path):
            try:
                # Load audio
                y, sr = librosa.load(file_path, duration=30)
                
                # Feature A: 20 MFCCs (Better than 13 for clustering) 
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                mfcc_mean = np.mean(mfccs.T, axis=0)
                
                # Feature B: Spectral Centroid (Indicates 'brightness')
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                centroid_mean = np.mean(centroid)
                
                # Feature C: Chroma (Harmonic content/Music key)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma.T, axis=0)
                
                # Feature D: RMS (Energy/Volume) 
                rms = librosa.feature.rms(y=y)
                rms_mean = np.mean(rms)

                # Combine everything into a clean row
                entry = {
                    'filename': row['filename'],
                    'genre': row['genre'],
                    'songTitle': row['songTitle'],
                    'artistName': row['artistName'],
                    'lyrics': row['lyrics'],
                    'spectral_centroid': centroid_mean,
                    'energy_rms': rms_mean
                }
                
                # Add MFCC (0-19) and Chroma (0-11) as columns
                for i, val in enumerate(mfcc_mean): entry[f'mfcc_{i}'] = val
                for i, val in enumerate(chroma_mean): entry[f'chroma_{i}'] = val
                
                refined_data.append(entry)
                
                if len(refined_data) % 50 == 0:
                    print(f"Processed {len(refined_data)} songs...")

            except Exception as e:
                print(f"Error processing {row['filename']}: {e}")
        else:
            print(f"Warning: File {row['filename']} missing during refinement.")

    # 4. Save the high-quality final CSV
    final_df = pd.DataFrame(refined_data)
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)
    
    print(f"\n--- SUCCESS! ---")
    print(f"Final Refined Dataset saved to: {FINAL_OUTPUT_PATH}")
    print(f"Total rows: {len(final_df)}")

if __name__ == "__main__":
    refine_dataset()