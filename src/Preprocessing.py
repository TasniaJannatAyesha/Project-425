import os
import pandas as pd
import librosa
import numpy as np
import warnings


warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")

INPUT_PATH = os.path.join(BASE_DIR, "data", "Processed_Music_Dataset.csv") 

FINAL_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "Final_Refined_Dataset.csv")

def refine_dataset():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Could not find {INPUT_PATH}. Please run your transcription script first.")
        return

   
    df = pd.read_csv(INPUT_PATH)
    print(f"Initial songs loaded: {len(df)}")

   
    invalid_tags = ["Transcription Failed", "No vocals detected"]
    df = df[~df['lyrics'].isin(invalid_tags)]
    df = df[df['lyrics'].str.len() > 20] 
    
    print(f"Songs remaining after cleaning lyrics: {len(df)}")

    refined_data = []

    
    print("--- Extracting Advanced Audio Features ---")
    for index, row in df.iterrows():
        file_path = os.path.join(AUDIO_DIR, row['filename'])
        
        if os.path.exists(file_path):
            try:
                
                y, sr = librosa.load(file_path, duration=30)
                
                
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                mfcc_mean = np.mean(mfccs.T, axis=0)
                
               
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                centroid_mean = np.mean(centroid)
                
                
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma.T, axis=0)
                
                
                rms = librosa.feature.rms(y=y)
                rms_mean = np.mean(rms)

               
                entry = {
                    'filename': row['filename'],
                    'genre': row['genre'],
                    'songTitle': row['songTitle'],
                    'artistName': row['artistName'],
                    'lyrics': row['lyrics'],
                    'spectral_centroid': centroid_mean,
                    'energy_rms': rms_mean
                }
                
                
                for i, val in enumerate(mfcc_mean): entry[f'mfcc_{i}'] = val
                for i, val in enumerate(chroma_mean): entry[f'chroma_{i}'] = val
                
                refined_data.append(entry)
                
                if len(refined_data) % 50 == 0:
                    print(f"Processed {len(refined_data)} songs...")

            except Exception as e:
                print(f"Error processing {row['filename']}: {e}")
        else:
            print(f"Warning: File {row['filename']} missing during refinement.")


    final_df = pd.DataFrame(refined_data)
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)
    
    print(f"\n--- SUCCESS! ---")
    print(f"Final Refined Dataset saved to: {FINAL_OUTPUT_PATH}")
    print(f"Total rows: {len(final_df)}")

if __name__ == "__main__":
    refine_dataset()