import os
import pandas as pd
import librosa
import numpy as np
import whisper
import warnings


warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
METADATA_PATH = os.path.join(BASE_DIR, "data", "GTZAN_metadata.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "Processed_Music_Dataset.csv")


print("Loading Whisper AI model (this may take a moment)...")
model = whisper.load_model("tiny")

def process_music_data():
   
    if not os.path.exists(METADATA_PATH):
        print(f"ERROR: Could not find {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    all_mfccs = []
    all_lyrics = []

    print(f"--- Starting Processing: {len(df)} files ---")
    
    for index, row in df.iterrows():
        
        file_path = os.path.join(AUDIO_DIR, row['filename'])
        
        
        current_mfcc = np.zeros(13)
        current_lyric = "Transcription Failed"

        if os.path.exists(file_path):
            try:
                
                y, sr = librosa.load(file_path, duration=30)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                current_mfcc = np.mean(mfcc.T, axis=0)
                
               
                result = model.transcribe(file_path, language='en', initial_prompt="Song lyrics")
                current_lyric = result['text'].strip()
                
                
                if not current_lyric:
                    current_lyric = "No vocals detected"

                print(f"[{index+1}/{len(df)}] ✅ Processed: {row['filename']}")

            except Exception as e:
                print(f"[{index+1}/{len(df)}] ❌ Error on {row['filename']}: {e}")
        else:
            print(f"[{index+1}/{len(df)}] ⚠️ File NOT FOUND: {row['filename']}")

      
        all_mfccs.append(current_mfcc)
        all_lyrics.append(current_lyric)

    
    df['lyrics'] = all_lyrics
    
    mfcc_columns = [f'mfcc_{i}' for i in range(13)]
    mfcc_df = pd.DataFrame(all_mfccs, columns=mfcc_columns)
    
  
    final_df = pd.concat([df.reset_index(drop=True), mfcc_df], axis=1)
    
    
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n--- SUCCESS! ---")
    print(f"Final dataset saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(final_df)}")

if __name__ == "__main__":
    process_music_data()