import gc
import json
import os
import torch
from bert_score import score
from transformers import AutoTokenizer, AutoModel
from joblib import Parallel, delayed
from tqdm import tqdm  # Import tqdm for progress bar

# Load tokenizer and model globally to avoid loading them multiple times
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Function to split lines into songs based on the "*" delimiter, removing blank lines
def split_into_songs(lines):
    songs = []
    current_song = []
    for line in lines:
        if "*" in line.strip():
            if current_song:
                songs.append(current_song)
                current_song = []
        else:
            if line.strip():  # Only add non-blank lines
                current_song.append(line.strip())
    if current_song:
        songs.append(current_song)
    return songs

# what about sentence score
def compute_bertscore(ref_song, test_song):
    with torch.no_grad():
        _, _, F1 = score(test_song, ref_song, lang="es", batch_size=8, model_type=MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu", verbose=False)
    return F1.mean().item()  # Average F1 score across song

def clear_memory():
    gc.collect()  # Collect garbage to free memory
    torch.cuda.empty_cache()  # If using CUDA, clear the GPU cache (if applicable)


if __name__ == "__main__":
    
    # Read reference and hypothesis files
    with open("data/spanish/references.txt", "r", encoding="utf-8") as f:
        references = f.readlines()

    with open("data/spanish/gpts.txt", "r", encoding="utf-8") as f:
        hypotheses = f.readlines()

    # Ensure JSON file exists
    json_path = "results/spanish/gpts.json"
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r", encoding="utf-8") as file:
            try:
                song_objects = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: {json_path} is not valid. Resetting it.")
                song_objects = []
    else:
        song_objects = []

    # Split into songs
    ref_songs = split_into_songs(references)
    test_songs = split_into_songs(hypotheses)

    if len(ref_songs) != len(test_songs):
        raise ValueError(f"Mismatch in song count: {len(ref_songs)} references vs {len(test_songs)} hypotheses")

    # Process each song in parallel and collect results
    results = Parallel(n_jobs=10)(
        delayed(compute_bertscore)(ref_song, test_song)
        for ref_song, test_song in tqdm(zip(ref_songs, test_songs), total=len(ref_songs), desc="Processing Songs")
    )

    # Update song_objects with results
    for i, result in enumerate(results):
        song_objects[i]["bertscore"] = result

    # Save updated results to JSON file
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(song_objects, file, indent=4, ensure_ascii=False)

    clear_memory()
