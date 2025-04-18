from sacrebleu.metrics import BLEU
import json

# Function to split lines into songs based on the "*" delimiter
def split_into_songs(lines):
    songs = []
    current_song = []
    for line in lines:
        if "*" in line.strip():
            if current_song:
                songs.append(current_song)
                current_song = []
        else:
            if line.strip() == "":
                current_song.append("")  # Preserve stanza break
            else:
                current_song.append(line.strip())  # Clean up content line
    if current_song:
        songs.append(current_song)
    return songs


if __name__ == '__main__':

    # Load hypothesis translations (machine-generated)
    with open("data/irish/nllbs.txt", "r", encoding="utf-8") as f: 
        test_lines = [line.strip() for line in f.readlines()]

    # Load reference translations
    with open("data/irish/references.txt", "r", encoding="utf-8") as f:
        references = [line.strip() for line in f.readlines()]

    json_path = "results/irish/nllbs.json"
    with open(json_path, "r", encoding="utf-8") as file:
        song_objects = json.load(file)
    
    # Split references and test_lines into songs
    ref_songs = split_into_songs(references)
    test_songs = split_into_songs(test_lines)

    # Compute BLEU score
    bleu = BLEU(effective_order=True)
    for i, (ref_song, hyp_song) in enumerate(zip(ref_songs, test_songs)):
        print(i)
        bleu_score = 0
        for j, (ref_line, hyp_line) in enumerate(zip(ref_song, hyp_song)):
            # stanza break skip logic
            if ref_line == "" or hyp_line == "":
                continue
            bleu_score += bleu.sentence_score(hyp_line, [ref_line]).score
        song_objects[i]["sacrebleu"] = (bleu_score/len(ref_song)) / 100

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(song_objects, file, indent=4, ensure_ascii=False)
