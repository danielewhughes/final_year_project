import subprocess
import tempfile
import os
import json
from ufal.udpipe import Model, Pipeline

METEOR_JAR_PATH = "/Users/danielhughes/Documents/College/Fourth-Year/Capstone/Dissertation/metrics/semantics/meteor_ga/meteor-1.5.jar"
MODEL_PATH = "irish-idt-ud-2.5-191206.udpipe"

def lemmatise(conllu_output):
    lemmas = []
    for line in conllu_output.splitlines():
        if line and not line.startswith("#"):
            columns = line.split("\t")
            if len(columns) > 2:  # Ensure the line has enough columns
                lemmas.append(columns[2])  # LEMMA is in the 3rd column
    return lemmas


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


def compute_meteor_score(references, hypotheses, meteor_jar_path, pipeline, language):
    assert len(references) == len(hypotheses), f"reference and hypothesis lists must be the same length. {references}"

    # Write temp files with one sentence per line
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as ref_file, \
         tempfile.NamedTemporaryFile(mode='w+', delete=False) as hyp_file:
        
        for ref in references:
            ref = " ".join(lemmatise(pipeline.process(ref)))
            ref_file.write(ref.strip() + '\n')
        for hyp in hypotheses:
            hyp = " ".join(lemmatise(pipeline.process(hyp)))
            hyp_file.write(hyp.strip() + '\n')

        ref_file_path = ref_file.name
        hyp_file_path = hyp_file.name

    try:
        cmd = [
            "java", "-Xmx2G", "-jar", meteor_jar_path,
            hyp_file_path, ref_file_path,
            "-l", language
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Extract final score from METEOR output
        score_line = [line for line in result.stdout.split('\n') if 'Final score:' in line]
        score = float(score_line[0].split()[-1]) if score_line else 0.0
        
    except Exception as e:
        print(f"Error during METEOR scoring: {e}")
        score = 0.0

    finally:
        os.remove(ref_file_path)
        os.remove(hyp_file_path)

    return score


if __name__ == '__main__':
    ### Using UDPipe lemmatiser
    model = Model.load(MODEL_PATH)
    if not model:
        print("Error: Could not load the UDPipe model.")
        exit(1)
    pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

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

    # Compute METEOR score
    for i, (ref_song, hyp_song) in enumerate(zip(ref_songs, test_songs)):
        print(i)
        boolean = song_objects[i].get("meteor")
        if boolean:
            continue
        meteor_score = compute_meteor_score(ref_song, hyp_song, METEOR_JAR_PATH, pipeline, "other")
        print(meteor_score)
        song_objects[i]["meteor"] = meteor_score

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(song_objects, file, indent=4, ensure_ascii=False)
