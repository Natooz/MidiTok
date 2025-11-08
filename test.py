import glob
from importlib.resources import path
from pathlib import Path
from symusic import Score
from miditok.tokenizations.remi import REMI
from miditok.utils.split import split_files_for_training
import time

start_time = time.perf_counter()
root_dir = Path("/home/wombat/Documents/projects/music/test/")
tracks_root = Path("/media/wombat/c6928dc9-ba03-411d-9483-8e28df5973b9/Music Data/Traning Data/maestro-v3.0.0/")
tracks = list(glob.glob(str(tracks_root) + "/*/*.midi"))

track_paths = []
for t in tracks:
    track_paths.append(Path(tracks_root / t))

end_time = time.perf_counter()
print(f"Track paths collected in {end_time - start_time:.2f} seconds.")


tokenizer_file = "/media/wombat/c6928dc9-ba03-411d-9483-8e28df5973b9/Music Data/HuggingFace_Mistral_Transformer_Single_Instrument/HuggingFace_Mistral_Transformer_Single_Instrument_v4_single_track.json"
tokenizer = REMI(params=Path(tokenizer_file))


def preprocessing_method(score: Score) -> Score:
    non_drum_tracks = [track for track in score.tracks if not track.is_drum]
    score.tracks = non_drum_tracks # Replace the track list

    return score

parallel_workers = 1
start_time = time.perf_counter()
split_files_for_training(track_paths, tokenizer, root_dir, 512, preprocessing_method=preprocessing_method, parallel_workers=parallel_workers)
end_time = time.perf_counter()
print(f"File splitting completed in {end_time - start_time:.2f} seconds. With {parallel_workers} parallel workers.")