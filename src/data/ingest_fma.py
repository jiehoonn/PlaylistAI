"""
Functions for ingesting FMA metadata and building an index of audio files.

This script reads the FMA `tracks.csv` metadata file (two header rows) to build
a compact index of tracks in the “small” subset, including metadata and absolute
audio paths.  You can run this module as a script.
"""
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from src.config import get_settings
from src.utils.logging import get_logger
from .fma_paths import track_id_to_relpath

log = get_logger(__name__)

def build_small_index() -> Path:
    """Load FMA metadata and construct a track index for the small subset."""
    tracks_csv = get_settings().FMA_METADATA_DIR / "tracks.csv"
    if not tracks_csv.is_file():
        raise FileNotFoundError(f"tracks.csv not found: {tracks_csv}")

    df = pd.read_csv(tracks_csv, header=[0, 1], index_col=0, low_memory=False)

    # After selecting the small subset:
    subset_col = df[('set', 'subset')]
    small_mask = subset_col == 'small'
    small_df = df[small_mask].copy()

    # Drop non-numeric index rows (e.g. the line starting with 'track_id')
    numeric_idx = pd.to_numeric(small_df.index, errors='coerce')
    small_df = small_df[~numeric_idx.isna()].copy()

    # Convert the remaining index to integers
    small_df.index = small_df.index.astype(int)

    wanted_cols: List[tuple] = [
        ('track', 'genre_top'),
        ('track', 'title'),
        ('album', 'title'),
        ('artist', 'name'),
        ('track', 'date_created'),
        ('track', 'genres_all'),
    ]
    data: Dict[str, pd.Series] = {}
    for col in wanted_cols:
        if col in small_df.columns:
            col_name = f"{col[0]}_{col[1]}"
            data[col_name] = small_df[col]
    data['track_id'] = small_df.index.astype(int)

    relpaths, abspaths = [], []
    for tid in data['track_id']:
        rel_path = track_id_to_relpath(tid)
        abs_path = get_settings().FMA_SMALL_DIR / rel_path
        relpaths.append(str(rel_path))
        abspaths.append(str(abs_path.resolve()))

    data['relpath'] = relpaths
    data['audio_path'] = abspaths

    index_df = pd.DataFrame(data)

    exists = []
    for p in tqdm(index_df['audio_path'], desc="verifying audio files"):
        exists.append(Path(p).is_file())
    index_df['exists'] = exists
    kept_df = index_df[index_df['exists']].drop(columns=['exists']).reset_index(drop=True)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "fma_small_index.parquet"
    kept_df.to_parquet(out_file, index=False)
    log.info("Saved index with %s rows → %s", len(kept_df), out_file)
    return out_file

if __name__ == "__main__":
    build_small_index()