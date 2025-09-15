"""
Audio feature extraction utilities for the FMA dataset.

This module provides functions to compute simple audio descriptors
(MFCC mean and standard deviation) for tracks listed in a processed
index file.  It relies on `src.config.Settings` to load paths from
environment variables (see `.env`), but it does not read audio files
directly from `FMA_SMALL_DIR`—instead it expects the index file to
contain an `audio_path` column with absolute paths to each MP3 file.

Functions:
  - mfcc_mean_std: compute MFCC mean and std for one audio file.
  - tiny_feature_check: extract MFCC features for a small sample of tracks.
  - extract_features_full: compute MFCC features for all tracks in an index.

To run the full extraction from the command line:
    python -m src.features.audio_features
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# Import settings to ensure .env is loaded and directories exist.
# Even though we don't use FMA_SMALL_DIR directly here, instantiating
# Settings will validate that the environment is configured correctly.
from src.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


def mfcc_mean_std(
    audio_path: Path,
    sr: int = 22_050,
    n_mfcc: int = 13,
) -> Optional[np.ndarray]:
    """
    Load an audio file and compute the mean and standard deviation of its MFCCs.

    Parameters
    ----------
    audio_path : Path
        Absolute path to the audio file (MP3).
    sr : int, optional
        Target sampling rate; audio will be resampled if needed.  Default is 22050 Hz.
    n_mfcc : int, optional
        Number of MFCC coefficients to compute.  Default is 13.

    Returns
    -------
    np.ndarray or None
        A 2*n_mfcc-length array containing the MFCC means followed by
        the MFCC standard deviations.  Returns None if the audio fails
        to load or is shorter than one second.
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception:
        return None
    # Skip files shorter than one second
    if y.size < sr:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def tiny_feature_check(
    index_path: Path,
    sample_size: int = 10,
    out_path: Path = Path("data/interim/mfcc_10.parquet"),
) -> Path:
    """
    Sample a few tracks from the index and compute MFCC features as a sanity check.

    Parameters
    ----------
    index_path : Path
        Path to a processed FMA index file (Parquet) with `track_id` and `audio_path`.
    sample_size : int, optional
        Number of random tracks to sample.  Default is 10.
    out_path : Path, optional
        Where to write the resulting Parquet file.  Default is `data/interim/mfcc_10.parquet`.

    Returns
    -------
    Path
        The path to the written Parquet file.
    """
    df = pd.read_parquet(index_path)
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    rows = []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="extracting MFCC"):
        vec = mfcc_mean_std(Path(row["audio_path"]))
        if vec is None:
            continue
        rows.append({"track_id": int(row["track_id"]), "feature": vec})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    log.info("Wrote %d feature rows → %s", len(rows), out_path)
    return out_path


def extract_features_full(
    index_path: Path,
    out_path: Path = Path("data/processed/fma_small_mfcc.parquet"),
) -> Path:
    """
    Compute MFCC mean/std features for every track listed in the index.

    Parameters
    ----------
    index_path : Path
        Path to a processed FMA index file (Parquet) with `track_id` and `audio_path`.
    out_path : Path, optional
        Where to write the resulting Parquet file.  Default is `data/processed/fma_small_mfcc.parquet`.

    Returns
    -------
    Path
        The path to the written Parquet file.
    """
    df = pd.read_parquet(index_path)
    features = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting features"):
        vec = mfcc_mean_std(Path(row["audio_path"]))
        if vec is None:
            continue
        features.append({"track_id": int(row["track_id"]), "feature": vec})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(features).to_parquet(out_path, index=False)
    log.info("Saved features for %d tracks → %s", len(features), out_path)
    return out_path

def _agg(mat: np.ndarray) -> np.ndarray:
    """mean/std along time for a (F x T) matrix (or 1 x T)."""
    return np.concatenate([mat.mean(axis=1), mat.std(axis=1)])

def compute_features_v2(audio_path: Path, sr: int = 22050) -> np.ndarray | None:
    """
    MFCC(13) + CHROMA(12) + spectral stats + tempo:
      26 + 24 + 2+2+2+2+2 + 1 = 61 dims
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception:
        return None
    if y.size < sr:
        return None

    # Mel-spectrogram (dB) for MFCCs
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    M_db = librosa.power_to_db(M, ref=np.max)

    # MFCCs (13)
    mfcc = librosa.feature.mfcc(S=M_db, n_mfcc=13)                 # 13 x T  -> 26
    # Chroma (12)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)                # 12 x T  -> 24

    # Spectral one-band features (1 x T each) -> 2 each when agg
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)             # 1 x T
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)            # 1 x T
    sr85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)  # 1 x T
    sf = librosa.feature.spectral_flatness(y=y)                    # 1 x T
    zcr = librosa.feature.zero_crossing_rate(y)                    # 1 x T

    # Tempo (scalar)
    tempo = librosa.feature.rhythm.tempo(y=y, sr=sr, aggregate=np.mean).reshape(1)

    vec = np.concatenate([
        _agg(mfcc),           # 26
        _agg(chroma),         # 24
        _agg(sc), _agg(sb), _agg(sr85), _agg(sf), _agg(zcr),  # 2*5 = 10
        tempo                 # 1
    ])
    # total = 61
    return vec

def extract_features_full_v2(index_path: Path,
                             out_path: Path = Path("data/processed/fma_small_feats_v2.parquet")) -> Path:
    df = pd.read_parquet(index_path)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="extracting features v2"):
        v = compute_features_v2(Path(r["audio_path"]))
        if v is None:
            continue
        rows.append({"track_id": int(r["track_id"]), "feature": v})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Saved v2 features for {len(rows)} tracks → {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute MFCC features for tracks in the FMA small subset."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If >0, sample this many tracks instead of processing all tracks.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/processed/fma_small_index.parquet"),
        help="Path to the processed index file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Parquet file. "
             "If not provided, defaults to `data/interim/mfcc_<sample>.parquet` when sampling "
             "or `data/processed/fma_small_mfcc.parquet` when processing all.",
    )

    args = parser.parse_args()

    index_path = args.index
    if args.sample > 0:
        if args.out is None:
            out_file = Path(f"data/interim/mfcc_{args.sample}.parquet")
        else:
            out_file = args.out
        tiny_feature_check(index_path, sample_size=args.sample, out_path=out_file)
    else:
        out_file = (
            args.out
            if args.out is not None
            else Path("data/processed/fma_small_mfcc.parquet")
        )
        extract_features_full(index_path, out_path=out_file)