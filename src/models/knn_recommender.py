"""
KNN audio recommender built on MFCC mean/std features.

Usage:
  # Train the model (fits scaler + KNN, saves artifacts)
  python -m src.models.knn_recommender --build

  # Query recommendations for a seed track_id (prints titles/artists)
  python -m src.models.knn_recommender --query 2 --top 10
"""
from __future__ import annotations

from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from src.utils.logging import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]

FEAT_PATH  = ROOT / "data/processed/fma_small_feats_v2.parquet"
INDEX_PATH = ROOT / "data/processed/fma_small_index.parquet"
MODEL_PATH = ROOT / "artifacts/knn_audio_v2.joblib"


def _load_features(feat_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load features and return (df, X, track_ids)."""
    df = pd.read_parquet(feat_path)
    # Defensive: drop any NaNs/None features just in case
    df = df[~df["feature"].isna()].copy()
    X = np.stack(df["feature"].values)            # (n_tracks, 26)
    track_ids = df["track_id"].astype(int).values
    return df, X, track_ids


def _fit_transform(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Z-score per feature dimension, then L2-normalize rows."""
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    Xnorm = normalize(Xz, norm="l2")
    return Xnorm, scaler


def _transform(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    Xz = scaler.transform(X)
    return normalize(Xz, norm="l2")


def build_model(
    k: int = 51,
    feat_path: Path = FEAT_PATH,
    model_path: Path = MODEL_PATH,
) -> None:
    """Fit KNN (cosine) on MFCC features and persist scaler + model + track_ids."""
    log.info("Loading features from %s", feat_path)
    df, X, track_ids = _load_features(feat_path)
    log.info("Loaded %d rows; feature shape=%s", len(df), X.shape)
    log.info("Feature matrix shape: %s; unique tracks: %d", X.shape, len(track_ids))

    Xnorm, scaler = _fit_transform(X)

    # n_neighbors = k (50) + 1 (the item itself)
    model = NearestNeighbors(n_neighbors=k, metric="cosine")
    model.fit(Xnorm)
    log.info("KNN fit complete (metric=cosine, n_neighbors=%d).", k)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "scaler": scaler,
        "track_ids": track_ids,        # order aligned with fitted Xnorm
        "feature_dim": X.shape[1],
        "metric": "cosine",
    }
    joblib.dump(payload, model_path)
    log.info("Saved KNN model artifacts → %s", model_path)


def _matrix_in_saved_order(
    feat_path: Path, saved_track_ids: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Rebuild X in the same track order used at fit time, regardless of
    current DataFrame row ordering.
    """
    df = pd.read_parquet(feat_path)[["track_id", "feature"]]
    df["track_id"] = df["track_id"].astype(int)
    # Index by track_id, then take rows in saved order
    lut = df.set_index("track_id")["feature"].to_dict()
    # Some tracks might have been missing/failed; assert coverage:
    missing = [tid for tid in saved_track_ids if tid not in lut]
    if missing:
        raise RuntimeError(f"{len(missing)} track_ids missing from features at query time, e.g. {missing[:5]}")
    X = np.stack([lut[int(tid)] for tid in saved_track_ids])
    id_to_idx = {int(tid): i for i, tid in enumerate(saved_track_ids)}
    return X, id_to_idx


def recommend(
    seed_track_id: int,
    k: int = 50,
    feat_path: Path = FEAT_PATH,
    model_path: Path = MODEL_PATH,
) -> tuple[list[int], list[float]]:
    """Return (recommended_track_ids, distances) for a given seed track."""
    payload = joblib.load(model_path)
    model: NearestNeighbors = payload["model"]
    scaler: StandardScaler = payload["scaler"]
    track_ids: np.ndarray = payload["track_ids"]

    X_saved_order, id_to_idx = _matrix_in_saved_order(feat_path, track_ids)
    Xq = _transform(X_saved_order, scaler)

    if seed_track_id not in id_to_idx:
        raise ValueError(f"Track ID {seed_track_id} not found in model.")
    seed_idx = id_to_idx[seed_track_id]

    dists, idxs = model.kneighbors(Xq[seed_idx].reshape(1, -1), n_neighbors=k + 1)
    # Drop self if present
    neighbors = [(i, d) for i, d in zip(idxs[0].tolist(), dists[0].tolist()) if i != seed_idx][:k]
    rec_ids = [int(track_ids[i]) for i, _ in neighbors]
    rec_dists = [float(d) for _, d in neighbors]
    return rec_ids, rec_dists


def display(seed_track_id: int, top: int = 10, index_path: Path = INDEX_PATH) -> None:
    """Pretty-print top-N recs with metadata."""
    rec_ids, rec_dists = recommend(seed_track_id, k=top)
    idx_df = pd.read_parquet(index_path)
    cols = ["track_id", "artist_name", "track_title", "album_title", "track_genre_top"]

    order = {tid: i for i, tid in enumerate(rec_ids)}
    sub = idx_df[idx_df["track_id"].isin(rec_ids)][cols].copy()
    sub["order"] = sub["track_id"].map(order)
    sub = sub.sort_values("order").drop(columns="order")
    sub["distance"] = rec_dists  # cosine distance; smaller = more similar

    print(f"\nTop {top} recommendations for track_id {seed_track_id} (smaller distance = more similar):\n")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN MFCC recommender")
    parser.add_argument("--build", action="store_true", help="Fit and save the KNN model")
    parser.add_argument("--query", type=int, help="Seed track_id to query recommendations for")
    parser.add_argument("--top", type=int, default=10, help="How many recs to print in --query mode")
    args = parser.parse_args()

    if args.build:
        build_model()
    if args.query is not None:
        display(args.query, top=args.top)
    if not args.build and args.query is None:
        parser.print_help()