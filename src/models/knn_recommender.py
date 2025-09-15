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
META_PATH  = ROOT / "data/processed/fma_small_meta.parquet"
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


def recommend_candidates(
    seed_track_id: int,
    candidate_k: int = 200,
    feat_path: Path = FEAT_PATH,
    model_path: Path = MODEL_PATH,
) -> tuple[list[int], list[float]]:
    """Get a larger candidate set (ids, distances) before re-ranking."""
    return recommend(seed_track_id, k=candidate_k, feat_path=feat_path, model_path=model_path)


def recommend_reranked(
    seed_track_id: int,
    top: int = 50,
    candidate_k: int = 200,
    genre_weight: float = 0.15,   # +0.15 if same top-genre
    artist_penalty: float = 0.05, # −0.05 per prior occurrence of the same artist
    index_path: Path = INDEX_PATH,
) -> tuple[list[int], list[float]]:
    """Rerank KNN candidates by (similarity + genre bonus − artist penalty)."""
    rec_ids, dists = recommend_candidates(seed_track_id, candidate_k)
    idx_df = pd.read_parquet(index_path)[["track_id","artist_name","track_genre_top"]]
    idx_df["track_id"] = idx_df["track_id"].astype(int)

    # Seed metadata
    seed_meta = idx_df[idx_df["track_id"] == int(seed_track_id)]
    seed_genre = seed_meta["track_genre_top"].iloc[0] if not seed_meta.empty else None

    # Lookup metadata for candidates
    meta = idx_df[idx_df["track_id"].isin(rec_ids)].set_index("track_id").to_dict(orient="index")

    # Convert cosine distance -> similarity (higher is better)
    sim = 1.0 - np.clip(np.array(dists, dtype=float), 0.0, 1.0)

    scores = []
    artist_counts: dict[str, int] = {}
    for tid, s in zip(rec_ids, sim):
        info = meta.get(int(tid), {})
        artist = info.get("artist_name", "")
        genre  = info.get("track_genre_top", None)

        g_bonus = genre_weight if (seed_genre is not None and genre == seed_genre) else 0.0
        a_pen   = artist_penalty * artist_counts.get(artist, 0)

        score = float(s + g_bonus - a_pen)
        scores.append(score)
        artist_counts[artist] = artist_counts.get(artist, 0) + 1

    # Top-N by score, keeping candidate order only for tie-break
    order = np.argsort(-np.array(scores))[:top]
    final_ids    = [int(rec_ids[i]) for i in order]
    final_scores = [float(scores[i]) for i in order]
    return final_ids, final_scores


def display_reranked(seed_track_id: int, top: int = 10, index_path: Path = INDEX_PATH) -> None:
    rec_ids, scores = recommend_reranked(seed_track_id, top=top, index_path=index_path)
    idx_df = pd.read_parquet(index_path)
    cols = ["track_id","artist_name","track_title","album_title","track_genre_top"]

    order = {tid:i for i, tid in enumerate(rec_ids)}
    sub = idx_df[idx_df["track_id"].isin(rec_ids)][cols].copy()
    sub["order"] = sub["track_id"].map(order)
    sub = sub.sort_values("order").drop(columns="order")
    sub["score"] = [scores[order[tid]] for tid in sub["track_id"]]
    print(f"\nTop {top} (reranked) for seed {seed_track_id}:\n")
    print(sub.to_string(index=False))
    
def _load_meta() -> pd.DataFrame:
    meta = pd.read_parquet(META_PATH)
    # normalize types
    meta["track_id"] = meta["track_id"].astype(int)
    meta["genres_all"] = meta["genres_all"].apply(lambda x: x if isinstance(x, list) else [])
    return meta.set_index("track_id")

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

def recommend_hybrid(
    seed_track_id: int,
    top: int = 50,
    candidate_k: int = 200,
    w_audio: float = 1.0,     # base similarity from KNN
    w_genre: float = 0.35,    # Jaccard of genres_all
    w_tempo: float = 0.15,    # closeness of BPM
    w_year: float = 0.10,     # closeness of release year
    artist_penalty: float = 0.05,   # diversity
) -> tuple[list[int], list[float]]:
    # 1) get audio-based candidates
    rec_ids, dists = recommend(seed_track_id, k=candidate_k, feat_path=FEAT_PATH, model_path=MODEL_PATH)
    audio_sim = 1.0 - np.clip(np.array(dists), 0.0, 1.0)

    # 2) load metadata
    meta = _load_meta()
    if seed_track_id not in meta.index:
        raise ValueError("Seed not found in metadata")
    seed = meta.loc[seed_track_id]
    seed_gen = set(seed.get("genres_all", []))
    seed_tempo = seed.get("tempo")
    seed_year  = seed.get("year")

    # precompute candidate meta
    pop = []
    for tid in rec_ids:
        m = meta.loc[tid] if tid in meta.index else None
        p = float(np.log1p(m.get("listens", 0))) if m is not None else 0.0
        pop.append(p)
    # min-max normalize popularity over the candidate set (kept modest)
    pop = np.array(pop, dtype=float)
    rng = np.ptp(pop)
    pop = (pop - pop.min()) / (rng + 1e-9)
    w_pop = 0.05  # small tie-breaker

    scores, artists = [], []
    for i, tid in enumerate(rec_ids):
        m = meta.loc[tid] if tid in meta.index else None
        g = _jaccard(seed_gen, set(m.get("genres_all", []))) if m is not None else 0.0
        # tempo/year similarity with soft kernels if both present
        def soft(x, y, scale):
            if x is None or pd.isna(x) or y is None or pd.isna(y): return 0.0
            return float(np.exp(-abs(float(x) - float(y)) / scale))
        t = soft(seed_tempo, (m.get("tempo") if m is not None else None), scale=30.0)  # ~±30 BPM tolerance
        y = soft(seed_year,  (m.get("year")  if m is not None else None),  scale=10.0) # decade-ish

        artist = (m.get("artist_name") if m is not None else "")
        artists.append(artist)

        score = (w_audio * float(audio_sim[i])
                 + w_genre * g
                 + w_tempo * t
                 + w_year  * y
                 + w_pop   * float(pop[i]))
        scores.append(score)

    # artist diversity (greedy downweight repeats)
    final = []
    seen_counts: dict[str, int] = {}
    order = np.argsort(-np.array(scores))  # high → low
    for j in order:
        tid, sc, art = rec_ids[j], float(scores[j]), artists[j]
        sc -= artist_penalty * seen_counts.get(art, 0)
        final.append((tid, sc))
        seen_counts[art] = seen_counts.get(art, 0) + 1
        if len(final) == top:
            break

    final.sort(key=lambda x: -x[1])
    return [tid for tid, _ in final], [sc for _, sc in final]

def display_hybrid(seed_track_id: int, top: int = 10, index_path: Path = INDEX_PATH, w_audio=1.0, w_genre=0.35, w_tempo=0.15, w_year=0.10, artist_penalty=0.05, candidate_k=200) -> None:
    rec_ids, scores = recommend_hybrid(seed_track_id, top=top, candidate_k=candidate_k,
                                       w_audio=w_audio, w_genre=w_genre, w_tempo=w_tempo,
                                       w_year=w_year, artist_penalty=artist_penalty)
    idx_df = pd.read_parquet(index_path)
    cols = ["track_id","artist_name","track_title","album_title","genre_top"]
    if "track_genre_top" in idx_df.columns:
        cols[-1] = "track_genre_top"
    order = {tid:i for i, tid in enumerate(rec_ids)}
    sub = idx_df[idx_df["track_id"].isin(rec_ids)][["track_id","artist_name","track_title","album_title", cols[-1]]].copy()
    sub["order"] = sub["track_id"].map(order)
    sub = sub.sort_values("order").drop(columns="order")
    sub["score"] = [scores[order[tid]] for tid in sub["track_id"]]
    print(f"\nTop {top} (hybrid: audio+meta) for seed {seed_track_id}:\n")
    print(sub.to_string(index=False))

def _features_for(tids: list[int], feat_path: Path, scaler: StandardScaler) -> np.ndarray:
    """Return normalized feature rows for given track_ids (order-preserving)."""
    df = pd.read_parquet(feat_path)[["track_id","feature"]].copy()
    lut = df.set_index("track_id")["feature"].to_dict()
    X = np.stack([lut[int(t)] for t in tids])
    X = scaler.transform(X)
    return normalize(X, norm="l2")

def build_playlist(
    seed_track_id: int,
    n: int = 50,
    candidate_k: int = 500,
    lambda_mmr: float = 0.3,            # 0=diversity only, 1=similarity only
    max_per_artist: int = 2,
    max_per_album: int = 3,
    feat_path: Path = FEAT_PATH,
    model_path: Path = MODEL_PATH,
    index_path: Path = INDEX_PATH,
) -> list[int]:
    """Return a 50-track playlist using hybrid scores + MMR with artist/album caps."""
    # 1) candidates & base scores (hybrid)
    cand_ids, base_scores = recommend_hybrid(seed_track_id, top=candidate_k)
    base_scores = np.array(base_scores, dtype=float)

    # 2) audio similarity matrix among candidates (cosine on normalized features)
    payload = joblib.load(model_path)
    scaler: StandardScaler = payload["scaler"]
    Xc = _features_for(cand_ids, feat_path, scaler)         # (K, D)
    # pairwise cosine sim via dot product (K x K)
    sim = (Xc @ Xc.T).astype(float)

    # 3) metadata for caps
    meta = pd.read_parquet(index_path)[["track_id","artist_name","album_title"]].copy()
    meta["track_id"] = meta["track_id"].astype(int)
    m = meta.set_index("track_id").to_dict(orient="index")

    selected: list[int] = []
    artist_ct: dict[str,int] = {}
    album_ct: dict[str,int] = {}

    # greedy MMR
    taken = np.zeros(len(cand_ids), dtype=bool)
    while len(selected) < n and (~taken).any():
        best_idx = None
        best_score = -1e9
        for i, tid in enumerate(cand_ids):
            if taken[i]:
                continue
            art = (m.get(tid, {}).get("artist_name") or "")
            alb = (m.get(tid, {}).get("album_title") or "")
            if artist_ct.get(art, 0) >= max_per_artist:
                continue
            if album_ct.get(alb, 0) >= max_per_album:
                continue

            # diversity term: max similarity to any already selected
            if not selected:
                div_pen = 0.0
            else:
                sel_idxs = [cand_ids.index(t) for t in selected]
                div_pen = float(sim[i, sel_idxs].max())

            mmr = lambda_mmr * float(base_scores[i]) - (1.0 - lambda_mmr) * div_pen
            if mmr > best_score:
                best_score, best_idx = mmr, i

        if best_idx is None:
            break  # no candidate passes caps

        taken[best_idx] = True
        chosen_tid = cand_ids[best_idx]
        selected.append(chosen_tid)
        a = (m.get(chosen_tid, {}).get("artist_name") or "")
        al = (m.get(chosen_tid, {}).get("album_title") or "")
        artist_ct[a] = artist_ct.get(a, 0) + 1
        album_ct[al] = album_ct.get(al, 0) + 1

    return selected

def display_playlist(seed_track_id: int, n: int = 50, index_path: Path = INDEX_PATH) -> None:
    tids = build_playlist(seed_track_id, n=n)
    idx_df = pd.read_parquet(index_path)
    cols = ["track_id","artist_name","track_title","album_title","track_genre_top"]
    order = {tid:i for i, tid in enumerate(tids)}
    sub = idx_df[idx_df["track_id"].isin(tids)][cols].copy()
    sub["order"] = sub["track_id"].map(order)
    sub = sub.sort_values("order").drop(columns="order")
    print(f"\nMMR playlist ({n}) for seed {seed_track_id}:\n")
    print(sub.to_string(index=False))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN MFCC recommender")
    parser.add_argument("--build", action="store_true", help="Fit and save the KNN model")
    parser.add_argument("--query", type=int, help="Seed track_id to query recommendations for")
    parser.add_argument("--query-hybrid", type=int, help="Seed track_id to query with audio+metadata hybrid rerank")
    parser.add_argument("--top", type=int, default=10, help="How many recs to print in --query mode")
    parser.add_argument("--query-rerank", type=int, help="Seed track_id to query with genre/diversity re-rank")
    parser.add_argument("--genre-weight", type=float, default=0.15)
    parser.add_argument("--w-audio", type=float, default=1.0)
    parser.add_argument("--w-genre", type=float, default=0.35)
    parser.add_argument("--w-tempo", type=float, default=0.15)
    parser.add_argument("--w-year",  type=float, default=0.10)
    parser.add_argument("--artist-penalty", type=float, default=0.05)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--playlist", type=int, help="Seed track_id to build a 50-track MMR playlist for")
    parser.add_argument("--n", type=int, default=50, help="Playlist length")
    args = parser.parse_args()

    handled = False
    if args.build:
        build_model(); handled = True
    if args.query is not None:
        display(args.query, top=args.top); handled = True
    if getattr(args, "query_rerank", None) is not None:
        display_reranked(args.query_rerank, top=args.top); handled = True
    if getattr(args, "query_hybrid", None) is not None:
        display_hybrid(args.query_hybrid, top=args.top,
                    w_audio=args.w_audio, w_genre=args.w_genre,
                    w_tempo=args.w_tempo, w_year=args.w_year,
                    artist_penalty=args.artist_penalty, candidate_k=args.candidate_k)
        handled = True
    if getattr(args, "playlist", None) is not None:
        display_playlist(args.playlist, n=args.n); handled = True
    if not handled:
        parser.print_help()