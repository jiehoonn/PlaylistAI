from __future__ import annotations
from pathlib import Path
import ast
import pandas as pd
from src.config import settings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/fma_small_meta.parquet"

def _parse_list(x):
    if isinstance(x, list): return x
    try: return list(ast.literal_eval(x))
    except Exception: return []

def build_small_metadata() -> Path:
    meta_dir = settings.FMA_METADATA_DIR
    tracks_csv   = meta_dir / "tracks.csv"
    echonest_csv = meta_dir / "echonest.csv"
    genres_csv   = meta_dir / "genres.csv"

    df = pd.read_csv(tracks_csv, header=[0, 1], index_col=0, low_memory=False)

    # keep only the "small" subset
    small = df[df[("set", "subset")] == "small"].copy()
    small.index = small.index.astype(int)

    meta = pd.DataFrame(index=small.index)
    meta["track_id"]     = small.index
    meta["track_title"]  = small[("track", "title")]
    meta["artist_name"]  = small[("artist", "name")]
    meta["album_title"]  = small[("album", "title")]
    meta["genre_top"]    = small[("track", "genre_top")]
    meta["genres_all"]   = small[("track", "genres_all")].map(_parse_list)
    meta["duration"]     = pd.to_numeric(small[("track", "duration")], errors="coerce")
    meta["listens"]      = pd.to_numeric(small[("track", "listens")], errors="coerce")
    meta["favorites"]    = pd.to_numeric(small[("track", "favorites")], errors="coerce")
    meta["date_released"]= pd.to_datetime(small[("album", "date_released")], errors="coerce")
    meta["year"]         = meta["date_released"].dt.year

    # map genre ids to names for convenience
    try:
        g = pd.read_csv(genres_csv, index_col=0)
        id2name = g["title"].to_dict()
        meta["genres_all_names"] = meta["genres_all"].map(lambda ids: [id2name.get(int(i)) for i in ids if int(i) in id2name])
    except Exception:
        meta["genres_all_names"] = [[] for _ in range(len(meta))]

    # --- EchoNest join (handles 2-row header) ---
    try:
        en = pd.read_csv(echonest_csv, header=[0, 1], index_col=0, low_memory=False)
        # Many FMA dumps have top level "echonest" and second level the actual field names
        if isinstance(en.columns, pd.MultiIndex):
            # take the lower level (real names) and normalize
            en.columns = (
                en.columns.get_level_values(1)
                .str.strip().str.lower()
                .str.replace(r"[^a-z0-9]+", "_", regex=True)
                .str.replace(r"_+$", "", regex=True)
            )
        else:
            # fallback if your file is already single-level
            en.columns = (
                en.columns.str.strip().str.lower()
                .str.replace(r"[^a-z0-9]+", "_", regex=True)
                .str.replace(r"^(echonest_|audio_features_|audio_)", "", regex=True)
                .str.replace(r"_+$", "", regex=True)
            )

        # standardize index dtypes to match meta
        en.index = pd.to_numeric(en.index, errors="coerce").astype("Int64")
        meta.index = pd.to_numeric(meta.index, errors="coerce").astype("Int64")

        wanted = [
            "tempo","key","mode","time_signature","energy","danceability","valence",
            "loudness","acousticness","instrumentalness"
        ]
        keep = [c for c in wanted if c in en.columns]
        if keep:
            meta = meta.join(en[keep], how="left")
            print("EchoNest columns joined:", sorted(keep))
        else:
            print("WARN: no matching EchoNest columns found after flattening.")
    except Exception as e:
        print(f"WARN: EchoNest join failed: {e}")

    # --- Fallback: tempo from feature parquet if EchoNest tempo missing ---
    try:
        if "tempo" not in meta.columns or meta["tempo"].isna().all():
            feats_path = ROOT / "data/processed/fma_small_feats_v2.parquet"
            if feats_path.exists():
                f = pd.read_parquet(feats_path)[["track_id", "feature"]].copy()
                f["tempo_feat"] = f["feature"].apply(
                    lambda v: float(v[-1]) if isinstance(v, (list, tuple)) and len(v) > 0 else None
                )
                f = f.drop(columns=["feature"])
                # work in column-space to avoid dtype/index alignment issues
                meta = meta.reset_index().rename(columns={"index": "track_id"})
                meta = meta.merge(f, on="track_id", how="left")
                if "tempo" not in meta.columns:
                    meta["tempo"] = None
                meta["tempo"] = meta["tempo"].fillna(meta["tempo_feat"])
                meta = meta.drop(columns=["tempo_feat"]).set_index("track_id")
                print("Backfilled 'tempo' from v2 features.")
            else:
                print(f"WARN: features parquet not found for tempo backfill: {feats_path}")
    except Exception as e:
        print(f"WARN: tempo backfill failed: {e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(OUT, index=False)
    print(f"Saved metadata for {len(meta):,} tracks → {OUT}")
    return OUT

if __name__ == "__main__":
    build_small_metadata()