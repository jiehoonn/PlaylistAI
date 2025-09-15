# PlaylistAI — Audio-based Playlist Generator (FMA)

A practical, inspectable system that takes one seed song and builds a 50-track playlist using:

- Audio features (MFCCs + spectral + tempo)
- Metadata (genre, year, popularity)
- A KNN audio similarity model
- A hybrid re-ranker (audio + meta)
- An MMR (Maximal Marginal Relevance) playlist builder to balance similarity vs. diversity

**Data source & credit:** This project uses the Free Music Archive (FMA) dataset by mdeff et al.  
Repository: https://github.com/mdeff/fma  
Please download the FMA audio subsets (e.g., fma_small) and metadata (fma_metadata) from their repo or mirrors, and review their licenses/usage terms.

> **Note:** The `fma_small` dataset contains very niche, independent tracks that are not well-known by the general public. To verify that the recommendation model is working correctly, use the `play()` function in `notebooks/06_search_top_tracks.ipynb` to listen to recommended tracks and judge similarity yourself.

---

## Contents

- [Quick Start](#quick-start)
- [Project Layout](#project-layout)
- [Environment & Configuration](#environment--configuration)
- [End-to-End Workflow](#end-to-end-workflow)
  1. [Build the track index](#1-build-the-track-index)
  2. [Extract audio features](#2-extract-audio-features)
  3. [Build metadata](#3-build-metadata)
  4. [Train the KNN audio model](#4-train-the-knn-audio-model)
  5. [Query recommendations](#5-query-recommendations)
  6. [Hybrid re-ranking (audio + metadata)](#6-hybrid-re-ranking-audio--metadata)
  7. [Build a 50-track playlist (MMR)](#7-build-a-50-track-playlist-mmr)
- [Verification & Sanity Checks](#verification--sanity-checks)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [Next Steps Plan](#next-steps-plan)

---

## Quick Start

```bash
# 0) Python & virtual env (Mac/Linux)
python3 -m venv .venv
source .venv/bin/activate
python -V   # Python 3.13.x recommended

# 1) Install deps
pip install -U pip
pip install -r requirements.txt
# If parquet errors appear: pip install pyarrow

# 2) Create .env (see below) with your FMA paths
cp .env.example .env
# Edit .env to point to your local FMA folders

# 3) Build index → extract features → build metadata → train model
python -m src.data.ingest_fma
python -m src.features.audio_features            # feature set v2 is supported; see below
python -m src.data.build_metadata
python -m src.models.knn_recommender --build

# 4) Try queries
python -m src.models.knn_recommender --query 2 --top 10
python -m src.models.knn_recommender --query-hybrid 2 --top 10

# 5) Build a 50-track playlist
python -m src.models.knn_recommender --playlist 2 --n 50
```

---

## Project Layout

```
PlaylistAI/
├─ src/
│  ├─ config/                # settings loader (.env)
│  ├─ data/
│  │  ├─ fma_paths.py        # track_id → relative audio path (xxx/xxxxxx.mp3)
│  │  ├─ ingest_fma.py       # builds index parquet (track_id + metadata + paths)
│  │  └─ build_metadata.py   # builds enriched metadata parquet (genre/year/tempo)
│  ├─ features/
│  │  └─ audio_features.py   # feature extraction (MFCCs, spectral, tempo)
│  ├─ models/
│  │  └─ knn_recommender.py  # KNN fit/query + hybrid rerank + MMR playlist
│  └─ utils/
│     └─ logging.py          # project logger
├─ data/
│  └─ processed/             # *.parquet artifacts (index, features, metadata)
├─ artifacts/                # trained models (joblib, etc.)
├─ logs/                     # runtime logs
├─ notebooks/                # optional notebooks
├─ .env                      # your local FMA paths (see below)
├─ requirements.txt
└─ .gitignore
```

---

## Environment & Configuration

Create a `.env` at repo root:

```bash
# Required (paths to your local FMA data)
FMA_SMALL_DIR=/absolute/path/to/fma_small
FMA_METADATA_DIR=/absolute/path/to/fma_metadata

# Optional: prepare for larger subsets (not required now)
FMA_SUBSET=small      # small | medium | large | full
FMA_MEDIUM_DIR=
FMA_LARGE_DIR=
FMA_FULL_DIR=
```

If you later switch to medium/large/full, the pipeline remains the same; you'll just point to different paths and re-run the steps.

---

## End-to-End Workflow

### 1) Build the track index

Reads `tracks.csv` (two-row header), filters to the chosen subset (default small), and writes a compact index with track_id, artist/album/title, genre, and absolute audio paths. Also verifies files exist on disk.

```bash
python -m src.data.ingest_fma
# Output: data/processed/fma_small_index.parquet
```

**What it contains** (columns like):

- `track_id`, `artist_name`, `track_title`, `album_title`, `track_genre_top`, `relpath`, `audio_path`

---

### 2) Extract audio features

Two ways:

**A. Module default** (simple MFCC mean/std or v2 depending on your version):

```bash
python -m src.features.audio_features
# Output (v2 in this repo): data/processed/fma_small_feats_v2.parquet
```

**B. Explicit v2 call** (if you prefer a one-liner):

```bash
python -c "from pathlib import Path; \
from src.features.audio_features import extract_features_full_v2; \
extract_features_full_v2(Path('data/processed/fma_small_index.parquet'))"
# Output: data/processed/fma_small_feats_v2.parquet
```

**What it contains:**

- One row per track_id
- `feature` = a fixed-length vector (e.g., 61 dims: MFCC stats + spectral + tempo)

You may see mpg123/audioread warnings for a few corrupted MP3s; the extractor skips those and continues.

---

### 3) Build metadata

Parses `tracks.csv` (two-row header), brings in fields like `genre_top`, `genres_all`, `year`, `listens`, `favorites`.
Also attempts to join `echonest.csv` (two-row header). If EchoNest tempo is unavailable, we backfill tempo from the features parquet.

```bash
python -m src.data.build_metadata
# Output: data/processed/fma_small_meta.parquet
```

**What it contains** (typical columns):

- `track_id`, `artist_name`, `track_title`, `album_title`, `genre_top`, `genres_all`, `year`, `listens`, `favorites`, `tempo` (backfilled if needed)

---

### 4) Train the KNN audio model

Fits a StandardScaler + NearestNeighbors (cosine) on your feature matrix and persists artifacts.

```bash
python -m src.models.knn_recommender --build
# Output artifacts: artifacts/knn_audio_v2.joblib
```

---

### 5) Query recommendations

Raw KNN neighbors by audio similarity:

```bash
python -m src.models.knn_recommender --query 2 --top 10
```

Output shows track_id, artist, title, album, top-genre, and cosine distance (smaller = more similar).

---

### 6) Hybrid re-ranking (audio + metadata)

Re-scores KNN candidates using:

- Audio similarity (from KNN)
- Genre Jaccard overlap (`genres_all`)
- Tempo proximity (BPM)
- Year proximity
- Small popularity tie-breaker
- Artist diversity penalty

Tune with flags:

```bash
python -m src.models.knn_recommender --query-hybrid 2 --top 10 \
  --w-genre 0.35 --w-tempo 0.15 --w-year 0.10 --artist-penalty 0.05 --candidate-k 200
```

If you see duplicates by the same artist, increase `--artist-penalty` or move to the playlist builder which enforces caps.

---

### 7) Build a 50-track playlist (MMR)

Greedy MMR balances "stay similar to the seed" vs. "avoid redundancy," with caps per artist/album.

```bash
python -m src.models.knn_recommender --playlist 2 --n 50
```

**Defaults** (inside code):

- `lambda_mmr = 0.3` (0 = diversity only, 1 = similarity only)
- `max_per_artist = 2`, `max_per_album = 3`

You can tune these by exposing CLI flags in `knn_recommender.py` (see Next Steps).

---

## Jupyter Notebooks

The `notebooks/` directory contains exploratory notebooks for inspecting and verifying each stage of the pipeline:

**01_inspect_fma_index.ipynb**  
Loads and inspects the track index parquet file. Shows basic stats like row count, columns, and sample data to verify the ingest step worked correctly.

**02_inspect_mfcc_features.ipynb**  
Deep dive into MFCC feature extraction. Displays feature vector lengths, shows MFCC coefficient breakdowns (energy, tilt, curvature), and includes visualization of mel spectrograms and MFCC matrices for individual tracks.

**03_verify_knn_output.ipynb**  
Tests the basic KNN recommender functionality. Runs queries, checks structural invariants (no duplicates, proper sorting), and examines genre distributions in recommendations to ensure audio similarity makes sense.

**04_inspect_mfcc_featuresv2.ipynb**  
Quick verification of the v2 feature format (61-dimensional vectors with MFCC stats + spectral features + tempo). Confirms the enhanced feature set is properly extracted.

**05_verify_metadata_knn_output.ipynb**  
Comprehensive testing of metadata integration and hybrid recommendations. Explores echonest data joining, tempo backfilling from audio features, and validates that hybrid re-ranking produces sensible tempo/year/genre alignments.

**06_search_top_tracks.ipynb**  
Utility for finding popular tracks in the dataset by listens/favorites. Includes a helper function to play tracks locally on macOS (`afplay`) for manual verification of recommendations.

These notebooks are useful for debugging issues, understanding the data flow, and manually verifying that each pipeline component produces expected results.

---

## Verification & Sanity Checks

Inspect the generated artifacts in a notebook or shell:

```python
import pandas as pd
idx  = pd.read_parquet("data/processed/fma_small_index.parquet")
feats= pd.read_parquet("data/processed/fma_small_feats_v2.parquet")
meta = pd.read_parquet("data/processed/fma_small_meta.parquet")

print("Index rows:", len(idx), "columns:", list(idx.columns)[:8])
print("Features rows:", len(feats), "example feature length:",
      len(feats['feature'].iloc[0]) if len(feats) else None)
print("Metadata rows:", len(meta), "has tempo:", 'tempo' in meta.columns,
      "tempo non-null:", meta['tempo'].notna().sum() if 'tempo' in meta.columns else 0)
```

**Optional:** play a local file on macOS to sanity-check a seed:

```python
import subprocess
row = idx[idx.track_id == 75389].iloc[0]  # Kellee Maize "City of Champions"
print(row.artist_name, "—", row.track_title, "\n", row.audio_path)
subprocess.run(["afplay", row.audio_path])
```

---

## Troubleshooting

- **Parquet ImportError:**

  ```
  ImportError: Unable to find a usable engine; tried 'pyarrow', 'fastparquet'
  ```

  Install a parquet engine:

  ```bash
  pip install pyarrow  # (recommended)
  ```

- **Feature extraction warnings** (mpg123/audioread):  
  Occasional corrupt MP3s cause messages like "Illegal Audio-MPEG-Header" or "dequantization failed."
  The extractor falls back (audioread) and skips truly bad files; you'll end up with slightly fewer rows (e.g., 7,994 of 8,000).

- **EchoNest columns missing / tempo all NaN:**  
  `echonest.csv` uses a two-row header. `build_metadata.py` flattens it; if still missing tempo, we backfill from features (last dim of v2 vector). Re-run:

  ```bash
  python -m src.data.build_metadata
  ```

- **Model paths not found:**  
  Make sure you trained the model (`--build`) and the default constants/paths in `knn_recommender.py` match your filenames. You can also add CLI overrides for `--feat-path`, `--index-path`, `--meta-path`, `--model-path`.

---

## Acknowledgements

- **FMA dataset** by mdeff et al. — https://github.com/mdeff/fma  
  Please download the audio subsets (e.g., fma_small) and metadata (fma_metadata) from their repository or provided mirrors. Respect the dataset's licenses and terms.
- **Libraries:** librosa, scikit-learn, numpy, pandas, pyarrow, tqdm, joblib.

---

## Next Steps Plan

1. **Expose MMR knobs via CLI**  
   Add `--lambda-mmr`, `--max-per-artist`, `--max-per-album` to the playlist command.  
   Default suggestion: `lambda_mmr=0.35`, `max_per_artist=1`, `max_per_album=2`.

2. **Offline evaluation script**  
   `scripts/eval_recs.py` to compute: same-genre@k, unique-artist@k, avg |tempo diff|, year spread; write CSV to `artifacts/eval/`.

3. **Unit tests (pytest)**  
   Smoke tests for: path mapping, ingest integrity, feature shape/no-nulls, model load/query, hybrid output non-empty.

4. **Metadata hardening**  
   Keep ndarray-safe tempo backfill; log coverage counts (tempo non-null, year non-null, genres_all present).

5. **Playlist export**  
   Add `src/service/export.py` with `write_m3u` and `write_csv`, plus a CLI flag (`--playlist-out`, `--out-path`).

6. **Full CLI configurability**  
   Add `--feat-path`, `--index-path`, `--meta-path`, `--model-path` to all modes (build/query/hybrid/playlist).

7. **Richer audio features (optional)**  
   Add chroma/key features (e.g., `librosa.feature.chroma_cqt`) and a small key-similarity weight in the hybrid reranker.

8. **Approximate nearest neighbors (scale-up)**  
   New `ann_recommender.py` using HNSW (hnswlib) or FAISS; same API as current KNN.

9. **Minimal API for the webapp**  
   A small FastAPI service: `/health`, `/recommend?track_id=&top=`, `/playlist?track_id=&n=`; loads artifacts once on startup.

10. **Simple UI prototype**  
    Streamlit or a tiny React client hitting FastAPI; search by track or track_id, show recs, export M3U.

11. **DX & reproducibility**  
    Makefile (or justfile) targets: ingest, features, metadata, build, query, hybrid, playlist, eval.  
    Add `.env.example` and versioned artifacts (`artifacts/knn_audio_v2_small_YYYYMMDD.joblib`).

12. **Guardrails & errors**  
    Friendly messages if seed is missing, metadata sparse, or path misconfigurations; assert tempo coverage before enabling `w_tempo`.

---

**Happy listening!** If you want, open an issue or PR to add exports, ANN, or the FastAPI wrapper, and we'll wire it up.
