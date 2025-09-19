#!/usr/bin/env python3
"""
HNSW-based Approximate Nearest Neighbors Recommender

High-performance music recommendation using hnswlib (Hierarchical Navigable Small World).
Designed to replace exact KNN for FMA Large scalability (100K+ tracks).

Key features:
- Sub-second query times (vs 6+ seconds for exact KNN)  
- Same API as existing KNN system for seamless integration
- Memory-efficient indices for production deployment
- Accuracy benchmarking vs exact KNN
- Simple and reliable hnswlib implementation
"""

import argparse
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import hnswlib
from sklearn.preprocessing import StandardScaler

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.knn_recommender import (
    _load_features, _load_meta, recommend_hybrid as knn_recommend_hybrid,
    build_playlist as knn_build_playlist, display as knn_display,
    display_hybrid as knn_display_hybrid, display_playlist as knn_display_playlist,
    recommend as knn_recommend
)
from src.utils.logging import get_logger

log = get_logger(__name__)

# Default paths (matching existing system)
ROOT = Path(__file__).resolve().parents[2]
FEAT_PATH = ROOT / "data/processed/fma_small_feats_v2.parquet"
INDEX_PATH = ROOT / "data/processed/fma_small_index.parquet"
META_PATH = ROOT / "data/processed/fma_small_meta.parquet"
MODEL_PATH = ROOT / "artifacts/knn_audio_v2.joblib"
HNSW_PATH = ROOT / "artifacts/hnsw_audio_v1.bin"


class HNSWRecommender:
    """HNSW-based approximate nearest neighbors for music recommendations."""
    
    def __init__(self, 
                 feat_path: Path = FEAT_PATH,
                 model_path: Path = MODEL_PATH,
                 hnsw_path: Path = HNSW_PATH):
        self.feat_path = feat_path
        self.model_path = model_path
        self.hnsw_path = hnsw_path
        
        # Loaded components
        self.features_df = None
        self.scaler = None
        self.track_ids = None
        self.hnsw_index = None
        self.track_id_to_hnsw_idx = None
        self.hnsw_idx_to_track_id = None
        
    def load_data(self) -> None:
        """Load features and preprocessing components."""
        log.info("Loading features and scaler...")
        self.features_df, _, self.track_ids = _load_features(self.feat_path)
        
        # Load scaler from existing KNN model
        if self.model_path.exists():
            payload = joblib.load(self.model_path)
            self.scaler = payload["scaler"]
            log.info(f"Loaded scaler from {self.model_path}")
        else:
            log.warning(f"No existing model at {self.model_path}, will need to fit scaler")
            
    def _create_mappings(self) -> None:
        """Create bidirectional mappings between track_ids and HNSW indices."""
        # HNSW requires sequential integer indices [0, 1, 2, ...]
        # But our track_ids are arbitrary [2, 5, 10, 140, ...]
        
        self.track_id_to_hnsw_idx = {tid: i for i, tid in enumerate(self.track_ids)}
        self.hnsw_idx_to_track_id = {i: tid for i, tid in enumerate(self.track_ids)}
        
        log.info(f"Created mappings for {len(self.track_ids)} tracks")
        
    def build_index(self, 
                   ef_construction: int = 200,
                   M: int = 16,
                   max_elements: int = None) -> None:
        """
        Build and save HNSW index.
        
        Args:
            ef_construction: Size of dynamic candidate list (higher = better quality, slower)
            M: Number of bidirectional links for new elements (higher = better recall, more memory)
            max_elements: Maximum number of elements (auto-detected if None)
        """
        if self.features_df is None:
            self.load_data()
            
        self._create_mappings()
        
        # Get feature dimension and count
        feature_dim = len(self.features_df.iloc[0]["feature"])
        n_elements = len(self.track_ids)
        
        if max_elements is None:
            max_elements = n_elements
            
        log.info(f"Building HNSW index: {n_elements} elements, {feature_dim}D, "
                f"ef_construction={ef_construction}, M={M}")
        
        # Initialize HNSW index (cosine similarity)
        self.hnsw_index = hnswlib.Index(space='cosine', dim=feature_dim)
        self.hnsw_index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        
        # Prepare all feature vectors
        build_start = time.time()
        feature_matrix = np.zeros((n_elements, feature_dim), dtype=np.float32)
        
        for i, track_id in enumerate(self.track_ids):
            # Get normalized feature vector from DataFrame row
            feature_vector = self.features_df.iloc[i]["feature"]
            
            # Apply same scaling as KNN system
            if self.scaler is not None:
                feature_vector = self.scaler.transform([feature_vector])[0]
                
            feature_matrix[i] = feature_vector.astype(np.float32)
            
        # Add all vectors to HNSW index at once (more efficient)
        log.info("Adding vectors to HNSW index...")
        hnsw_indices = np.arange(n_elements)
        self.hnsw_index.add_items(feature_matrix, hnsw_indices)
        
        build_time = time.time() - build_start
        
        # Save index
        self.hnsw_path.parent.mkdir(parents=True, exist_ok=True)
        self.hnsw_index.save_index(str(self.hnsw_path))
        
        log.info(f"Built and saved HNSW index in {build_time:.2f}s to {self.hnsw_path}")
        
        # Save metadata for loading
        metadata = {
            "ef_construction": ef_construction,
            "M": M,
            "feature_dim": feature_dim,
            "n_tracks": len(self.track_ids),
            "track_ids": self.track_ids,
            "track_id_to_hnsw_idx": self.track_id_to_hnsw_idx,
            "hnsw_idx_to_track_id": self.hnsw_idx_to_track_id,
            "feat_path": str(self.feat_path),
            "model_path": str(self.model_path)
        }
        
        metadata_path = self.hnsw_path.with_suffix('.joblib')
        joblib.dump(metadata, metadata_path)
        log.info(f"Saved index metadata to {metadata_path}")
        
    def load_index(self) -> None:
        """Load pre-built HNSW index and metadata."""
        if not self.hnsw_path.exists():
            raise FileNotFoundError(f"HNSW index not found at {self.hnsw_path}. Run build_index() first.")
            
        metadata_path = self.hnsw_path.with_suffix('.joblib')
        if not metadata_path.exists():
            raise FileNotFoundError(f"HNSW metadata not found at {metadata_path}")
            
        # Load metadata
        metadata = joblib.load(metadata_path)
        self.track_ids = metadata["track_ids"]
        self.track_id_to_hnsw_idx = metadata["track_id_to_hnsw_idx"]
        self.hnsw_idx_to_track_id = metadata["hnsw_idx_to_track_id"]
        
        # Load HNSW index
        feature_dim = metadata["feature_dim"]
        self.hnsw_index = hnswlib.Index(space='cosine', dim=feature_dim)
        self.hnsw_index.load_index(str(self.hnsw_path))
        
        log.info(f"Loaded HNSW index: {metadata['n_tracks']} tracks, "
                f"ef_construction={metadata['ef_construction']}, M={metadata['M']}, {feature_dim}D")
        
        # Load scaler
        if self.model_path.exists():
            payload = joblib.load(self.model_path)
            self.scaler = payload["scaler"]
        else:
            log.warning("No scaler found, features may not be properly normalized")
            
    def recommend(self, 
                 seed_track_id: int, 
                 top: int = 50,
                 ef: int = None) -> Tuple[List[int], List[float]]:
        """
        Get approximate nearest neighbors for a seed track.
        
        Args:
            seed_track_id: ID of seed track
            top: Number of recommendations to return
            ef: Search parameter (higher = better accuracy, slower)
            
        Returns:
            (recommended_track_ids, distances)
        """
        if self.hnsw_index is None:
            self.load_index()
            
        if seed_track_id not in self.track_id_to_hnsw_idx:
            raise ValueError(f"Track ID {seed_track_id} not found in index")
            
        # Set search parameter (default to good balance)
        if ef is None:
            ef = max(top * 2, 50)  # Rule of thumb: ef >= k
        self.hnsw_index.set_ef(ef)
            
        # Get HNSW index for seed track
        seed_hnsw_idx = self.track_id_to_hnsw_idx[seed_track_id]
        
        # Get the seed track's feature vector
        if self.features_df is None:
            self.load_data()
        
        # Find the row for this track_id 
        seed_row_idx = list(self.track_ids).index(seed_track_id)
        seed_feature = self.features_df.iloc[seed_row_idx]["feature"]
        
        if self.scaler is not None:
            seed_feature = self.scaler.transform([seed_feature])[0]
        
        # Query HNSW index
        query_start = time.time()
        neighbor_indices, distances = self.hnsw_index.knn_query(
            seed_feature.astype(np.float32), 
            k=top + 1  # +1 because seed might be included
        )
        query_time = time.time() - query_start
        
        # Convert back to track IDs and remove seed if present
        recommended_track_ids = []
        final_distances = []
        
        for hnsw_idx, distance in zip(neighbor_indices[0], distances[0]):
            track_id = self.hnsw_idx_to_track_id[hnsw_idx]
            if track_id != seed_track_id:  # Exclude seed track
                recommended_track_ids.append(track_id)
                final_distances.append(distance)
                
        # Truncate to requested size
        recommended_track_ids = recommended_track_ids[:top]
        final_distances = final_distances[:top]
        
        log.debug(f"HNSW query took {query_time:.4f}s for {top} recommendations")
        
        return recommended_track_ids, final_distances
    
    def benchmark_vs_knn(self, 
                        test_seeds: List[int] = None,
                        top: int = 50) -> Dict[str, Any]:
        """
        Benchmark HNSW vs exact KNN for accuracy and performance.
        
        Args:
            test_seeds: List of seed tracks to test (random selection if None)
            top: Number of recommendations per seed
            
        Returns:
            Benchmark results dictionary
        """
        if self.hnsw_index is None:
            self.load_index()
            
        # Select test seeds
        if test_seeds is None:
            available_seeds = [tid for tid in self.track_ids if tid in self.track_id_to_hnsw_idx]
            test_seeds = np.random.choice(available_seeds, min(10, len(available_seeds)), replace=False)
            
        log.info(f"Benchmarking HNSW vs KNN on {len(test_seeds)} seeds...")
        
        results = {
            "test_seeds": test_seeds,
            "top": top,
            "hnsw_times": [],
            "knn_times": [],
            "accuracy_scores": [],
            "overlap_counts": []
        }
        
        for seed in test_seeds:
            log.debug(f"Testing seed {seed}...")
            
            # HNSW recommendations
            hnsw_start = time.time()
            hnsw_recs, hnsw_dists = self.recommend(seed, top)
            hnsw_time = time.time() - hnsw_start
            
            # KNN recommendations (using pure audio, not hybrid)
            knn_start = time.time()
            try:
                knn_recs, knn_dists = knn_recommend(seed, top)
                knn_time = time.time() - knn_start
            except Exception as e:
                log.warning(f"KNN failed for seed {seed}: {e}")
                continue
                
            # Calculate accuracy metrics
            hnsw_set = set(hnsw_recs)
            knn_set = set(knn_recs)
            
            overlap = len(hnsw_set & knn_set)
            accuracy = overlap / top if top > 0 else 0
            
            results["hnsw_times"].append(hnsw_time)
            results["knn_times"].append(knn_time)
            results["accuracy_scores"].append(accuracy)
            results["overlap_counts"].append(overlap)
            
        # Calculate summary statistics
        if results["hnsw_times"]:
            results["summary"] = {
                "avg_hnsw_time": np.mean(results["hnsw_times"]),
                "avg_knn_time": np.mean(results["knn_times"]),
                "speedup_factor": np.mean(results["knn_times"]) / np.mean(results["hnsw_times"]),
                "avg_accuracy": np.mean(results["accuracy_scores"]),
                "min_accuracy": np.min(results["accuracy_scores"]),
                "max_accuracy": np.max(results["accuracy_scores"]),
                "avg_overlap": np.mean(results["overlap_counts"])
            }
            
        return results


def build_hnsw_index(feat_path: Path = FEAT_PATH,
                     model_path: Path = MODEL_PATH,
                     hnsw_path: Path = HNSW_PATH,
                     ef_construction: int = 200,
                     M: int = 16) -> None:
    """Build HNSW index for the dataset."""
    recommender = HNSWRecommender(feat_path, model_path, hnsw_path)
    recommender.build_index(ef_construction=ef_construction, M=M)


def benchmark_hnsw(feat_path: Path = FEAT_PATH,
                   model_path: Path = MODEL_PATH,
                   hnsw_path: Path = HNSW_PATH,
                   n_seeds: int = 10,
                   top: int = 50) -> None:
    """Run accuracy and performance benchmark."""
    recommender = HNSWRecommender(feat_path, model_path, hnsw_path)
    
    # Select random test seeds
    recommender.load_data()
    available_seeds = [tid for tid in recommender.track_ids]
    test_seeds = np.random.choice(available_seeds, min(n_seeds, len(available_seeds)), replace=False)
    
    results = recommender.benchmark_vs_knn(test_seeds, top)
    
    # Display results
    if "summary" in results:
        summary = results["summary"]
        print("\n🎯 HNSW vs KNN BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Test Seeds: {len(test_seeds)}")
        print(f"Recommendations per Seed: {top}")
        print()
        print("⚡ PERFORMANCE:")
        print(f"  HNSW avg time: {summary['avg_hnsw_time']:.4f}s")
        print(f"  KNN avg time:  {summary['avg_knn_time']:.4f}s")
        print(f"  Speedup:       {summary['speedup_factor']:.1f}x faster")
        print()
        print("🎯 ACCURACY:")
        print(f"  Average accuracy: {summary['avg_accuracy']:.1%}")
        print(f"  Min accuracy:     {summary['min_accuracy']:.1%}")
        print(f"  Max accuracy:     {summary['max_accuracy']:.1%}")
        print(f"  Avg overlap:      {summary['avg_overlap']:.1f}/{top} tracks")
        print()
        
        # Assess results
        if summary['avg_accuracy'] >= 0.80:
            print("✅ EXCELLENT accuracy (≥80% overlap with exact KNN)")
        elif summary['avg_accuracy'] >= 0.70:
            print("⚠️  GOOD accuracy (≥70% overlap with exact KNN)")
        else:
            print("🚨 LOW accuracy (<70% overlap, consider tuning)")
            
        if summary['speedup_factor'] >= 5:
            print("🚀 EXCELLENT speedup (≥5x faster than KNN)")
        elif summary['speedup_factor'] >= 2:
            print("⚡ GOOD speedup (≥2x faster than KNN)")
        else:
            print("🐌 MINIMAL speedup (<2x, check implementation)")


# CLI Interface (matching existing KNN system)
def main():
    """CLI interface for HNSW recommender."""
    parser = argparse.ArgumentParser(description="HNSW-based Music Recommender")
    
    # Operations
    parser.add_argument("--build", action="store_true", help="Build HNSW index")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark vs KNN")
    parser.add_argument("--query", type=int, help="Get recommendations for track ID")
    
    # Parameters
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW ef_construction (default: 200)")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter (default: 16)")
    parser.add_argument("--top", type=int, default=50, help="Number of recommendations (default: 50)")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of test seeds for benchmark (default: 10)")
    
    # Paths (configurable like existing system)
    parser.add_argument("--feat-path", type=Path, default=FEAT_PATH, help=f"Features file (default: {FEAT_PATH})")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help=f"KNN model file (default: {MODEL_PATH})")
    parser.add_argument("--hnsw-path", type=Path, default=HNSW_PATH, help=f"HNSW index file (default: {HNSW_PATH})")
    
    args = parser.parse_args()
    
    if args.build:
        print("🔨 Building HNSW index...")
        build_hnsw_index(args.feat_path, args.model_path, args.hnsw_path, 
                        args.ef_construction, args.M)
        print("✅ HNSW index built successfully!")
        
    elif args.benchmark:
        print("📊 Running HNSW vs KNN benchmark...")
        benchmark_hnsw(args.feat_path, args.model_path, args.hnsw_path, 
                      args.n_seeds, args.top)
        
    elif args.query is not None:
        print(f"🎵 Getting HNSW recommendations for track {args.query}...")
        recommender = HNSWRecommender(args.feat_path, args.model_path, args.hnsw_path)
        recs, dists = recommender.recommend(args.query, args.top)
        
        print(f"\nTop {len(recs)} recommendations:")
        for i, (track_id, dist) in enumerate(zip(recs, dists), 1):
            print(f"{i:2d}. Track {track_id} (distance: {dist:.3f})")
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
