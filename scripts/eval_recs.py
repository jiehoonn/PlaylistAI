"""
Offline evaluation script for recommendation system quality metrics.

Computes comprehensive metrics to assess recommendation quality:
- Genre consistency (same-genre@k)
- Artist diversity (unique-artist@k) 
- Tempo coherence (avg |tempo diff|)
- Year spread (temporal diversity)
- Overall recommendation quality scores

Usage:
    python scripts/eval_recs.py --algorithm hybrid --k 10 --num-seeds 100
    python scripts/eval_recs.py --compare-all --export-results
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.knn_recommender import (
    recommend, recommend_hybrid, build_playlist, _load_meta, _jaccard
)
from src.utils.logging import get_logger

warnings.filterwarnings('ignore')  # Suppress librosa warnings during evaluation
log = get_logger(__name__)

# Paths
EVAL_DIR = ROOT / "artifacts/eval"
INDEX_PATH = ROOT / "data/processed/fma_small_index.parquet"
META_PATH = ROOT / "data/processed/fma_small_meta.parquet"


class RecommendationEvaluator:
    """Comprehensive recommendation system evaluator."""
    
    def __init__(self):
        """Initialize evaluator with metadata."""
        self.metadata = self._load_metadata()
        self.index_data = self._load_index_data()
        log.info(f"Loaded metadata for {len(self.metadata)} tracks")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and prepare metadata for evaluation."""
        try:
            meta = _load_meta()  # Use the same function as recommender
            return meta
        except Exception as e:
            log.warning(f"Failed to load metadata via _load_meta: {e}")
            # Fallback to direct loading
            meta = pd.read_parquet(META_PATH)
            meta["track_id"] = meta["track_id"].astype(int)
            if "genres_all" in meta.columns:
                meta["genres_all"] = meta["genres_all"].apply(
                    lambda x: x if isinstance(x, list) else []
                )
            return meta.set_index("track_id")
    
    def _load_index_data(self) -> pd.DataFrame:
        """Load index data for track metadata."""
        try:
            return pd.read_parquet(INDEX_PATH)
        except Exception as e:
            log.error(f"Failed to load index data: {e}")
            return pd.DataFrame()
    
    def get_valid_seeds(self, num_seeds: int = 100, min_popularity: int = 1) -> List[int]:
        """Get valid seed tracks for evaluation."""
        valid_tracks = []
        
        # Filter tracks with sufficient metadata
        for track_id in self.metadata.index:
            meta = self.metadata.loc[track_id]
            
            # Check if track has basic metadata (relaxed criteria)
            has_genre = (pd.notna(meta.get("genre_top")) or
                        (isinstance(meta.get("genres_all"), list) and 
                         len(meta.get("genres_all", [])) > 0))
            has_tempo = pd.notna(meta.get("tempo"))
            has_listens = (pd.notna(meta.get("listens")) and 
                          meta.get("listens", 0) >= min_popularity)
            
            # Require genre, tempo, and listens (year is optional)
            if has_genre and has_tempo and has_listens:
                valid_tracks.append(track_id)
        
        if not valid_tracks:
            log.warning("No tracks found with strict criteria, using relaxed criteria...")
            # Fallback: just require basic metadata exists
            for track_id in self.metadata.index:
                meta = self.metadata.loc[track_id]
                if (pd.notna(meta.get("listens")) and 
                    pd.notna(meta.get("tempo"))):
                    valid_tracks.append(track_id)
        
        if not valid_tracks:
            log.error("Still no valid tracks found! Using first few tracks...")
            valid_tracks = list(self.metadata.index[:100])
        
        # Sample random seeds
        np.random.seed(42)  # Reproducible
        selected = np.random.choice(valid_tracks, 
                                   min(num_seeds, len(valid_tracks)), 
                                   replace=False)
        
        log.info(f"Selected {len(selected)} valid seed tracks from {len(valid_tracks)} candidates")
        return selected.tolist()
    
    def evaluate_recommendations(self, 
                                seed_track_id: int,
                                rec_track_ids: List[int],
                                k: int = 10) -> Dict[str, float]:
        """Evaluate a single set of recommendations."""
        if not rec_track_ids:
            return self._empty_metrics()
        
        # Limit to top-k
        rec_track_ids = rec_track_ids[:k]
        
        # Get seed metadata
        if seed_track_id not in self.metadata.index:
            return self._empty_metrics()
        
        seed_meta = self.metadata.loc[seed_track_id]
        
        # Calculate metrics
        metrics = {}
        
        # 1. Genre Consistency
        metrics["genre_consistency@k"] = self._calculate_genre_consistency(
            seed_meta, rec_track_ids
        )
        
        # 2. Artist Diversity  
        metrics["artist_diversity@k"] = self._calculate_artist_diversity(rec_track_ids)
        
        # 3. Tempo Coherence
        metrics["avg_tempo_diff"] = self._calculate_tempo_coherence(
            seed_meta, rec_track_ids
        )
        
        # 4. Year Spread
        metrics["year_spread"] = self._calculate_year_spread(rec_track_ids)
        
        # 5. Popularity Distribution
        metrics["avg_popularity"] = self._calculate_avg_popularity(rec_track_ids)
        
        # 6. Overall Quality Score (composite)
        metrics["quality_score"] = self._calculate_quality_score(metrics)
        
        return metrics
    
    def _calculate_genre_consistency(self, seed_meta: pd.Series, rec_ids: List[int]) -> float:
        """Calculate percentage of recommendations with genre overlap."""
        # Use genres_all if available, otherwise fall back to genre_top
        seed_genres_all = seed_meta.get("genres_all", [])
        seed_genre_top = seed_meta.get("genre_top")
        
        if isinstance(seed_genres_all, list) and len(seed_genres_all) > 0:
            seed_genres = set(seed_genres_all)
        elif pd.notna(seed_genre_top):
            seed_genres = {seed_genre_top}
        else:
            return 0.0
        
        overlap_count = 0
        valid_recs = 0
        
        for rec_id in rec_ids:
            if rec_id in self.metadata.index:
                rec_meta = self.metadata.loc[rec_id]
                rec_genres_all = rec_meta.get("genres_all", [])
                rec_genre_top = rec_meta.get("genre_top")
                
                # Use genres_all if available, otherwise genre_top
                if isinstance(rec_genres_all, list) and len(rec_genres_all) > 0:
                    rec_genres = set(rec_genres_all)
                elif pd.notna(rec_genre_top):
                    rec_genres = {rec_genre_top}
                else:
                    continue  # Skip if no genre data
                
                valid_recs += 1
                if _jaccard(seed_genres, rec_genres) > 0:
                    overlap_count += 1
        
        return overlap_count / max(valid_recs, 1)
    
    def _calculate_artist_diversity(self, rec_ids: List[int]) -> float:
        """Calculate ratio of unique artists to total recommendations."""
        artists = []
        
        for rec_id in rec_ids:
            if rec_id in self.metadata.index:
                artist = self.metadata.loc[rec_id].get("artist_name")
                if pd.notna(artist):
                    artists.append(artist)
        
        if not artists:
            return 0.0
        
        return len(set(artists)) / len(artists)
    
    def _calculate_tempo_coherence(self, seed_meta: pd.Series, rec_ids: List[int]) -> float:
        """Calculate average tempo difference from seed."""
        seed_tempo = seed_meta.get("tempo")
        if pd.isna(seed_tempo):
            return float('inf')
        
        tempo_diffs = []
        
        for rec_id in rec_ids:
            if rec_id in self.metadata.index:
                rec_tempo = self.metadata.loc[rec_id].get("tempo")
                if pd.notna(rec_tempo):
                    tempo_diffs.append(abs(float(seed_tempo) - float(rec_tempo)))
        
        return np.mean(tempo_diffs) if tempo_diffs else float('inf')
    
    def _calculate_year_spread(self, rec_ids: List[int]) -> float:
        """Calculate year spread (max - min) in recommendations."""
        years = []
        
        for rec_id in rec_ids:
            if rec_id in self.metadata.index:
                year = self.metadata.loc[rec_id].get("year")
                if pd.notna(year):
                    years.append(int(year))
        
        if len(years) < 2:
            return 0.0
        
        return float(max(years) - min(years))
    
    def _calculate_avg_popularity(self, rec_ids: List[int]) -> float:
        """Calculate average popularity (log listens) of recommendations."""
        popularities = []
        
        for rec_id in rec_ids:
            if rec_id in self.metadata.index:
                listens = self.metadata.loc[rec_id].get("listens", 0)
                if pd.notna(listens) and listens > 0:
                    popularities.append(np.log1p(float(listens)))
        
        return np.mean(popularities) if popularities else 0.0
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite quality score (0-1, higher is better)."""
        # Normalize and weight metrics
        genre_score = metrics.get("genre_consistency@k", 0.0)  # 0-1, higher better
        diversity_score = metrics.get("artist_diversity@k", 0.0)  # 0-1, higher better
        
        # Tempo coherence: convert to score (lower diff = higher score)
        tempo_diff = metrics.get("avg_tempo_diff", float('inf'))
        if tempo_diff == float('inf'):
            tempo_score = 0.0
        else:
            # Good coherence: within 20 BPM = score 1.0, 50+ BPM = score 0.0
            tempo_score = max(0.0, 1.0 - tempo_diff / 50.0)
        
        # Year spread: moderate diversity is good (5-15 years ideal)
        year_spread = metrics.get("year_spread", 0.0)
        if year_spread <= 5:
            year_score = year_spread / 5.0  # 0-5 years: linearly increase
        elif year_spread <= 15:
            year_score = 1.0  # 5-15 years: optimal
        else:
            year_score = max(0.0, 1.0 - (year_spread - 15) / 20.0)  # 15+ years: decrease
        
        # Weighted composite
        weights = {
            'genre': 0.3,      # Genre consistency most important
            'diversity': 0.25,  # Artist diversity important
            'tempo': 0.25,     # Tempo coherence important  
            'year': 0.2        # Year diversity less critical
        }
        
        quality = (weights['genre'] * genre_score +
                  weights['diversity'] * diversity_score +
                  weights['tempo'] * tempo_score +
                  weights['year'] * year_score)
        
        return quality
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics for failed evaluations."""
        return {
            "genre_consistency@k": 0.0,
            "artist_diversity@k": 0.0,
            "avg_tempo_diff": float('inf'),
            "year_spread": 0.0,
            "avg_popularity": 0.0,
            "quality_score": 0.0
        }


def evaluate_algorithm(algorithm: str, 
                      seeds: List[int],
                      k: int = 10,
                      **kwargs) -> List[Dict]:
    """Evaluate a specific algorithm on seed tracks."""
    evaluator = RecommendationEvaluator()
    results = []
    
    log.info(f"Evaluating {algorithm} algorithm on {len(seeds)} seeds (k={k})")
    
    for seed_id in tqdm(seeds, desc=f"Evaluating {algorithm}"):
        try:
            # Generate recommendations based on algorithm
            if algorithm == "audio":
                rec_ids, _ = recommend(seed_id, k=k)
            elif algorithm == "hybrid":
                rec_ids, _ = recommend_hybrid(seed_id, top=k, **kwargs)
            elif algorithm == "playlist":
                rec_ids = build_playlist(seed_id, n=k, **kwargs)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Evaluate recommendations
            metrics = evaluator.evaluate_recommendations(seed_id, rec_ids, k)
            
            # Add metadata
            result = {
                "seed_track_id": seed_id,
                "algorithm": algorithm,
                "algorithm_name": algorithm.title(),  # Add display name
                "k": k,
                **metrics,
                **kwargs  # Include algorithm parameters
            }
            
            results.append(result)
            
        except Exception as e:
            log.warning(f"Failed to evaluate seed {seed_id} with {algorithm}: {e}")
            # Add failed result
            result = {
                "seed_track_id": seed_id,
                "algorithm": algorithm,
                "algorithm_name": algorithm.title(),  # Add display name
                "k": k,
                **evaluator._empty_metrics(),
                "error": str(e),
                **kwargs
            }
            results.append(result)
    
    return results


def compare_algorithms(seeds: List[int], k: int = 10) -> pd.DataFrame:
    """Compare multiple algorithms on the same seed set."""
    all_results = []
    
    # Algorithm configurations to compare
    configs = [
        {"algorithm": "audio", "name": "Pure Audio"},
        {"algorithm": "hybrid", "name": "Hybrid (default)", 
         "w_genre": 0.35, "w_tempo": 0.15, "w_year": 0.10},
        {"algorithm": "hybrid", "name": "Hybrid (genre-focused)",
         "w_genre": 0.50, "w_tempo": 0.10, "w_year": 0.05},
        {"algorithm": "playlist", "name": "MMR λ=0.3", "lambda_mmr": 0.3},
        {"algorithm": "playlist", "name": "MMR λ=0.1 (diverse)", "lambda_mmr": 0.1},
    ]
    
    for config in configs:
        algorithm = config.pop("algorithm")
        name = config.pop("name")
        
        results = evaluate_algorithm(algorithm, seeds, k, **config)
        
        # Add display name
        for result in results:
            result["algorithm_name"] = name
        
        all_results.extend(results)
    
    return pd.DataFrame(all_results)


def generate_summary_report(results_df: pd.DataFrame) -> Dict:
    """Generate summary statistics and insights."""
    summary = {}
    
    # Group by algorithm
    for alg_name in results_df["algorithm_name"].unique():
        alg_results = results_df[results_df["algorithm_name"] == alg_name]
        
        # Skip if no valid results
        valid_results = alg_results[alg_results["quality_score"] > 0]
        if len(valid_results) == 0:
            continue
        
        alg_summary = {
            "num_evaluations": len(alg_results),
            "num_successful": len(valid_results),
            "success_rate": len(valid_results) / len(alg_results),
            "metrics": {
                "genre_consistency": {
                    "mean": float(valid_results["genre_consistency@k"].mean()),
                    "std": float(valid_results["genre_consistency@k"].std()),
                    "median": float(valid_results["genre_consistency@k"].median())
                },
                "artist_diversity": {
                    "mean": float(valid_results["artist_diversity@k"].mean()),
                    "std": float(valid_results["artist_diversity@k"].std()),
                    "median": float(valid_results["artist_diversity@k"].median())
                },
                "avg_tempo_diff": {
                    "mean": float(valid_results["avg_tempo_diff"].replace([np.inf], np.nan).mean()),
                    "std": float(valid_results["avg_tempo_diff"].replace([np.inf], np.nan).std()),
                    "median": float(valid_results["avg_tempo_diff"].replace([np.inf], np.nan).median())
                },
                "year_spread": {
                    "mean": float(valid_results["year_spread"].mean()),
                    "std": float(valid_results["year_spread"].std()),
                    "median": float(valid_results["year_spread"].median())
                },
                "quality_score": {
                    "mean": float(valid_results["quality_score"].mean()),
                    "std": float(valid_results["quality_score"].std()),
                    "median": float(valid_results["quality_score"].median())
                }
            }
        }
        
        summary[alg_name] = alg_summary
    
    return summary


def print_evaluation_report(summary: Dict):
    """Print human-readable evaluation report."""
    print("\n" + "="*80)
    print("RECOMMENDATION SYSTEM EVALUATION REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create comparison table
    algorithms = list(summary.keys())
    if not algorithms:
        print("❌ No successful evaluations found!")
        return
    
    print("ALGORITHM PERFORMANCE COMPARISON")
    print("-" * 80)
    
    # Headers
    print(f"{'Algorithm':<25} {'Genre@k':<10} {'Diversity':<10} {'Tempo Diff':<12} {'Quality':<10}")
    print("-" * 80)
    
    # Results
    for alg_name in algorithms:
        alg_data = summary[alg_name]
        metrics = alg_data["metrics"]
        
        genre_score = metrics["genre_consistency"]["mean"]
        diversity_score = metrics["artist_diversity"]["mean"] 
        tempo_diff = metrics["avg_tempo_diff"]["mean"]
        quality_score = metrics["quality_score"]["mean"]
        
        print(f"{alg_name:<25} {genre_score:<10.3f} {diversity_score:<10.3f} "
              f"{tempo_diff:<12.1f} {quality_score:<10.3f}")
    
    print("-" * 80)
    print()
    
    # Insights
    print("KEY INSIGHTS:")
    print("-" * 20)
    
    # Find best performers
    best_quality = max(summary.items(), key=lambda x: x[1]["metrics"]["quality_score"]["mean"])
    best_genre = max(summary.items(), key=lambda x: x[1]["metrics"]["genre_consistency"]["mean"])
    best_diversity = max(summary.items(), key=lambda x: x[1]["metrics"]["artist_diversity"]["mean"])
    
    print(f"🏆 Best Overall Quality: {best_quality[0]} (score: {best_quality[1]['metrics']['quality_score']['mean']:.3f})")
    print(f"🎵 Best Genre Consistency: {best_genre[0]} ({best_genre[1]['metrics']['genre_consistency']['mean']:.3f})")
    print(f"🎨 Best Artist Diversity: {best_diversity[0]} ({best_diversity[1]['metrics']['artist_diversity']['mean']:.3f})")
    
    # Tempo analysis
    tempo_scores = [(name, data["metrics"]["avg_tempo_diff"]["mean"]) 
                   for name, data in summary.items()]
    best_tempo = min(tempo_scores, key=lambda x: x[1])
    print(f"🎼 Best Tempo Coherence: {best_tempo[0]} ({best_tempo[1]:.1f} BPM avg diff)")
    
    print()


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate recommendation system quality")
    
    parser.add_argument("--algorithm", type=str, choices=["audio", "hybrid", "playlist"],
                       help="Single algorithm to evaluate")
    parser.add_argument("--compare-all", action="store_true",
                       help="Compare multiple algorithms")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of recommendations to evaluate")
    parser.add_argument("--num-seeds", type=int, default=100,
                       help="Number of seed tracks to test")
    parser.add_argument("--export-results", action="store_true",
                       help="Export detailed results to CSV")
    
    # Algorithm-specific parameters
    parser.add_argument("--w-genre", type=float, default=0.35,
                       help="Genre weight for hybrid algorithm")
    parser.add_argument("--w-tempo", type=float, default=0.15,
                       help="Tempo weight for hybrid algorithm")
    parser.add_argument("--w-year", type=float, default=0.10,
                       help="Year weight for hybrid algorithm")
    parser.add_argument("--lambda-mmr", type=float, default=0.3,
                       help="MMR lambda for playlist algorithm")
    
    args = parser.parse_args()
    
    # Setup
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get evaluation seeds
    evaluator = RecommendationEvaluator()
    seeds = evaluator.get_valid_seeds(args.num_seeds)
    
    if not seeds:
        log.error("No valid seed tracks found!")
        return
    
    # Run evaluation
    if args.compare_all:
        log.info("Running comprehensive algorithm comparison...")
        results_df = compare_algorithms(seeds, k=args.k)
        
        # Export results
        if args.export_results:
            results_path = EVAL_DIR / f"eval_results_{timestamp}.csv"
            results_df.to_csv(results_path, index=False)
            log.info(f"Results exported to: {results_path}")
        
        # Generate summary
        summary = generate_summary_report(results_df)
        
        # Export summary
        summary_path = EVAL_DIR / f"eval_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        log.info(f"Summary exported to: {summary_path}")
        
        # Print report
        print_evaluation_report(summary)
        
    elif args.algorithm:
        log.info(f"Evaluating {args.algorithm} algorithm...")
        
        # Algorithm parameters
        kwargs = {}
        if args.algorithm == "hybrid":
            kwargs = {"w_genre": args.w_genre, "w_tempo": args.w_tempo, "w_year": args.w_year}
        elif args.algorithm == "playlist":
            kwargs = {"lambda_mmr": args.lambda_mmr}
        
        results = evaluate_algorithm(args.algorithm, seeds, args.k, **kwargs)
        results_df = pd.DataFrame(results)
        
        # Print basic stats
        valid_results = results_df[results_df["quality_score"] > 0]
        print(f"\n{args.algorithm.upper()} ALGORITHM EVALUATION")
        print("-" * 50)
        print(f"Seeds evaluated: {len(results)}")
        print(f"Successful: {len(valid_results)} ({len(valid_results)/len(results)*100:.1f}%)")
        
        if len(valid_results) > 0:
            print(f"Avg Quality Score: {valid_results['quality_score'].mean():.3f}")
            print(f"Avg Genre Consistency: {valid_results['genre_consistency@k'].mean():.3f}")
            print(f"Avg Artist Diversity: {valid_results['artist_diversity@k'].mean():.3f}")
            print(f"Avg Tempo Difference: {valid_results['avg_tempo_diff'].replace([np.inf], np.nan).mean():.1f} BPM")
        
        # Export if requested
        if args.export_results:
            results_path = EVAL_DIR / f"eval_{args.algorithm}_{timestamp}.csv"
            results_df.to_csv(results_path, index=False)
            log.info(f"Results exported to: {results_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
