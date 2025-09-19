#!/usr/bin/env python3
"""
Performance Audit Script - Phase 2 Scale Preparation

Comprehensive profiling of the current recommendation system to identify
bottlenecks before scaling to FMA Large (100K+ tracks).

Analyzes:
- Memory usage patterns
- Computation time breakdown
- I/O operation costs
- Scalability projections
"""

import time
import psutil
import tracemalloc
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, Any
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.knn_recommender import (
    _load_features, _load_meta,
    recommend_hybrid, build_playlist, _matrix_in_saved_order
)
from src.utils.logging import get_logger

log = get_logger(__name__)

class PerformanceProfiler:
    """Comprehensive performance profiling for recommendation system."""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def profile_memory_usage(self) -> Dict[str, Any]:
        """Profile memory usage across different operations."""
        print("🧠 PROFILING MEMORY USAGE")
        print("=" * 50)
        
        tracemalloc.start()
        
        # Baseline memory
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Profile data loading
        print("📁 Loading features...")
        start_snapshot = tracemalloc.take_snapshot()
        features_df, _, track_ids = _load_features(Path("data/processed/fma_small_feats_v2.parquet"))
        features_snapshot = tracemalloc.take_snapshot()
        
        print("📋 Loading metadata...")
        meta_df = _load_meta()
        meta_snapshot = tracemalloc.take_snapshot()
        
        print("🔗 Loading index...")
        index_df = pd.read_parquet("data/processed/fma_small_index.parquet")
        index_snapshot = tracemalloc.take_snapshot()
        
        # Calculate memory deltas
        features_stats = features_snapshot.compare_to(start_snapshot, 'lineno')
        meta_stats = meta_snapshot.compare_to(features_snapshot, 'lineno')
        index_stats = index_snapshot.compare_to(meta_snapshot, 'lineno')
        
        # Current memory after all loads
        loaded_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Profile feature matrix creation
        print("🔢 Creating feature matrix...")
        matrix_start = time.time()
        feature_matrix, idx_map = _matrix_in_saved_order(
            Path("data/processed/fma_small_feats_v2.parquet"), 
            np.array(track_ids)
        )
        matrix_time = time.time() - matrix_start
        matrix_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        tracemalloc.stop()
        
        return {
            "baseline_memory_mb": baseline_memory,
            "loaded_memory_mb": loaded_memory,
            "matrix_memory_mb": matrix_memory,
            "total_memory_increase_mb": loaded_memory - baseline_memory,
            "matrix_creation_time_sec": matrix_time,
            "features_size_mb": sum(stat.size for stat in features_stats) / 1024 / 1024,
            "meta_size_mb": sum(stat.size for stat in meta_stats) / 1024 / 1024,
            "index_size_mb": sum(stat.size for stat in index_stats) / 1024 / 1024,
            "num_tracks": len(track_ids),
            "feature_dimensions": feature_matrix.shape[1] if feature_matrix.size > 0 else 0,
            "memory_per_track_kb": (loaded_memory - baseline_memory) * 1024 / len(track_ids)
        }
    
    def profile_recommendation_performance(self) -> Dict[str, Any]:
        """Profile recommendation generation performance."""
        print("\n⚡ PROFILING RECOMMENDATION PERFORMANCE")
        print("=" * 50)
        
        # Load data once
        features_df, _, track_ids = _load_features(Path("data/processed/fma_small_feats_v2.parquet"))
        meta_df = _load_meta()
        index_df = pd.read_parquet("data/processed/fma_small_index.parquet")
        
        # Load pre-built model
        print("🔧 Loading KNN model...")
        model_start = time.time()
        # Load model payload (timing only, actual model not used in profiling)
        joblib.load("artifacts/knn_audio_v2.joblib")
        model_build_time = time.time() - model_start
        
        # Test recommendation generation with different parameters
        test_cases = [
            {"n": 5, "description": "Small playlist (5 tracks)"},
            {"n": 20, "description": "Medium playlist (20 tracks)"},
            {"n": 50, "description": "Large playlist (50 tracks)"},
            {"n": 100, "description": "XL playlist (100 tracks)"}
        ]
        
        # Get valid seed tracks
        valid_seeds = index_df[index_df["track_id"].isin(features_df.index)]["track_id"].head(10).tolist()
        
        performance_results = []
        
        for test_case in test_cases:
            print(f"🎵 Testing {test_case['description']}...")
            
            times = []
            for seed in valid_seeds[:3]:  # Test with 3 different seeds
                start_time = time.time()
                
                try:
                    # Hybrid recommendation
                    recommendations, scores = recommend_hybrid(seed, test_case["n"])
                    
                    # Playlist building (if we have enough recommendations)
                    if len(recommendations) >= min(test_case["n"], 5):
                        playlist = build_playlist(seed, min(test_case["n"], len(recommendations)))
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                except Exception as e:
                    log.warning(f"Failed test for seed {seed}, n={test_case['n']}: {e}")
                    continue
            
            if times:
                performance_results.append({
                    "playlist_size": test_case["n"],
                    "description": test_case["description"],
                    "avg_time_sec": np.mean(times),
                    "min_time_sec": np.min(times),
                    "max_time_sec": np.max(times),
                    "std_time_sec": np.std(times)
                })
        
        return {
            "model_build_time_sec": model_build_time,
            "performance_by_size": performance_results,
            "num_features": len(features_df),
            "feature_dim": features_df.iloc[0]["feature"].shape[0] if len(features_df) > 0 else 0
        }
    
    def profile_io_operations(self) -> Dict[str, Any]:
        """Profile I/O operation performance."""
        print("\n💾 PROFILING I/O OPERATIONS")
        print("=" * 50)
        
        # File paths
        feat_path = Path("data/processed/fma_small_feats_v2.parquet")
        meta_path = Path("data/processed/fma_small_meta.parquet")
        index_path = Path("data/processed/fma_small_index.parquet")
        model_path = Path("artifacts/knn_audio_v2.joblib")
        
        io_results = {}
        
        # Profile parquet loading
        for name, path in [("features", feat_path), ("metadata", meta_path), ("index", index_path)]:
            if path.exists():
                print(f"📊 Loading {name} from {path}...")
                start_time = time.time()
                df = pd.read_parquet(path)
                load_time = time.time() - start_time
                
                io_results[f"{name}_load_time_sec"] = load_time
                io_results[f"{name}_file_size_mb"] = path.stat().st_size / 1024 / 1024
                io_results[f"{name}_rows"] = len(df)
                io_results[f"{name}_cols"] = len(df.columns)
                io_results[f"{name}_memory_mb"] = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Profile model loading
        if model_path.exists():
            print(f"🤖 Loading model from {model_path}...")
            start_time = time.time()
            joblib.load(model_path)
            model_load_time = time.time() - start_time
            
            io_results["model_load_time_sec"] = model_load_time
            io_results["model_file_size_mb"] = model_path.stat().st_size / 1024 / 1024
        
        return io_results
    
    def project_scalability(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Project performance for FMA Large scale."""
        print("\n📈 SCALABILITY PROJECTION")
        print("=" * 50)
        
        # FMA Large estimates
        fma_large_tracks = 106574  # Actual FMA Large track count
        current_tracks = current_results["memory"]["num_tracks"]
        scale_factor = fma_large_tracks / current_tracks
        
        # Memory projections (linear scaling assumption)
        projected_memory_mb = current_results["memory"]["total_memory_increase_mb"] * scale_factor
        projected_per_track_kb = current_results["memory"]["memory_per_track_kb"]
        
        # Time projections (KNN is O(n) for search, O(n log n) for k-selection)
        # Conservative estimate: O(n log n) scaling for safety
        time_scale_factor = scale_factor * np.log2(fma_large_tracks) / np.log2(current_tracks)
        
        projected_times = {}
        if "performance" in current_results and "performance_by_size" in current_results["performance"]:
            for perf in current_results["performance"]["performance_by_size"]:
                projected_times[f"playlist_{perf['playlist_size']}"] = {
                    "current_avg_sec": perf["avg_time_sec"],
                    "projected_avg_sec": perf["avg_time_sec"] * time_scale_factor,
                    "projected_avg_min": (perf["avg_time_sec"] * time_scale_factor) / 60
                }
        
        return {
            "scale_factor": scale_factor,
            "fma_large_tracks": fma_large_tracks,
            "current_tracks": current_tracks,
            "projected_memory_mb": projected_memory_mb,
            "projected_memory_gb": projected_memory_mb / 1024,
            "projected_per_track_kb": projected_per_track_kb,
            "time_scale_factor": time_scale_factor,
            "projected_performance": projected_times,
            "feasibility_assessment": self._assess_feasibility(projected_memory_mb, projected_times)
        }
    
    def _assess_feasibility(self, projected_memory_mb: float, projected_times: Dict) -> Dict[str, str]:
        """Assess feasibility of current approach for FMA Large."""
        assessment = {}
        
        # Memory assessment
        if projected_memory_mb < 2000:  # < 2GB
            assessment["memory"] = "✅ EXCELLENT - Fits comfortably in typical RAM"
        elif projected_memory_mb < 8000:  # < 8GB
            assessment["memory"] = "⚠️  MODERATE - Requires decent server RAM"
        elif projected_memory_mb < 16000:  # < 16GB
            assessment["memory"] = "🚨 CHALLENGING - Needs high-memory server"
        else:
            assessment["memory"] = "❌ INFEASIBLE - Requires optimization/streaming"
        
        # Performance assessment
        if projected_times:
            max_time = max(t.get("projected_avg_sec", 0) for t in projected_times.values())
            if max_time < 1:
                assessment["performance"] = "✅ EXCELLENT - Sub-second responses"
            elif max_time < 5:
                assessment["performance"] = "⚠️  MODERATE - Acceptable for most use cases"
            elif max_time < 30:
                assessment["performance"] = "🚨 CHALLENGING - May need optimization"
            else:
                assessment["performance"] = "❌ INFEASIBLE - Requires ANN or other optimization"
        
        return assessment
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete performance audit."""
        print("🚀 STARTING COMPREHENSIVE PERFORMANCE AUDIT")
        print("=" * 60)
        print("This will analyze current system performance and project FMA Large scalability")
        print()
        
        results = {}
        
        # Memory profiling
        results["memory"] = self.profile_memory_usage()
        
        # Performance profiling
        results["performance"] = self.profile_recommendation_performance()
        
        # I/O profiling
        results["io"] = self.profile_io_operations()
        
        # Scalability projection
        results["scalability"] = self.project_scalability(results)
        
        return results
    
    def generate_audit_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable audit report."""
        report = []
        report.append("📊 PERFORMANCE AUDIT REPORT")
        report.append("=" * 60)
        
        # Current System Stats
        report.append("\n🎯 CURRENT SYSTEM (FMA Small)")
        report.append(f"Tracks: {results['memory']['num_tracks']:,}")
        report.append(f"Features: {results['memory']['feature_dimensions']} dimensions")
        report.append(f"Memory Usage: {results['memory']['total_memory_increase_mb']:.1f} MB")
        report.append(f"Memory per Track: {results['memory']['memory_per_track_kb']:.2f} KB")
        
        # Performance Breakdown
        if "performance_by_size" in results['performance']:
            report.append("\n⚡ RECOMMENDATION PERFORMANCE")
            for perf in results['performance']['performance_by_size']:
                report.append(f"  {perf['description']}: {perf['avg_time_sec']:.3f}s ± {perf['std_time_sec']:.3f}s")
        
        # I/O Performance
        report.append("\n💾 I/O PERFORMANCE")
        io = results['io']
        for key, value in io.items():
            if "_time_sec" in key:
                name = key.replace("_load_time_sec", "").replace("_time_sec", "")
                report.append(f"  {name.title()} load: {value:.3f}s")
        
        # Scalability Projection
        scale = results['scalability']
        report.append(f"\n📈 FMA LARGE PROJECTION ({scale['fma_large_tracks']:,} tracks)")
        report.append(f"Scale Factor: {scale['scale_factor']:.1f}x")
        report.append(f"Projected Memory: {scale['projected_memory_gb']:.1f} GB")
        
        if scale['projected_performance']:
            report.append("\n⏱️  PROJECTED PERFORMANCE")
            for name, proj in scale['projected_performance'].items():
                if proj['projected_avg_min'] < 1:
                    time_str = f"{proj['projected_avg_sec']:.1f}s"
                else:
                    time_str = f"{proj['projected_avg_min']:.1f}m"
                report.append(f"  {name}: {time_str}")
        
        # Feasibility Assessment
        assess = scale['feasibility_assessment']
        report.append("\n🎯 FEASIBILITY ASSESSMENT")
        report.append(f"Memory: {assess.get('memory', 'N/A')}")
        report.append(f"Performance: {assess.get('performance', 'N/A')}")
        
        # Recommendations
        report.append("\n💡 OPTIMIZATION RECOMMENDATIONS")
        if "❌" in assess.get('memory', '') or "🚨" in assess.get('memory', ''):
            report.append("  🧠 MEMORY: Implement lazy loading or streaming")
        if "❌" in assess.get('performance', '') or "🚨" in assess.get('performance', ''):
            report.append("  ⚡ PERFORMANCE: Implement Approximate Nearest Neighbors (ANN)")
        
        report.append("  🔍 Consider FAISS or HNSW for large-scale similarity search")
        report.append("  📦 Implement batch processing for multiple recommendations")
        report.append("  🐳 Containerize for consistent deployment environments")
        
        return "\n".join(report)


def main():
    """Run the performance audit."""
    profiler = PerformanceProfiler()
    
    try:
        # Run full audit
        results = profiler.run_full_audit()
        
        # Generate and display report
        report = profiler.generate_audit_report(results)
        print("\n" + report)
        
        # Save detailed results
        output_dir = Path("scripts/audit_results")
        output_dir.mkdir(exist_ok=True)
        
        import json
        with open(output_dir / "performance_audit.json", "w") as f:
            # Convert numpy types for JSON serialization
            serializable_results = _make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {output_dir / 'performance_audit.json'}")
        print("\n🎯 AUDIT COMPLETE - Ready for Phase 2 optimization planning!")
        
    except Exception as e:
        log.error(f"Audit failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def _make_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


if __name__ == "__main__":
    exit(main())
