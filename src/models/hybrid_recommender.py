#!/usr/bin/env python3
"""
Enhanced Hybrid Recommender with Guardrails

Extends the existing hybrid recommendation system with comprehensive validation
and error handling for production deployment.

Key features:
- Tempo coverage validation for w_tempo parameter
- Data completeness checks before recommendation
- Graceful error handling with actionable guidance  
- Enhanced hybrid recommendation with safety checks
"""

import argparse
from pathlib import Path
from typing import Tuple, List

import pandas as pd

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.knn_recommender import (
    recommend_hybrid, build_playlist, _load_meta,
    display_hybrid, display_playlist
)
from src.utils.guardrails import (
    GuardrailValidator, validate_recommendation_request,
    RecommendationSystemError, ConfigurationError
)
from src.utils.logging import get_logger

log = get_logger(__name__)

# Default paths
ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "data/processed/fma_small_index.parquet"
META_PATH = ROOT / "data/processed/fma_small_meta.parquet"


def recommend_hybrid_safe(seed_track_id: int,
                         top: int = 50,
                         candidate_k: int = 200,
                         w_audio: float = 1.0,
                         w_genre: float = 0.35,
                         w_tempo: float = 0.15,
                         w_year: float = 0.10,
                         artist_penalty: float = 0.05,
                         validate_data: bool = True) -> Tuple[List[int], List[float]]:
    """
    Enhanced hybrid recommendation with comprehensive validation.
    
    Args:
        seed_track_id: ID of seed track
        top: Number of recommendations to return
        candidate_k: Number of audio candidates to consider
        w_audio: Weight for audio similarity (base KNN)
        w_genre: Weight for genre Jaccard similarity  
        w_tempo: Weight for tempo proximity
        w_year: Weight for year proximity
        artist_penalty: Diversity penalty for same artist
        validate_data: Whether to run data validation (disable for performance)
        
    Returns:
        (recommended_track_ids, scores)
        
    Raises:
        RecommendationSystemError: If validation fails with helpful guidance
        ConfigurationError: If system configuration is invalid
    """
    validator = GuardrailValidator()
    
    if validate_data:
        # Load data for validation
        meta_df = _load_meta()
        index_df = pd.read_parquet(INDEX_PATH)
        
        # Validate tempo usage
        try:
            validator.validate_tempo_for_hybrid(meta_df, w_tempo, min_coverage=0.7)
        except ConfigurationError as e:
            log.error(str(e))
            raise
            
        # Validate seed track with enhanced error messages
        available_tracks = index_df["track_id"].tolist()
        try:
            validate_recommendation_request(seed_track_id, top, available_tracks, index_df)
        except RecommendationSystemError as e:
            log.error(str(e))
            raise
            
        # Validate data completeness
        try:
            from src.models.knn_recommender import _load_features, FEAT_PATH
            features_df, _, _ = _load_features(FEAT_PATH)
            validator.validate_data_completeness(features_df, meta_df, index_df)
        except Exception as e:
            log.warning(f"Data validation warning: {e}")
    
    # Call original hybrid recommendation
    try:
        return recommend_hybrid(
            seed_track_id, top, candidate_k,
            w_audio, w_genre, w_tempo, w_year, artist_penalty
        )
    except Exception as e:
        # Enhance error message for common issues
        if "not found" in str(e).lower():
            raise RecommendationSystemError(
                f"❌ Recommendation failed: {str(e)}\n\n"
                f"💡 This might indicate:\n"
                f"• Seed track not in feature dataset\n"
                f"• Model/data mismatch\n"
                f"• Corrupted data files\n\n"
                f"🔧 Try rebuilding the model or checking data integrity"
            )
        else:
            raise RecommendationSystemError(f"❌ Recommendation failed: {str(e)}")


def build_playlist_safe(seed_track_id: int,
                       n: int = 50,
                       candidate_k: int = 500,
                       lambda_mmr: float = 0.3,
                       max_per_artist: int = 2,
                       max_per_album: int = 3,
                       validate_data: bool = True) -> List[int]:
    """
    Enhanced MMR playlist building with validation.
    
    Args:
        seed_track_id: ID of seed track
        n: Number of tracks in playlist
        candidate_k: Number of candidates for MMR selection
        lambda_mmr: MMR trade-off (0=diversity, 1=similarity)
        max_per_artist: Maximum tracks per artist
        max_per_album: Maximum tracks per album
        validate_data: Whether to run data validation
        
    Returns:
        List of track IDs in playlist
        
    Raises:
        RecommendationSystemError: If validation fails
    """
    validator = GuardrailValidator()
    
    if validate_data:
        # Load data for validation
        index_df = pd.read_parquet(INDEX_PATH)
        available_tracks = index_df["track_id"].tolist()
        
        # Validate playlist request
        try:
            validate_recommendation_request(seed_track_id, n, available_tracks, index_df)
        except RecommendationSystemError as e:
            log.error(str(e))
            raise
            
        # Additional MMR-specific validation
        if lambda_mmr < 0 or lambda_mmr > 1:
            raise RecommendationSystemError(
                f"❌ Invalid lambda_mmr: {lambda_mmr}\n\n"
                f"💡 lambda_mmr must be between 0 and 1:\n"
                f"• 0.0 = maximum diversity (very different tracks)\n"
                f"• 1.0 = maximum similarity (very similar tracks)\n"
                f"• 0.3 = balanced (recommended default)"
            )
            
        if max_per_artist <= 0:
            raise RecommendationSystemError(
                f"❌ Invalid max_per_artist: {max_per_artist}\n\n"
                f"💡 max_per_artist must be positive (e.g., 2 for diversity)"
            )
    
    # Call original playlist builder
    try:
        return build_playlist(
            seed_track_id, n, candidate_k,
            lambda_mmr, max_per_artist, max_per_album
        )
    except Exception as e:
        raise RecommendationSystemError(f"❌ Playlist generation failed: {str(e)}")


def display_hybrid_safe(seed_track_id: int, 
                       top: int = 10,
                       w_audio: float = 1.0,
                       w_genre: float = 0.35, 
                       w_tempo: float = 0.15,
                       w_year: float = 0.10,
                       artist_penalty: float = 0.05) -> None:
    """Enhanced hybrid display with validation."""
    try:
        recommend_hybrid_safe(
            seed_track_id, top, 200,
            w_audio, w_genre, w_tempo, w_year, artist_penalty
        )
        # If validation passes, call original display
        display_hybrid(seed_track_id, top, w_audio, w_genre, w_tempo, w_year, artist_penalty)
    except (RecommendationSystemError, ConfigurationError) as e:
        print(str(e))


def display_playlist_safe(seed_track_id: int, 
                         n: int = 50,
                         lambda_mmr: float = 0.3,
                         max_per_artist: int = 2,
                         max_per_album: int = 3) -> None:
    """Enhanced playlist display with validation."""
    try:
        build_playlist_safe(
            seed_track_id, n, 500,
            lambda_mmr, max_per_artist, max_per_album
        )
        # If validation passes, call original display
        display_playlist(seed_track_id, n)
    except RecommendationSystemError as e:
        print(str(e))


# CLI interface with enhanced error handling
def main():
    """CLI interface for safe hybrid recommendations."""
    parser = argparse.ArgumentParser(description="Enhanced Hybrid Music Recommender with Guardrails")
    
    # Operations
    parser.add_argument("--hybrid", type=int, help="Get hybrid recommendations for track ID")
    parser.add_argument("--playlist", type=int, help="Generate MMR playlist for track ID")
    
    # Recommendation parameters
    parser.add_argument("--top", type=int, default=10, help="Number of recommendations (default: 10)")
    parser.add_argument("--n", type=int, default=50, help="Playlist size (default: 50)")
    
    # Hybrid weights with validation
    parser.add_argument("--w-audio", type=float, default=1.0, help="Audio weight (default: 1.0)")
    parser.add_argument("--w-genre", type=float, default=0.35, help="Genre weight (default: 0.35)")
    parser.add_argument("--w-tempo", type=float, default=0.15, help="Tempo weight (default: 0.15)")
    parser.add_argument("--w-year", type=float, default=0.10, help="Year weight (default: 0.10)")
    parser.add_argument("--artist-penalty", type=float, default=0.05, help="Artist diversity penalty (default: 0.05)")
    
    # MMR parameters
    parser.add_argument("--lambda-mmr", type=float, default=0.3, help="MMR trade-off (default: 0.3)")
    parser.add_argument("--max-per-artist", type=int, default=2, help="Max tracks per artist (default: 2)")
    parser.add_argument("--max-per-album", type=int, default=3, help="Max tracks per album (default: 3)")
    
    # System options
    parser.add_argument("--no-validation", action="store_true", help="Skip data validation for performance")
    
    args = parser.parse_args()
    
    validate_data = not args.no_validation
    
    try:
        if args.hybrid is not None:
            print(f"🎵 Getting enhanced hybrid recommendations for track {args.hybrid}...")
            display_hybrid_safe(
                args.hybrid, args.top,
                args.w_audio, args.w_genre, args.w_tempo, args.w_year, args.artist_penalty
            )
            
        elif args.playlist is not None:
            print(f"🎶 Generating enhanced MMR playlist for track {args.playlist}...")
            display_playlist_safe(
                args.playlist, args.n,
                args.lambda_mmr, args.max_per_artist, args.max_per_album
            )
            
        else:
            parser.print_help()
            
    except (RecommendationSystemError, ConfigurationError) as e:
        print(f"\n{str(e)}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("🔧 Please check your data setup and try again")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
