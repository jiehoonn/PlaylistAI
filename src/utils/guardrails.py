#!/usr/bin/env python3
"""
Guardrails & Error Handling for Music Recommendation System

Provides friendly error messages and data validation to ensure robust operation.
Prevents common user errors and provides helpful guidance for troubleshooting.

Key features:
- Seed track validation with helpful suggestions
- Data completeness checks with specific guidance
- Path configuration validation
- Tempo coverage validation for hybrid features
- Graceful error handling with actionable messages
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import joblib

from src.utils.logging import get_logger

log = get_logger(__name__)


class RecommendationSystemError(Exception):
    """Base exception for recommendation system errors with helpful messages."""
    pass


class SeedTrackNotFoundError(RecommendationSystemError):
    """Raised when seed track is not found in the dataset."""
    pass


class DataValidationError(RecommendationSystemError):
    """Raised when data validation fails."""
    pass


class ConfigurationError(RecommendationSystemError):
    """Raised when system configuration is invalid."""
    pass


class GuardrailValidator:
    """Comprehensive validation and error handling for the recommendation system."""
    
    def __init__(self):
        self.validation_cache = {}
        
    def validate_seed_track(self, 
                           seed_track_id: int, 
                           available_tracks: List[int],
                           index_df: Optional[pd.DataFrame] = None) -> None:
        """
        Validate that seed track exists and provide helpful suggestions if not.
        
        Args:
            seed_track_id: The track ID to validate
            available_tracks: List of valid track IDs
            index_df: Optional index DataFrame for enhanced suggestions
            
        Raises:
            SeedTrackNotFoundError: With helpful suggestions
        """
        if seed_track_id in available_tracks:
            return  # Valid seed
            
        # Generate helpful error message with suggestions
        suggestions = []
        
        # Find numerically close track IDs
        available_array = np.array(available_tracks)
        distances = np.abs(available_array - seed_track_id)
        closest_indices = np.argsort(distances)[:5]
        closest_tracks = available_array[closest_indices].tolist()
        
        suggestions.append(f"Closest track IDs: {closest_tracks}")
        
        # Provide random samples for exploration
        if len(available_tracks) > 10:
            random_samples = np.random.choice(available_tracks, 5, replace=False).tolist()
            suggestions.append(f"Random valid tracks to try: {random_samples}")
        else:
            suggestions.append(f"All valid tracks: {available_tracks[:10]}")
            
        # Add track info if index is available
        if index_df is not None and len(closest_tracks) > 0:
            track_info = []
            for track_id in closest_tracks[:3]:
                track_row = index_df[index_df["track_id"] == track_id]
                if not track_row.empty:
                    artist = track_row.iloc[0].get("artist_name", "Unknown")
                    title = track_row.iloc[0].get("track_title", "Unknown")
                    track_info.append(f"Track {track_id}: {artist} - {title}")
            if track_info:
                suggestions.append("Track details:")
                suggestions.extend([f"  {info}" for info in track_info])
                
        suggestion_text = "\n".join(suggestions)
        
        raise SeedTrackNotFoundError(
            f"❌ Seed track {seed_track_id} not found in dataset.\n\n"
            f"💡 Suggestions:\n{suggestion_text}\n\n"
            f"📊 Dataset info: {len(available_tracks):,} tracks available "
            f"(range: {min(available_tracks)} to {max(available_tracks)})"
        )
    
    def validate_data_completeness(self, 
                                 features_df: pd.DataFrame,
                                 meta_df: pd.DataFrame,
                                 index_df: pd.DataFrame,
                                 min_coverage: float = 0.8) -> Dict[str, Any]:
        """
        Validate data completeness and provide guidance on data issues.
        
        Args:
            features_df: Features DataFrame
            meta_df: Metadata DataFrame  
            index_df: Index DataFrame
            min_coverage: Minimum required coverage ratio
            
        Returns:
            Validation report dictionary
            
        Raises:
            DataValidationError: If critical validation fails
        """
        report = {
            "features": self._validate_features(features_df),
            "metadata": self._validate_metadata(meta_df, min_coverage),
            "index": self._validate_index(index_df),
            "alignment": self._validate_alignment(features_df, meta_df, index_df)
        }
        
        # Check for critical issues
        critical_issues = []
        warnings = []
        
        if report["features"]["null_count"] > 0:
            critical_issues.append(f"Features: {report['features']['null_count']} tracks with null features")
            
        if report["metadata"]["tempo_coverage"] < min_coverage:
            warnings.append(f"Tempo coverage: {report['metadata']['tempo_coverage']:.1%} < {min_coverage:.1%} threshold")
            
        if report["alignment"]["feature_meta_ratio"] < 0.8:  # Less than 80% overlap
            warnings.append(f"Data alignment: {report['alignment']['feature_meta_ratio']:.1%} feature-metadata overlap")
            
        # Report issues
        if critical_issues:
            issue_text = "\n".join([f"• {issue}" for issue in critical_issues])
            raise DataValidationError(
                f"❌ Critical data validation failures:\n\n{issue_text}\n\n"
                f"🔧 Please check your data pipeline and re-run feature extraction."
            )
            
        if warnings:
            warning_text = "\n".join([f"• {warning}" for warning in warnings])
            log.warning(f"⚠️  Data validation warnings:\n{warning_text}")
            
        return report
    
    def _validate_features(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate features DataFrame."""
        null_count = features_df["feature"].isnull().sum()
        
        # Check feature dimensions
        valid_features = features_df["feature"].dropna()
        if len(valid_features) > 0:
            first_feature = valid_features.iloc[0]
            feature_dim = len(first_feature) if hasattr(first_feature, '__len__') else 0
            
            # Check dimension consistency
            dim_consistency = all(
                len(feat) == feature_dim if hasattr(feat, '__len__') else False 
                for feat in valid_features.head(100)  # Sample check
            )
        else:
            feature_dim = 0
            dim_consistency = False
            
        return {
            "total_tracks": len(features_df),
            "null_count": null_count,
            "valid_count": len(features_df) - null_count,
            "feature_dimension": feature_dim,
            "dimension_consistent": dim_consistency,
            "coverage": (len(features_df) - null_count) / len(features_df) if len(features_df) > 0 else 0
        }
    
    def _validate_metadata(self, meta_df: pd.DataFrame, min_coverage: float) -> Dict[str, Any]:
        """Validate metadata DataFrame."""
        # Check key fields
        required_fields = ["track_id", "genre_top", "year", "listens"]
        optional_fields = ["tempo", "genres_all", "favorites"]
        
        field_coverage = {}
        for field in required_fields + optional_fields:
            if field in meta_df.columns:
                non_null = meta_df[field].notna().sum()
                field_coverage[field] = non_null / len(meta_df) if len(meta_df) > 0 else 0
            else:
                field_coverage[field] = 0.0
                
        # Special tempo validation (critical for hybrid recommendations)
        tempo_coverage = field_coverage.get("tempo", 0.0)
        tempo_warning = tempo_coverage < min_coverage
        
        return {
            "total_tracks": len(meta_df),
            "field_coverage": field_coverage,
            "tempo_coverage": tempo_coverage,
            "tempo_warning": tempo_warning,
            "missing_fields": [f for f in required_fields if f not in meta_df.columns]
        }
    
    def _validate_index(self, index_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate index DataFrame."""
        required_cols = ["track_id", "artist_name", "track_title"]
        missing_cols = [col for col in required_cols if col not in index_df.columns]
        
        # Check for duplicates
        duplicate_tracks = index_df["track_id"].duplicated().sum()
        
        return {
            "total_tracks": len(index_df),
            "missing_columns": missing_cols,
            "duplicate_tracks": duplicate_tracks,
            "has_artist_info": "artist_name" in index_df.columns,
            "has_track_info": "track_title" in index_df.columns
        }
    
    def _validate_alignment(self, 
                          features_df: pd.DataFrame,
                          meta_df: pd.DataFrame, 
                          index_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate alignment between datasets."""
        feature_tracks = set(features_df["track_id"]) if "track_id" in features_df.columns else set()
        
        # Handle metadata - might be indexed by track_id or have track_id column
        if "track_id" in meta_df.columns:
            meta_tracks = set(meta_df["track_id"])
        elif hasattr(meta_df.index, 'name') and meta_df.index.name == 'track_id':
            meta_tracks = set(meta_df.index)
        else:
            # Assume index contains track_ids
            meta_tracks = set(meta_df.index)
            
        index_tracks = set(index_df["track_id"]) if "track_id" in index_df.columns else set()
        
        # Find mismatches (only critical if no overlap at all)
        feature_meta_overlap = len(feature_tracks & meta_tracks)
        feature_index_overlap = len(feature_tracks & index_tracks)
        
        # Calculate ratios instead of absolute mismatches
        feature_meta_ratio = feature_meta_overlap / len(feature_tracks) if feature_tracks else 0
        feature_index_ratio = feature_index_overlap / len(feature_tracks) if feature_tracks else 0
        
        # Common tracks (for recommendations)
        common_tracks = feature_tracks & meta_tracks & index_tracks
        
        return {
            "feature_tracks": len(feature_tracks),
            "meta_tracks": len(meta_tracks),
            "index_tracks": len(index_tracks),
            "common_tracks": len(common_tracks),
            "feature_meta_overlap": feature_meta_overlap,
            "feature_index_overlap": feature_index_overlap,
            "feature_meta_ratio": feature_meta_ratio,
            "feature_index_ratio": feature_index_ratio,
            "alignment_ratio": len(common_tracks) / len(feature_tracks) if feature_tracks else 0
        }
    
    def validate_tempo_for_hybrid(self, 
                                meta_df: pd.DataFrame,
                                w_tempo: float,
                                min_coverage: float = 0.7) -> None:
        """
        Validate tempo coverage when using tempo-based hybrid recommendations.
        
        Args:
            meta_df: Metadata DataFrame
            w_tempo: Tempo weight (>0 means tempo is used)
            min_coverage: Minimum required tempo coverage
            
        Raises:
            ConfigurationError: If tempo weight > 0 but coverage is insufficient
        """
        if w_tempo <= 0:
            return  # Tempo not used, no validation needed
            
        tempo_coverage = 0.0
        if "tempo" in meta_df.columns:
            tempo_coverage = meta_df["tempo"].notna().sum() / len(meta_df)
            
        if tempo_coverage < min_coverage:
            raise ConfigurationError(
                f"❌ Tempo weight w_tempo={w_tempo} but only {tempo_coverage:.1%} "
                f"of tracks have tempo data (minimum {min_coverage:.1%} required).\n\n"
                f"💡 Solutions:\n"
                f"• Set w_tempo=0 to disable tempo-based scoring\n"
                f"• Re-run feature extraction with tempo analysis\n"
                f"• Use lower tempo weight (e.g., w_tempo=0.05) for sparse data\n\n"
                f"📊 Current tempo coverage: {meta_df['tempo'].notna().sum():,} of "
                f"{len(meta_df):,} tracks"
            )
    
    def validate_path_configuration(self, 
                                  feat_path: Path,
                                  index_path: Path, 
                                  meta_path: Path,
                                  model_path: Path) -> None:
        """
        Validate file paths and provide helpful guidance for missing files.
        
        Args:
            feat_path: Features file path
            index_path: Index file path
            meta_path: Metadata file path  
            model_path: Model file path
            
        Raises:
            ConfigurationError: If critical files are missing
        """
        missing_files = []
        path_info = []
        
        for name, path in [
            ("Features", feat_path),
            ("Index", index_path), 
            ("Metadata", meta_path),
            ("Model", model_path)
        ]:
            if not path.exists():
                missing_files.append((name, path))
            else:
                # Get file size for info
                size_mb = path.stat().st_size / (1024 * 1024)
                path_info.append(f"✅ {name}: {path} ({size_mb:.1f} MB)")
                
        if missing_files:
            missing_text = "\n".join([f"❌ {name}: {path}" for name, path in missing_files])
            found_text = "\n".join(path_info) if path_info else "None found"
            
            # Provide specific guidance
            guidance = []
            if any("features" in name.lower() for name, _ in missing_files):
                guidance.append("🔧 Run feature extraction: python -m src.data.ingest_fma")
            if any("model" in name.lower() for name, _ in missing_files):
                guidance.append("🔧 Build KNN model: python -m src.models.knn_recommender --build")
            if any("metadata" in name.lower() for name, _ in missing_files):
                guidance.append("🔧 Build metadata: python -m src.data.build_metadata")
                
            guidance_text = "\n".join(guidance) if guidance else "Check data pipeline setup"
            
            raise ConfigurationError(
                f"❌ Missing required files:\n\n{missing_text}\n\n"
                f"📁 Found files:\n{found_text}\n\n"
                f"💡 Next steps:\n{guidance_text}"
            )
            
        log.info(f"✅ All required files found:\n" + "\n".join(path_info))
    
    def validate_model_compatibility(self, 
                                   model_path: Path,
                                   features_df: pd.DataFrame) -> None:
        """
        Validate that saved model is compatible with current features.
        
        Args:
            model_path: Path to saved model
            features_df: Current features DataFrame
            
        Raises:
            ConfigurationError: If model is incompatible
        """
        if not model_path.exists():
            return  # Will be caught by path validation
            
        try:
            # Load model metadata
            payload = joblib.load(model_path)
            
            if "track_ids" in payload and len(features_df) > 0:
                model_tracks = set(payload["track_ids"])
                current_tracks = set(features_df["track_id"]) if "track_id" in features_df.columns else set()
                
                overlap = len(model_tracks & current_tracks)
                model_only = len(model_tracks - current_tracks)
                current_only = len(current_tracks - model_tracks)
                
                if overlap == 0:
                    raise ConfigurationError(
                        f"❌ Model incompatible: No track overlap between saved model and current features.\n\n"
                        f"📊 Model tracks: {len(model_tracks):,}\n"
                        f"📊 Current tracks: {len(current_tracks):,}\n"
                        f"📊 Overlap: {overlap:,}\n\n"
                        f"🔧 Solution: Rebuild model with current features:\n"
                        f"   python -m src.models.knn_recommender --build"
                    )
                    
                if current_only > len(model_tracks) * 0.5:  # >50% new tracks
                    log.warning(
                        f"⚠️  Model may be outdated: {current_only:,} new tracks not in model "
                        f"({current_only/len(current_tracks):.1%} of current dataset)"
                    )
                    
        except Exception as e:
            raise ConfigurationError(
                f"❌ Cannot validate model compatibility: {str(e)}\n\n"
                f"🔧 Try rebuilding the model:\n"
                f"   python -m src.models.knn_recommender --build"
            )


def create_validator() -> GuardrailValidator:
    """Create a new guardrail validator instance."""
    return GuardrailValidator()


def validate_recommendation_request(seed_track_id: int,
                                  top: int,
                                  available_tracks: List[int],
                                  index_df: Optional[pd.DataFrame] = None) -> None:
    """
    Validate a recommendation request with helpful error messages.
    
    Args:
        seed_track_id: The seed track ID
        top: Number of recommendations requested
        available_tracks: List of valid track IDs
        index_df: Optional index DataFrame for enhanced messages
        
    Raises:
        RecommendationSystemError: If validation fails
    """
    validator = create_validator()
    
    # Validate top parameter
    if top <= 0:
        raise RecommendationSystemError(
            f"❌ Invalid recommendation count: {top}\n\n"
            f"💡 Recommendation count must be positive (e.g., top=10)"
        )
        
    if top > len(available_tracks):
        log.warning(
            f"⚠️  Requested {top} recommendations but only {len(available_tracks)} tracks available. "
            f"Will return all {len(available_tracks)} tracks."
        )
        
    # Validate seed track
    validator.validate_seed_track(seed_track_id, available_tracks, index_df)
