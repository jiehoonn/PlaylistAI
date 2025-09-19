#!/usr/bin/env python3
"""
Versioning & Reproducibility System

Provides versioned artifacts, improved logging, and dataset validation
for reproducible machine learning workflows.

Key features:
- Automatic artifact versioning with timestamps
- Dataset fingerprinting for reproducibility
- Enhanced logging with performance metrics
- Build provenance tracking
- Version-aware model loading
"""

import hashlib
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import joblib
import pandas as pd
import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)


class ArtifactVersionManager:
    """Manages versioned artifacts and build provenance."""
    
    def __init__(self, base_artifacts_dir: Path = None):
        if base_artifacts_dir is None:
            self.artifacts_dir = Path(__file__).resolve().parents[2] / "artifacts"
        else:
            self.artifacts_dir = base_artifacts_dir
            
        self.artifacts_dir.mkdir(exist_ok=True)
        
    def create_versioned_path(self, 
                             base_name: str,
                             dataset_suffix: str = "small",
                             timestamp: Optional[str] = None) -> Path:
        """
        Create a versioned artifact path.
        
        Args:
            base_name: Base name (e.g., "knn_audio")
            dataset_suffix: Dataset identifier (e.g., "small", "large")
            timestamp: Optional timestamp (auto-generated if None)
            
        Returns:
            Path with format: {base_name}_v2_{dataset_suffix}_{YYYYMMDD_HHMMSS}.{ext}
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Extract extension from base_name if present
        base_path = Path(base_name)
        name_part = base_path.stem
        ext_part = base_path.suffix or ".joblib"
        
        versioned_name = f"{name_part}_v2_{dataset_suffix}_{timestamp}{ext_part}"
        return self.artifacts_dir / versioned_name
    
    def save_artifact_with_metadata(self,
                                   artifact: Any,
                                   base_name: str,
                                   dataset_suffix: str = "small",
                                   metadata: Optional[Dict[str, Any]] = None,
                                   dataset_fingerprint: Optional[str] = None) -> Path:
        """
        Save artifact with comprehensive metadata for reproducibility.
        
        Args:
            artifact: The artifact to save (model, data, etc.)
            base_name: Base artifact name
            dataset_suffix: Dataset identifier
            metadata: Additional metadata to store
            dataset_fingerprint: Hash of input data for reproducibility
            
        Returns:
            Path where artifact was saved
        """
        # Create versioned path
        artifact_path = self.create_versioned_path(base_name, dataset_suffix)
        
        # Create comprehensive metadata
        build_metadata = {
            "artifact_name": base_name,
            "dataset_suffix": dataset_suffix,
            "build_timestamp": datetime.now().isoformat(),
            "dataset_fingerprint": dataset_fingerprint,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "build_host": os.uname().nodename if hasattr(os, 'uname') else "unknown",
            "artifact_path": str(artifact_path),
            "reproducibility": {
                "random_seed": getattr(np.random, 'get_state', lambda: None)(),
                "environment": dict(os.environ) if len(os.environ) < 100 else {"note": "environment_too_large"}
            }
        }
        
        # Add custom metadata
        if metadata:
            build_metadata.update(metadata)
            
        # Save artifact with metadata
        artifact_with_metadata = {
            "artifact": artifact,
            "metadata": build_metadata
        }
        
        joblib.dump(artifact_with_metadata, artifact_path)
        
        # Also save metadata separately for easy access
        metadata_path = artifact_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(build_metadata, f, indent=2, default=str)
            
        log.info(f"Saved versioned artifact: {artifact_path}")
        log.info(f"Saved metadata: {metadata_path}")
        
        return artifact_path
    
    def load_latest_artifact(self, 
                           base_name: str,
                           dataset_suffix: str = "small") -> Tuple[Any, Dict[str, Any]]:
        """
        Load the most recent version of an artifact.
        
        Args:
            base_name: Base artifact name to search for
            dataset_suffix: Dataset identifier
            
        Returns:
            (artifact, metadata) tuple
        """
        # Find all matching artifacts
        pattern = f"{base_name}_v2_{dataset_suffix}_*"
        matching_files = list(self.artifacts_dir.glob(pattern))
        
        if not matching_files:
            raise FileNotFoundError(f"No artifacts found for pattern: {pattern}")
            
        # Sort by timestamp (newest first)
        matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_file = matching_files[0]
        
        log.info(f"Loading latest artifact: {latest_file}")
        
        # Load artifact
        artifact_data = joblib.load(latest_file)
        
        if isinstance(artifact_data, dict) and "artifact" in artifact_data:
            # New format with metadata
            return artifact_data["artifact"], artifact_data.get("metadata", {})
        else:
            # Legacy format
            log.warning(f"Loading legacy artifact format: {latest_file}")
            return artifact_data, {}
    
    def list_artifact_versions(self, 
                             base_name: str,
                             dataset_suffix: str = "small") -> List[Dict[str, Any]]:
        """List all versions of an artifact with metadata."""
        pattern = f"{base_name}_v2_{dataset_suffix}_*"
        matching_files = list(self.artifacts_dir.glob(pattern))
        
        versions = []
        for file_path in sorted(matching_files, key=lambda p: p.stat().st_mtime, reverse=True):
            metadata_path = file_path.with_suffix('.json')
            
            version_info = {
                "path": str(file_path),
                "size_mb": file_path.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
            # Load metadata if available
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    version_info["metadata"] = metadata
                except Exception as e:
                    log.warning(f"Could not load metadata for {file_path}: {e}")
                    
            versions.append(version_info)
            
        return versions


class DatasetFingerprinter:
    """Creates reproducible fingerprints of datasets for change detection."""
    
    @staticmethod
    def fingerprint_dataframe(df: pd.DataFrame, 
                            include_values: bool = True,
                            sample_size: int = 1000) -> str:
        """
        Create a fingerprint of a DataFrame for reproducibility tracking.
        
        Args:
            df: DataFrame to fingerprint
            include_values: Whether to include actual data values (vs just structure)
            sample_size: Number of rows to sample for value-based fingerprinting
            
        Returns:
            Hex string fingerprint
        """
        hasher = hashlib.sha256()
        
        # DataFrame structure
        hasher.update(f"shape:{df.shape}".encode())
        hasher.update(f"columns:{sorted(df.columns.tolist())}".encode())
        hasher.update(f"dtypes:{df.dtypes.to_dict()}".encode())
        hasher.update(f"index_type:{type(df.index).__name__}".encode())
        
        if include_values and len(df) > 0:
            # Sample data for fingerprinting (deterministic sampling)
            if len(df) > sample_size:
                # Use fixed seed for reproducible sampling
                sample_df = df.sample(n=sample_size, random_state=42)
            else:
                sample_df = df
                
            # Hash sampled values
            for col in sorted(df.columns):
                if col in sample_df.columns:
                    try:
                        # Convert to string representation for hashing
                        col_data = sample_df[col].astype(str).sort_values()
                        hasher.update(f"{col}:{col_data.sum()}".encode())
                    except Exception:
                        # Fallback for complex data types
                        hasher.update(f"{col}:complex_type".encode())
                        
        return hasher.hexdigest()[:16]  # First 16 chars for readability
    
    @staticmethod
    def fingerprint_files(file_paths: List[Path]) -> str:
        """Create fingerprint based on file metadata and checksums."""
        hasher = hashlib.sha256()
        
        for file_path in sorted(file_paths):
            if file_path.exists():
                # File metadata
                stat = file_path.stat()
                hasher.update(f"{file_path.name}:{stat.st_size}:{stat.st_mtime}".encode())
                
                # Quick checksum of first/last KB for large files
                try:
                    with open(file_path, 'rb') as f:
                        # First KB
                        first_chunk = f.read(1024)
                        hasher.update(first_chunk)
                        
                        # Last KB for files > 2KB
                        if stat.st_size > 2048:
                            f.seek(-1024, 2)  # Seek to last 1KB
                            last_chunk = f.read(1024)
                            hasher.update(last_chunk)
                except Exception:
                    # Fallback to just file size/time
                    hasher.update(f"fallback:{stat.st_size}".encode())
            else:
                hasher.update(f"{file_path.name}:missing".encode())
                
        return hasher.hexdigest()[:16]


class ReproducibilityTracker:
    """Tracks reproducibility information for ML experiments."""
    
    def __init__(self):
        self.version_manager = ArtifactVersionManager()
        self.fingerprinter = DatasetFingerprinter()
        
    def create_build_context(self,
                           features_df: pd.DataFrame,
                           meta_df: pd.DataFrame,
                           index_df: pd.DataFrame,
                           additional_files: List[Path] = None) -> Dict[str, Any]:
        """
        Create comprehensive build context for reproducibility.
        
        Args:
            features_df: Features DataFrame
            meta_df: Metadata DataFrame
            index_df: Index DataFrame
            additional_files: Additional files to include in fingerprint
            
        Returns:
            Build context dictionary
        """
        import sys
        import os
        import platform
        
        # Dataset fingerprints
        features_fingerprint = self.fingerprinter.fingerprint_dataframe(features_df)
        meta_fingerprint = self.fingerprinter.fingerprint_dataframe(meta_df)
        index_fingerprint = self.fingerprinter.fingerprint_dataframe(index_df)
        
        # File fingerprints
        files_to_check = additional_files or []
        files_fingerprint = self.fingerprinter.fingerprint_files(files_to_check)
        
        # System context
        build_context = {
            "dataset_fingerprints": {
                "features": features_fingerprint,
                "metadata": meta_fingerprint,
                "index": index_fingerprint,
                "files": files_fingerprint
            },
            "system_info": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "architecture": platform.architecture(),
                "processor": platform.processor(),
                "hostname": platform.node()
            },
            "package_versions": self._get_package_versions(),
            "build_timestamp": datetime.now().isoformat(),
            "data_stats": {
                "features_shape": features_df.shape,
                "metadata_shape": meta_df.shape,
                "index_shape": index_df.shape,
                "total_tracks": len(features_df)
            }
        }
        
        return build_context
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages for reproducibility."""
        packages = ['numpy', 'pandas', 'scikit-learn', 'joblib', 'hnswlib']
        versions = {}
        
        for package in packages:
            try:
                module = __import__(package)
                versions[package] = getattr(module, '__version__', 'unknown')
            except ImportError:
                versions[package] = 'not_installed'
                
        return versions
    
    def save_model_with_provenance(self,
                                 model: Any,
                                 model_name: str,
                                 build_context: Dict[str, Any],
                                 training_params: Dict[str, Any] = None) -> Path:
        """
        Save model with full provenance tracking.
        
        Args:
            model: The trained model
            model_name: Name for the model
            build_context: Build context from create_build_context()
            training_params: Parameters used for training
            
        Returns:
            Path where model was saved
        """
        # Combine all metadata
        full_metadata = {
            "model_type": type(model).__name__,
            "training_params": training_params or {},
            "build_context": build_context,
            "reproducibility_note": "This model was built with full provenance tracking"
        }
        
        # Create composite fingerprint for the entire build
        composite_fingerprint = hashlib.sha256(
            json.dumps(build_context["dataset_fingerprints"], sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return self.version_manager.save_artifact_with_metadata(
            model, model_name, "small", full_metadata, composite_fingerprint
        )


# Convenience functions
def create_versioned_model_path(model_name: str = "knn_audio") -> Path:
    """Create a versioned path for a model."""
    manager = ArtifactVersionManager()
    return manager.create_versioned_path(model_name)


def save_model_with_reproducibility(model: Any,
                                   features_df: pd.DataFrame,
                                   meta_df: pd.DataFrame,
                                   index_df: pd.DataFrame,
                                   model_name: str = "knn_audio",
                                   training_params: Dict[str, Any] = None) -> Path:
    """
    Save a model with full reproducibility tracking.
    
    Args:
        model: The model to save
        features_df: Features used for training
        meta_df: Metadata used
        index_df: Index used
        model_name: Name for the model
        training_params: Training parameters used
        
    Returns:
        Path where model was saved
    """
    tracker = ReproducibilityTracker()
    build_context = tracker.create_build_context(features_df, meta_df, index_df)
    return tracker.save_model_with_provenance(model, model_name, build_context, training_params)


def load_latest_model(model_name: str = "knn_audio") -> Tuple[Any, Dict[str, Any]]:
    """Load the latest version of a model with metadata."""
    manager = ArtifactVersionManager()
    return manager.load_latest_artifact(model_name)


def list_model_versions(model_name: str = "knn_audio") -> List[Dict[str, Any]]:
    """List all versions of a model."""
    manager = ArtifactVersionManager()
    return manager.list_artifact_versions(model_name)


