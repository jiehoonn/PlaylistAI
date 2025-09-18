"""
Tests for data pipeline components (ingest, features, metadata).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.fma_paths import track_id_to_relpath
from src.features.audio_features import mfcc_mean_std


class TestFMAPaths:
    """Test FMA path mapping functionality."""
    
    def test_track_id_to_relpath_format(self):
        """Test that track ID to relative path conversion follows expected format."""
        # Test various track IDs - function returns Path objects
        assert str(track_id_to_relpath(2)) == "000/000002.mp3"
        assert str(track_id_to_relpath(99999)) == "099/099999.mp3"
        assert str(track_id_to_relpath(123456)) == "123/123456.mp3"
    
    def test_track_id_to_relpath_padding(self):
        """Test that track IDs are properly zero-padded."""
        assert str(track_id_to_relpath(1)) == "000/000001.mp3"
        assert str(track_id_to_relpath(42)) == "000/000042.mp3"
        assert str(track_id_to_relpath(9999)) == "009/009999.mp3"


class TestAudioFeatures:
    """Test audio feature extraction components."""
    
    @patch('librosa.load')
    @patch('librosa.feature.mfcc')
    def test_mfcc_mean_std_shape(self, mock_mfcc, mock_load):
        """Test that MFCC extraction returns correct shape."""
        # Mock librosa functions
        mock_load.return_value = (np.random.randn(44100), 22050)  # 1 second of audio
        mock_mfcc.return_value = np.random.randn(13, 100)  # 13 MFCCs, 100 time frames
        
        # Test feature extraction
        features = mfcc_mean_std("/fake/path.mp3")
        
        # Should return 26 features (13 means + 13 stds)
        assert len(features) == 26
        assert all(isinstance(f, float) for f in features)
    
    @patch('librosa.load')
    def test_mfcc_error_handling(self, mock_load):
        """Test that feature extraction handles audio loading errors gracefully."""
        # Mock librosa to raise an exception
        mock_load.side_effect = Exception("Audio loading failed")
        
        # Should return None on error
        result = mfcc_mean_std("/fake/path.mp3")
        assert result is None


class TestDataIntegrity:
    """Test data integrity and consistency across pipeline stages."""
    
    def test_index_data_types(self, sample_index_file):
        """Test that index file has correct data types."""
        df = pd.read_parquet(sample_index_file)
        
        # Check required columns exist
        required_cols = ['track_id', 'artist_name', 'track_title', 'audio_path']
        assert all(col in df.columns for col in required_cols)
        
        # Check data types
        assert df['track_id'].dtype in [np.int64, np.int32]
        assert df['artist_name'].dtype == object
        assert df['track_title'].dtype == object
        assert df['audio_path'].dtype == object
        
        # Check no null values in required fields
        assert not df['track_id'].isnull().any()
        assert not df['artist_name'].isnull().any()
        assert not df['track_title'].isnull().any()
    
    def test_features_data_integrity(self, sample_features_file):
        """Test that features file has correct structure."""
        df = pd.read_parquet(sample_features_file)
        
        # Check structure
        assert 'track_id' in df.columns
        assert 'feature' in df.columns
        assert len(df.columns) == 2
        
        # Check feature dimensions
        for _, row in df.iterrows():
            feature = row['feature']
            assert isinstance(feature, (list, np.ndarray))
            assert len(feature) == 61  # v2 features are 61-dimensional
            assert all(isinstance(f, (int, float, np.number)) for f in feature)
    
    def test_metadata_completeness(self, sample_metadata_file):
        """Test metadata file completeness and consistency."""
        df = pd.read_parquet(sample_metadata_file)
        
        # Check required columns
        expected_cols = ['track_id', 'artist_name', 'track_title']
        assert all(col in df.columns for col in expected_cols)
        
        # Check genres_all is properly formatted
        if 'genres_all' in df.columns:
            for _, row in df.iterrows():
                genres = row['genres_all']
                assert isinstance(genres, (list, np.ndarray))
                if isinstance(genres, np.ndarray):
                    genres = genres.tolist()
                assert all(isinstance(g, str) for g in genres)
        
        # Check tempo values are reasonable if present
        if 'tempo' in df.columns:
            tempo_values = df['tempo'].dropna()
            if len(tempo_values) > 0:
                assert all(60 <= t <= 200 for t in tempo_values), "Tempo values should be reasonable (60-200 BPM)"


class TestDataConsistency:
    """Test consistency across different data files."""
    
    def test_track_id_consistency(self, sample_index_file, sample_features_file, sample_metadata_file):
        """Test that track IDs are consistent across all data files."""
        index_df = pd.read_parquet(sample_index_file)
        features_df = pd.read_parquet(sample_features_file)
        metadata_df = pd.read_parquet(sample_metadata_file)
        
        index_ids = set(index_df['track_id'])
        feature_ids = set(features_df['track_id'])
        metadata_ids = set(metadata_df['track_id'])
        
        # Features should be subset of index (some tracks might fail feature extraction)
        assert feature_ids.issubset(index_ids), "Feature track IDs should be subset of index"
        
        # Metadata should match index (or be subset due to missing data)
        assert metadata_ids.issubset(index_ids), "Metadata track IDs should be subset of index"
    
    def test_no_duplicate_track_ids(self, sample_index_file, sample_features_file):
        """Test that there are no duplicate track IDs in data files."""
        index_df = pd.read_parquet(sample_index_file)
        features_df = pd.read_parquet(sample_features_file)
        
        assert not index_df['track_id'].duplicated().any(), "Index should have no duplicate track IDs"
        assert not features_df['track_id'].duplicated().any(), "Features should have no duplicate track IDs"
