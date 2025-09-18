"""
Tests for KNN recommender functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import joblib
import tempfile

from src.models.knn_recommender import (
    _load_features, _fit_transform, recommend, recommend_hybrid, 
    build_playlist, _jaccard
)


class TestKNNCore:
    """Test core KNN functionality."""
    
    def test_load_features(self, sample_features_file):
        """Test feature loading and preprocessing."""
        df, X, track_ids = _load_features(sample_features_file)
        
        # Check return types and shapes
        assert isinstance(df, pd.DataFrame)
        assert isinstance(X, np.ndarray)
        assert isinstance(track_ids, np.ndarray)
        
        # Check dimensions
        assert X.shape[0] == len(track_ids)
        assert X.shape[1] == 61  # v2 features
        assert len(df) == len(track_ids)
    
    def test_fit_transform(self, sample_features_file):
        """Test feature scaling and normalization."""
        _, X, _ = _load_features(sample_features_file)
        X_norm, scaler = _fit_transform(X)
        
        # Check output properties
        assert X_norm.shape == X.shape
        # scaler should be a StandardScaler instance
        from sklearn.preprocessing import StandardScaler
        assert isinstance(scaler, StandardScaler)
        
        # Check normalization (L2 norm should be ~1 for each row)
        row_norms = np.linalg.norm(X_norm, axis=1)
        assert np.allclose(row_norms, 1.0, atol=1e-6), "Rows should be L2 normalized"
        
        # Check that mean is approximately 0 after scaling (before normalization)
        X_scaled = scaler.transform(X)
        assert np.abs(X_scaled.mean()) < 0.1, "Scaled features should have low mean"


class TestRecommendations:
    """Test recommendation algorithms."""
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        set_a = {'rock', 'alternative'}
        set_b = {'rock', 'pop'}
        set_c = {'jazz'}
        
        # Test overlap
        assert _jaccard(set_a, set_b) == 1/3  # 1 intersection, 3 union
        
        # Test no overlap
        assert _jaccard(set_a, set_c) == 0.0
        
        # Test identical sets
        assert _jaccard(set_a, set_a) == 1.0
        
        # Test empty sets
        assert _jaccard(set(), set()) == 1.0  # Two empty sets are perfectly similar
        assert _jaccard(set_a, set()) == 0.0  # Non-empty vs empty = no similarity
    
    @patch('src.models.knn_recommender.joblib.load')
    @patch('src.models.knn_recommender._matrix_in_saved_order')
    def test_recommend_structure(self, mock_matrix, mock_joblib, sample_features_file):
        """Test that recommend function returns proper structure."""
        # Mock the model loading
        mock_model = MagicMock()
        mock_model.kneighbors.return_value = (
            np.array([[0.1, 0.2, 0.3]]),  # distances
            np.array([[1, 2, 3]])          # indices
        )
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.random.randn(1, 61)
        
        mock_joblib.return_value = {
            'model': mock_model,
            'scaler': mock_scaler,
            'track_ids': np.array([1, 2, 3, 4, 5])  # Use track IDs that exist in sample data
        }
        
        # Mock the matrix reconstruction
        mock_matrix.return_value = (np.random.randn(5, 61), {1: 0, 2: 1, 3: 2, 4: 3, 5: 4})
        
        # Test recommendation
        rec_ids, distances = recommend(seed_track_id=1, k=3)
        
        # Check return structure
        assert isinstance(rec_ids, list)
        assert isinstance(distances, list)
        assert len(rec_ids) == 3
        assert len(distances) == 3
        assert all(isinstance(rid, (int, np.integer)) for rid in rec_ids)
        assert all(isinstance(d, (float, np.floating)) for d in distances)


class TestPlaylistGeneration:
    """Test playlist generation with MMR."""
    
    def test_playlist_length_constraint(self):
        """Test that playlist respects length constraints."""
        # Mock the dependencies since we can't easily test the full pipeline
        with patch('src.models.knn_recommender.recommend_hybrid') as mock_hybrid, \
             patch('src.models.knn_recommender.joblib.load') as mock_joblib, \
             patch('src.models.knn_recommender.pd.read_parquet') as mock_read:
            
            # Setup mocks
            mock_hybrid.return_value = (list(range(20)), [1.0] * 20)  # 20 candidates
            mock_joblib.return_value = {
                'scaler': MagicMock(),
            }
            mock_read.return_value = pd.DataFrame({
                'track_id': list(range(20)),
                'artist_name': [f'Artist_{i}' for i in range(20)],
                'album_title': [f'Album_{i}' for i in range(20)]
            })
            
            # Mock feature loading
            with patch('src.models.knn_recommender._features_for') as mock_features:
                mock_features.return_value = np.random.randn(20, 61)
                
                # Test playlist generation
                playlist = build_playlist(seed_track_id=0, n=10, max_per_artist=1)
                
                # Check constraints
                assert len(playlist) <= 10, "Playlist should not exceed requested length"
                assert all(isinstance(tid, (int, np.integer)) for tid in playlist)
    
    def test_artist_diversity_constraint(self):
        """Test that playlist respects artist diversity constraints."""
        with patch('src.models.knn_recommender.recommend_hybrid') as mock_hybrid, \
             patch('src.models.knn_recommender.joblib.load') as mock_joblib, \
             patch('src.models.knn_recommender.pd.read_parquet') as mock_read:
            
            # Setup mocks with repeated artists
            mock_hybrid.return_value = (list(range(10)), [1.0] * 10)
            mock_joblib.return_value = {'scaler': MagicMock()}
            mock_read.return_value = pd.DataFrame({
                'track_id': list(range(10)),
                'artist_name': ['Artist_A'] * 5 + ['Artist_B'] * 5,  # Only 2 artists
                'album_title': [f'Album_{i}' for i in range(10)]
            })
            
            with patch('src.models.knn_recommender._features_for') as mock_features:
                mock_features.return_value = np.random.randn(10, 61)
                
                playlist = build_playlist(seed_track_id=0, n=10, max_per_artist=2)
                
                # Should have at most 4 tracks (2 per artist × 2 artists)
                assert len(playlist) <= 4


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_seed_track(self):
        """Test behavior when seed track is not found."""
        # Test with minimal mock to avoid complex dependencies
        with patch('src.models.knn_recommender.joblib.load') as mock_joblib:
            mock_joblib.return_value = {
                'model': MagicMock(),
                'scaler': MagicMock(),
                'track_ids': np.array([1, 2, 3, 4, 5])  # Seed 999 not in this list
            }
            
            # This should raise an error because seed 999 is not in track_ids
            with pytest.raises((ValueError, KeyError, IndexError, RuntimeError)):
                recommend(seed_track_id=999, k=5)
    
    def test_empty_playlist_candidates(self):
        """Test playlist generation with no valid candidates."""
        with patch('src.models.knn_recommender.recommend_hybrid') as mock_hybrid:
            mock_hybrid.return_value = ([], [])  # No candidates
            
            # Should handle empty candidates gracefully
            with pytest.raises((ValueError, IndexError)):
                build_playlist(seed_track_id=0, n=10)


class TestRecommendationQuality:
    """Test recommendation quality and consistency."""
    
    def test_distance_sorting_logic(self):
        """Test basic distance sorting logic."""
        # Simple unit test for sorting behavior
        distances = [0.3, 0.1, 0.2]
        indices = [2, 0, 1]
        
        # Sort by distance
        sorted_pairs = sorted(zip(distances, indices))
        sorted_distances = [d for d, _ in sorted_pairs]
        
        assert sorted_distances == [0.1, 0.2, 0.3]
    
    def test_basic_recommendation_properties(self):
        """Test basic properties that recommendations should have."""
        # Mock a simple case
        with patch('src.models.knn_recommender.recommend_hybrid') as mock_hybrid:
            mock_hybrid.return_value = ([1, 2, 3], [0.9, 0.8, 0.7])
            
            with patch('src.models.knn_recommender.joblib.load') as mock_joblib:
                mock_joblib.return_value = {'scaler': MagicMock()}
                
                with patch('src.models.knn_recommender.pd.read_parquet') as mock_read:
                    mock_read.return_value = pd.DataFrame({
                        'track_id': [1, 2, 3],
                        'artist_name': ['A', 'B', 'C'],
                        'album_title': ['X', 'Y', 'Z']
                    })
                    
                    with patch('src.models.knn_recommender._features_for') as mock_features:
                        mock_features.return_value = np.random.randn(3, 61)
                        
                        playlist = build_playlist(seed_track_id=1, n=3)
                        
                        # Basic properties
                        assert isinstance(playlist, list)
                        assert len(playlist) <= 3
                        assert all(isinstance(tid, (int, np.integer)) for tid in playlist)
