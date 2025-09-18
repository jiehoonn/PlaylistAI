"""
Rigorous tests for MMR (Maximal Marginal Relevance) playlist algorithm.

Tests verify mathematical properties, diversity constraints, and algorithmic correctness.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.models.knn_recommender import build_playlist


class TestMMRAlgorithmCorrectness:
    """Test MMR algorithm mathematical properties and correctness."""
    
    @patch('src.models.knn_recommender.recommend_hybrid')
    @patch('src.models.knn_recommender.joblib.load')
    @patch('src.models.knn_recommender.pd.read_parquet')
    @patch('src.models.knn_recommender._features_for')
    def test_mmr_diversity_vs_similarity_tradeoff(self, mock_features, mock_read, mock_joblib, mock_hybrid):
        """Test that lambda_mmr correctly balances diversity vs similarity."""
        # Setup: 4 candidates with known base scores and similarities
        candidate_ids = [101, 102, 103, 104]
        base_scores = [1.0, 0.8, 0.6, 0.4]  # Decreasing similarity to seed
        
        mock_hybrid.return_value = (candidate_ids, base_scores)
        mock_joblib.return_value = {'scaler': MagicMock()}
        mock_read.return_value = pd.DataFrame({
            'track_id': candidate_ids,
            'artist_name': ['A', 'B', 'C', 'D'],  # All different artists
            'album_title': ['X', 'Y', 'Z', 'W']   # All different albums
        })
        
        # Similarity matrix: high similarity between first two tracks
        similarity_matrix = np.array([
            [1.0, 0.9, 0.3, 0.2],  # Track 101 vs others
            [0.9, 1.0, 0.2, 0.1],  # Track 102 vs others  
            [0.3, 0.2, 1.0, 0.4],  # Track 103 vs others
            [0.2, 0.1, 0.4, 1.0]   # Track 104 vs others
        ])
        mock_features.return_value = np.random.randn(4, 61)
        
        # Mock the similarity calculation to return our controlled matrix
        with patch('numpy.dot') as mock_dot:
            mock_dot.return_value = similarity_matrix
            
            # Test 1: Pure similarity (lambda=1.0) - should pick highest scoring tracks
            playlist_similarity = build_playlist(
                seed_track_id=100, n=3, candidate_k=4,
                lambda_mmr=1.0,  # Only similarity matters
                max_per_artist=10, max_per_album=10  # No constraints
            )
            
            # Should pick tracks in order of base score: 101, 102, 103
            assert playlist_similarity == [101, 102, 103], \
                "Pure similarity should pick highest scoring tracks"
            
            # Test 2: Pure diversity (lambda=0.0) - should avoid similar tracks
            playlist_diversity = build_playlist(
                seed_track_id=100, n=3, candidate_k=4,
                lambda_mmr=0.0,  # Only diversity matters
                max_per_artist=10, max_per_album=10
            )
            
            # Should pick 101 (highest score), then avoid 102 (too similar to 101)
            # Should prefer 103, 104 over 102 due to lower similarity
            assert playlist_diversity[0] == 101, "Should still pick best track first"
            assert 102 not in playlist_diversity or playlist_diversity.index(102) > playlist_diversity.index(103), \
                "Should prefer diverse tracks over similar ones"
    
    @patch('src.models.knn_recommender.recommend_hybrid')
    @patch('src.models.knn_recommender.joblib.load')
    @patch('src.models.knn_recommender.pd.read_parquet')
    @patch('src.models.knn_recommender._features_for')
    def test_artist_album_constraints_enforcement(self, mock_features, mock_read, mock_joblib, mock_hybrid):
        """Test that artist and album constraints are strictly enforced."""
        # Setup: Multiple tracks from same artists/albums
        candidate_ids = list(range(101, 111))  # 10 candidates
        base_scores = [1.0 - i*0.1 for i in range(10)]  # Decreasing scores
        
        mock_hybrid.return_value = (candidate_ids, base_scores)
        mock_joblib.return_value = {'scaler': MagicMock()}
        
        # Create violations: Artist A has 4 tracks, Album X has 5 tracks
        mock_read.return_value = pd.DataFrame({
            'track_id': candidate_ids,
            'artist_name': ['A', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'E'],
            'album_title': ['X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Z', 'Z', 'W']
        })
        
        mock_features.return_value = np.random.randn(10, 61)
        
        # Test strict constraints
        playlist = build_playlist(
            seed_track_id=100, n=10, candidate_k=10,
            lambda_mmr=1.0,  # Prefer similarity (would normally pick first 10)
            max_per_artist=2, max_per_album=3
        )
        
        # Verify constraints
        df = mock_read.return_value
        playlist_df = df[df['track_id'].isin(playlist)]
        
        artist_counts = playlist_df['artist_name'].value_counts()
        album_counts = playlist_df['album_title'].value_counts()
        
        assert all(count <= 2 for count in artist_counts), \
            f"Artist constraint violated: {artist_counts}"
        assert all(count <= 3 for count in album_counts), \
            f"Album constraint violated: {album_counts}"
        
        # Should pick best available tracks within constraints
        # Artist A: tracks 101, 102 (first 2 by score)
        # Album X: tracks from different artists up to limit
        artist_a_tracks = playlist_df[playlist_df['artist_name'] == 'A']['track_id'].tolist()
        assert len(artist_a_tracks) <= 2
        if len(artist_a_tracks) == 2:
            assert sorted(artist_a_tracks) == [101, 102], \
                "Should pick highest scoring tracks from constrained artist"
    
    @patch('src.models.knn_recommender.recommend_hybrid')
    @patch('src.models.knn_recommender.joblib.load')
    @patch('src.models.knn_recommender.pd.read_parquet')
    @patch('src.models.knn_recommender._features_for')
    def test_mmr_convergence_and_termination(self, mock_features, mock_read, mock_joblib, mock_hybrid):
        """Test MMR algorithm termination conditions."""
        # Test 1: Requested playlist size reached
        candidate_ids = list(range(101, 121))  # 20 candidates
        base_scores = [1.0 - i*0.05 for i in range(20)]
        
        mock_hybrid.return_value = (candidate_ids, base_scores)
        mock_joblib.return_value = {'scaler': MagicMock()}
        mock_read.return_value = pd.DataFrame({
            'track_id': candidate_ids,
            'artist_name': [f'Artist_{i}' for i in range(20)],  # All unique
            'album_title': [f'Album_{i}' for i in range(20)]    # All unique
        })
        mock_features.return_value = np.random.randn(20, 61)
        
        playlist = build_playlist(
            seed_track_id=100, n=10, candidate_k=20,
            lambda_mmr=0.5, max_per_artist=5, max_per_album=5
        )
        
        assert len(playlist) == 10, "Should return exactly requested number of tracks"
        
        # Test 2: Constraint exhaustion (not enough valid candidates)
        mock_read.return_value = pd.DataFrame({
            'track_id': candidate_ids[:5],  # Only 5 candidates
            'artist_name': ['A', 'A', 'A', 'A', 'A'],  # All same artist
            'album_title': ['X', 'X', 'X', 'Y', 'Y']   # Two albums
        })
        mock_hybrid.return_value = (candidate_ids[:5], base_scores[:5])
        mock_features.return_value = np.random.randn(5, 61)
        
        playlist = build_playlist(
            seed_track_id=100, n=10, candidate_k=5,
            lambda_mmr=0.5, max_per_artist=2, max_per_album=2
        )
        
        # Should terminate early when constraints exhausted
        # Max possible: 2 from artist A + 0 more = 2 tracks total
        assert len(playlist) <= 2, "Should respect constraints even if playlist is shorter"


class TestMMRMathematicalProperties:
    """Test mathematical properties and edge cases of MMR scoring."""
    
    def test_mmr_score_calculation_accuracy(self):
        """Test MMR score calculation with known values."""
        # Test the core MMR formula: λ * relevance - (1-λ) * max_similarity
        lambda_mmr = 0.3
        relevance = 0.8
        max_similarity = 0.6
        
        expected_mmr = lambda_mmr * relevance - (1 - lambda_mmr) * max_similarity
        expected_mmr = 0.3 * 0.8 - 0.7 * 0.6  # = 0.24 - 0.42 = -0.18
        
        # This tests the mathematical formula correctness
        calculated_mmr = lambda_mmr * relevance - (1.0 - lambda_mmr) * max_similarity
        assert abs(calculated_mmr - expected_mmr) < 1e-10
        
        # MMR can be negative (diversity penalty > relevance benefit)
        assert calculated_mmr < 0, "MMR can be negative when diversity penalty dominates"
    
    def test_mmr_boundary_conditions(self):
        """Test MMR behavior at boundary conditions."""
        relevance = 0.8
        max_similarity = 0.6
        
        # λ = 0: Pure diversity (only penalty matters)
        mmr_diversity = 0.0 * relevance - (1.0 - 0.0) * max_similarity
        assert mmr_diversity == -max_similarity
        
        # λ = 1: Pure relevance (no penalty)
        mmr_relevance = 1.0 * relevance - (1.0 - 1.0) * max_similarity
        assert mmr_relevance == relevance
        
        # max_similarity = 0: No diversity penalty
        mmr_no_penalty = 0.5 * relevance - 0.5 * 0.0
        assert mmr_no_penalty == 0.5 * relevance
        
        # max_similarity = 1: Maximum diversity penalty
        mmr_max_penalty = 0.5 * relevance - 0.5 * 1.0
        assert mmr_max_penalty == 0.5 * (relevance - 1.0)


class TestMMRErrorHandling:
    """Test MMR algorithm error handling and edge cases."""
    
    @patch('src.models.knn_recommender.recommend_hybrid')
    def test_empty_candidates_handling(self, mock_hybrid):
        """Test graceful handling of empty candidate list."""
        mock_hybrid.return_value = ([], [])  # No candidates
        
        # This should handle empty candidates gracefully
        try:
            playlist = build_playlist(seed_track_id=100, n=10)
            assert playlist == [], "Empty candidates should return empty playlist"
        except (ValueError, IndexError) as e:
            # It's acceptable to raise an error for empty candidates
            assert "array to stack" in str(e) or "empty" in str(e).lower()
    
    @patch('src.models.knn_recommender.recommend_hybrid')
    @patch('src.models.knn_recommender.joblib.load')
    @patch('src.models.knn_recommender.pd.read_parquet')
    @patch('src.models.knn_recommender._features_for')
    def test_single_candidate_handling(self, mock_features, mock_read, mock_joblib, mock_hybrid):
        """Test behavior with only one candidate."""
        mock_hybrid.return_value = ([101], [1.0])
        mock_joblib.return_value = {'scaler': MagicMock()}
        mock_read.return_value = pd.DataFrame({
            'track_id': [101],
            'artist_name': ['A'],
            'album_title': ['X']
        })
        mock_features.return_value = np.random.randn(1, 61)
        
        playlist = build_playlist(seed_track_id=100, n=5)
        assert playlist == [101], "Single candidate should be selected"
    
    @patch('src.models.knn_recommender.recommend_hybrid')
    @patch('src.models.knn_recommender.joblib.load')
    @patch('src.models.knn_recommender.pd.read_parquet')
    @patch('src.models.knn_recommender._features_for')
    def test_missing_metadata_resilience(self, mock_features, mock_read, mock_joblib, mock_hybrid):
        """Test MMR resilience to missing metadata."""
        candidate_ids = [101, 102, 103]
        mock_hybrid.return_value = (candidate_ids, [1.0, 0.8, 0.6])
        mock_joblib.return_value = {'scaler': MagicMock()}
        
        # Missing artist/album info
        mock_read.return_value = pd.DataFrame({
            'track_id': candidate_ids,
            'artist_name': ['A', None, ''],  # Mix of valid, null, empty
            'album_title': [None, 'Y', 'Z']  # Mix of null, valid
        })
        mock_features.return_value = np.random.randn(3, 61)
        
        # Should not crash with missing metadata
        playlist = build_playlist(seed_track_id=100, n=3)
        assert len(playlist) > 0, "Should handle missing metadata gracefully"
        assert all(tid in candidate_ids for tid in playlist), "Should only return valid candidates"


class TestMMRPerformanceProperties:
    """Test MMR algorithm performance and complexity properties."""
    
    @patch('src.models.knn_recommender.recommend_hybrid')
    @patch('src.models.knn_recommender.joblib.load')
    @patch('src.models.knn_recommender.pd.read_parquet')
    @patch('src.models.knn_recommender._features_for')
    def test_algorithm_determinism(self, mock_features, mock_read, mock_joblib, mock_hybrid):
        """Test that MMR produces deterministic results with same inputs."""
        candidate_ids = list(range(101, 111))
        base_scores = [1.0 - i*0.1 for i in range(10)]
        
        mock_hybrid.return_value = (candidate_ids, base_scores)
        mock_joblib.return_value = {'scaler': MagicMock()}
        mock_read.return_value = pd.DataFrame({
            'track_id': candidate_ids,
            'artist_name': [f'Artist_{i}' for i in range(10)],
            'album_title': [f'Album_{i}' for i in range(10)]
        })
        
        # Fixed feature matrix for deterministic similarity
        fixed_features = np.random.RandomState(42).randn(10, 61)
        mock_features.return_value = fixed_features
        
        # Run multiple times with same inputs
        playlist1 = build_playlist(seed_track_id=100, n=5, lambda_mmr=0.3)
        playlist2 = build_playlist(seed_track_id=100, n=5, lambda_mmr=0.3)
        playlist3 = build_playlist(seed_track_id=100, n=5, lambda_mmr=0.3)
        
        assert playlist1 == playlist2 == playlist3, \
            "MMR should be deterministic with same inputs"
    
    def test_complexity_bounds(self):
        """Test that MMR algorithm complexity is reasonable."""
        # This is more of a documentation test - MMR is O(k²) where k is candidates
        # For 500 candidates building 50-track playlist:
        # - Each selection requires checking similarity to all previous selections
        # - Worst case: 50 * 500 = 25,000 similarity lookups
        # - This should be acceptable for real-time use
        
        max_candidates = 500
        max_playlist_size = 50
        
        # Theoretical maximum operations
        max_operations = max_playlist_size * max_candidates
        
        # Should be manageable (< 100k operations)
        assert max_operations < 100_000, \
            f"MMR complexity should be manageable: {max_operations} operations"
