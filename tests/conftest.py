"""
Pytest configuration and shared fixtures for PlaylistAI tests.
"""

import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_index_data():
    """Sample track index data for testing."""
    return {
        'track_id': [1, 2, 3, 4, 5],
        'artist_name': ['Artist A', 'Artist B', 'Artist A', 'Artist C', 'Artist B'],
        'track_title': ['Song 1', 'Song 2', 'Song 3', 'Song 4', 'Song 5'],
        'album_title': ['Album X', 'Album Y', 'Album Z', 'Album W', 'Album Y'],
        'track_genre_top': ['Rock', 'Pop', 'Rock', 'Jazz', 'Pop'],
        'relpath': ['000/000001.mp3', '000/000002.mp3', '000/000003.mp3', '000/000004.mp3', '000/000005.mp3'],
        'audio_path': ['/fake/path/000001.mp3', '/fake/path/000002.mp3', '/fake/path/000003.mp3', '/fake/path/000004.mp3', '/fake/path/000005.mp3']
    }


@pytest.fixture
def sample_features_data():
    """Sample audio features data for testing."""
    np.random.seed(42)
    return {
        'track_id': [1, 2, 3, 4, 5],
        'feature': [
            np.random.randn(61).tolist(),
            np.random.randn(61).tolist(),
            np.random.randn(61).tolist(),
            np.random.randn(61).tolist(),
            np.random.randn(61).tolist()
        ]
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        'track_id': [1, 2, 3, 4, 5],
        'artist_name': ['Artist A', 'Artist B', 'Artist A', 'Artist C', 'Artist B'],
        'track_title': ['Song 1', 'Song 2', 'Song 3', 'Song 4', 'Song 5'],
        'album_title': ['Album X', 'Album Y', 'Album Z', 'Album W', 'Album Y'],
        'genre_top': ['Rock', 'Pop', 'Rock', 'Jazz', 'Pop'],
        'genres_all': [['Rock'], ['Pop', 'Electronic'], ['Rock', 'Alternative'], ['Jazz'], ['Pop']],
        'year': [2020, 2019, 2021, 2018, 2020],
        'listens': [1000, 5000, 1500, 500, 3000],
        'favorites': [50, 200, 75, 25, 150],
        'tempo': [120.0, 128.5, 110.2, 95.7, 132.1]
    }


@pytest.fixture
def sample_index_file(temp_dir, sample_index_data):
    """Create a sample index parquet file."""
    index_path = temp_dir / "test_index.parquet"
    pd.DataFrame(sample_index_data).to_parquet(index_path, index=False)
    return index_path


@pytest.fixture
def sample_features_file(temp_dir, sample_features_data):
    """Create a sample features parquet file."""
    features_path = temp_dir / "test_features.parquet"
    pd.DataFrame(sample_features_data).to_parquet(features_path, index=False)
    return features_path


@pytest.fixture
def sample_metadata_file(temp_dir, sample_metadata):
    """Create a sample metadata parquet file."""
    metadata_path = temp_dir / "test_metadata.parquet"
    pd.DataFrame(sample_metadata).to_parquet(metadata_path, index=False)
    return metadata_path
