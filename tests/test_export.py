"""
Tests for playlist export functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.service.export import write_m3u, write_csv, export_playlist


class TestM3UExport:
    """Test M3U playlist export functionality."""
    
    def test_write_m3u_basic(self, temp_dir, sample_index_file):
        """Test basic M3U file creation."""
        track_ids = [1, 2, 3]
        output_path = temp_dir / "test.m3u"
        
        write_m3u(track_ids, output_path, index_path=sample_index_file)
        
        # Check file was created
        assert output_path.exists()
        
        # Check file content
        content = output_path.read_text()
        lines = content.strip().split('\n')
        
        # Should start with M3U header
        assert lines[0] == "#EXTM3U"
        
        # Should have EXTINF lines for each track
        extinf_lines = [line for line in lines if line.startswith("#EXTINF:")]
        assert len(extinf_lines) == len(track_ids)
        
        # Should have file path lines
        path_lines = [line for line in lines if not line.startswith("#")]
        assert len(path_lines) == len(track_ids)
    
    def test_write_m3u_with_title(self, temp_dir, sample_index_file):
        """Test M3U export with custom playlist title."""
        track_ids = [1, 2]
        output_path = temp_dir / "titled.m3u"
        title = "My Test Playlist"
        
        write_m3u(track_ids, output_path, title=title, index_path=sample_index_file)
        
        content = output_path.read_text()
        assert f"#PLAYLIST:{title}" in content
    
    def test_write_m3u_missing_tracks(self, temp_dir, sample_index_file):
        """Test M3U export with some missing track IDs."""
        track_ids = [1, 999, 2]  # 999 doesn't exist
        output_path = temp_dir / "missing.m3u"
        
        # Should not raise exception, just skip missing tracks
        write_m3u(track_ids, output_path, index_path=sample_index_file)
        
        assert output_path.exists()
        content = output_path.read_text()
        
        # Should only have 2 tracks (1 and 2), skipping 999
        extinf_lines = [line for line in content.split('\n') if line.startswith("#EXTINF:")]
        assert len(extinf_lines) == 2
    
    def test_write_m3u_relative_paths(self, temp_dir, sample_index_file):
        """Test M3U export with relative paths."""
        track_ids = [1, 2]
        output_path = temp_dir / "relative.m3u"
        
        write_m3u(track_ids, output_path, index_path=sample_index_file, use_relative_paths=True)
        
        content = output_path.read_text()
        # Should contain relative paths
        assert "000/000001.mp3" in content
        assert "000/000002.mp3" in content


class TestCSVExport:
    """Test CSV playlist export functionality."""
    
    def test_write_csv_basic(self, temp_dir, sample_index_file):
        """Test basic CSV file creation."""
        track_ids = [1, 2, 3]
        output_path = temp_dir / "test.csv"
        
        write_csv(track_ids, output_path, index_path=sample_index_file)
        
        # Check file was created and is valid CSV
        assert output_path.exists()
        df = pd.read_csv(output_path)
        
        # Check structure
        assert len(df) == len(track_ids)
        assert 'track_id' in df.columns
        assert 'artist_name' in df.columns
        assert 'track_title' in df.columns
        
        # Check track IDs match
        assert set(df['track_id']) == set(track_ids)
    
    def test_write_csv_track_order(self, temp_dir, sample_index_file):
        """Test that CSV preserves track order."""
        track_ids = [3, 1, 2]  # Non-sequential order
        output_path = temp_dir / "ordered.csv"
        
        write_csv(track_ids, output_path, index_path=sample_index_file)
        
        df = pd.read_csv(output_path)
        # Should preserve input order
        assert df['track_id'].tolist() == track_ids
    
    def test_write_csv_no_paths(self, temp_dir, sample_index_file):
        """Test CSV export without file paths."""
        track_ids = [1, 2]
        output_path = temp_dir / "no_paths.csv"
        
        write_csv(track_ids, output_path, index_path=sample_index_file, include_paths=False)
        
        df = pd.read_csv(output_path)
        # Should not include path columns
        assert 'audio_path' not in df.columns
        assert 'relpath' not in df.columns


class TestExportPlaylist:
    """Test the unified export_playlist function."""
    
    def test_export_multiple_formats(self, temp_dir, sample_index_file):
        """Test exporting to multiple formats at once."""
        track_ids = [1, 2, 3]
        seed_id = 1
        
        # Mock the index path temporarily
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        # Copy test data to temporary location  
        df = pd.read_parquet(sample_index_file)
        df.to_parquet(tmp_path, index=False)
        
        try:
            # Patch the default index path
            from src.service import export
            original_path = export.INDEX_PATH
            export.INDEX_PATH = tmp_path
            
            files = export_playlist(
                track_ids, seed_id, 
                output_dir=temp_dir, 
                formats=["m3u", "csv"]
            )
            
            # Should create 2 files
            assert len(files) == 2
            assert all(f.exists() for f in files)
            
            # Check file extensions
            extensions = {f.suffix for f in files}
            assert extensions == {'.m3u', '.csv'}
            
        finally:
            # Restore original path
            export.INDEX_PATH = original_path
            tmp_path.unlink()
    
    def test_export_custom_name(self, temp_dir, sample_index_file):
        """Test export with custom playlist name."""
        track_ids = [1, 2]
        seed_id = 1
        custom_name = "My Custom Playlist"
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        df = pd.read_parquet(sample_index_file)
        df.to_parquet(tmp_path, index=False)
        
        try:
            from src.service import export
            original_path = export.INDEX_PATH
            export.INDEX_PATH = tmp_path
            
            files = export_playlist(
                track_ids, seed_id,
                output_dir=temp_dir,
                formats=["m3u"],
                playlist_name=custom_name
            )
            
            # Should use custom name (sanitized)
            assert len(files) == 1
            assert "My_Custom_Playlist" in files[0].name
            
        finally:
            export.INDEX_PATH = original_path
            tmp_path.unlink()


class TestExportErrorHandling:
    """Test error handling in export functions."""
    
    def test_invalid_index_file(self, temp_dir):
        """Test handling of invalid index file."""
        track_ids = [1, 2, 3]
        output_path = temp_dir / "test.m3u"
        fake_index = temp_dir / "fake_index.parquet"
        
        # Should raise exception for non-existent file
        with pytest.raises(Exception):
            write_m3u(track_ids, output_path, index_path=fake_index)
    
    def test_empty_track_list(self, temp_dir, sample_index_file):
        """Test export with empty track list."""
        track_ids = []
        output_path = temp_dir / "empty.m3u"
        
        # Should raise ValueError for empty playlist
        with pytest.raises(ValueError, match="No valid tracks found"):
            write_m3u(track_ids, output_path, index_path=sample_index_file)
    
    def test_all_invalid_tracks(self, temp_dir, sample_index_file):
        """Test export when all track IDs are invalid."""
        track_ids = [999, 998, 997]  # None of these exist
        output_path = temp_dir / "invalid.m3u"
        
        # Should raise ValueError when no valid tracks found
        with pytest.raises(ValueError, match="No valid tracks found"):
            write_m3u(track_ids, output_path, index_path=sample_index_file)
    
    def test_directory_creation(self, temp_dir, sample_index_file):
        """Test that output directories are created automatically."""
        track_ids = [1, 2]
        nested_path = temp_dir / "nested" / "dir" / "test.m3u"
        
        # Directory doesn't exist yet
        assert not nested_path.parent.exists()
        
        write_m3u(track_ids, nested_path, index_path=sample_index_file)
        
        # Should create directory and file
        assert nested_path.exists()
        assert nested_path.parent.exists()
