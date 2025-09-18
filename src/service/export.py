"""
Playlist export utilities for converting track recommendations to standard formats.

Supports:
- M3U/M3U8 playlist format for media players
- CSV format for data analysis
- Relative and absolute path handling
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd
from src.utils.logging import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "data/processed/fma_small_index.parquet"


def write_m3u(
    track_ids: List[int],
    output_path: Path,
    title: Optional[str] = None,
    index_path: Path = INDEX_PATH,
    use_relative_paths: bool = False,
) -> None:
    """
    Export track IDs to M3U playlist format.
    
    Args:
        track_ids: List of track IDs to include in playlist
        output_path: Path where M3U file will be written
        title: Optional playlist title
        index_path: Path to track index parquet file
        use_relative_paths: If True, use relative paths instead of absolute
    """
    # Load track metadata
    try:
        index_df = pd.read_parquet(index_path)
        index_df["track_id"] = index_df["track_id"].astype(int)
    except Exception as e:
        log.error(f"Failed to load track index from {index_path}: {e}")
        raise
    
    # Filter to requested tracks and preserve order
    track_meta = []
    for track_id in track_ids:
        track_row = index_df[index_df["track_id"] == track_id]
        if track_row.empty:
            log.warning(f"Track ID {track_id} not found in index, skipping")
            continue
        track_meta.append(track_row.iloc[0])
    
    if not track_meta:
        raise ValueError("No valid tracks found for playlist export")
    
    # Write M3U file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # M3U header
        f.write("#EXTM3U\n")
        if title:
            f.write(f"#PLAYLIST:{title}\n")
        
        # Track entries
        for track in track_meta:
            # Extended info line
            artist = track.get("artist_name", "Unknown Artist")
            title_track = track.get("track_title", "Unknown Title")
            duration = -1  # M3U format: -1 means unknown duration
            
            f.write(f"#EXTINF:{duration},{artist} - {title_track}\n")
            
            # File path
            if use_relative_paths and "relpath" in track:
                file_path = track["relpath"]
            else:
                file_path = track.get("audio_path", "")
            
            f.write(f"{file_path}\n")
    
    log.info(f"Exported {len(track_meta)} tracks to M3U playlist: {output_path}")


def write_csv(
    track_ids: List[int],
    output_path: Path,
    index_path: Path = INDEX_PATH,
    include_paths: bool = True,
) -> None:
    """
    Export track IDs to CSV format with metadata.
    
    Args:
        track_ids: List of track IDs to include
        output_path: Path where CSV file will be written
        index_path: Path to track index parquet file
        include_paths: Whether to include file paths in export
    """
    # Load track metadata
    try:
        index_df = pd.read_parquet(index_path)
        index_df["track_id"] = index_df["track_id"].astype(int)
    except Exception as e:
        log.error(f"Failed to load track index from {index_path}: {e}")
        raise
    
    # Filter to requested tracks and preserve order
    playlist_df = index_df[index_df["track_id"].isin(track_ids)].copy()
    
    if playlist_df.empty:
        raise ValueError("No valid tracks found for playlist export")
    
    # Reorder to match input order
    track_order = {tid: i for i, tid in enumerate(track_ids)}
    playlist_df["playlist_order"] = playlist_df["track_id"].map(track_order)
    playlist_df = playlist_df.sort_values("playlist_order").drop(columns=["playlist_order"])
    
    # Select columns for export
    export_cols = ["track_id", "artist_name", "track_title", "album_title"]
    if "track_genre_top" in playlist_df.columns:
        export_cols.append("track_genre_top")
    
    if include_paths:
        if "audio_path" in playlist_df.columns:
            export_cols.append("audio_path")
        if "relpath" in playlist_df.columns:
            export_cols.append("relpath")
    
    # Export available columns only
    available_cols = [col for col in export_cols if col in playlist_df.columns]
    export_df = playlist_df[available_cols]
    
    # Write CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_df.to_csv(output_path, index=False)
    
    log.info(f"Exported {len(export_df)} tracks to CSV: {output_path}")


def export_playlist(
    track_ids: List[int],
    seed_track_id: int,
    output_dir: Path = ROOT / "playlists",
    formats: List[str] = ["m3u", "csv"],
    playlist_name: Optional[str] = None,
) -> List[Path]:
    """
    Export playlist in multiple formats with automatic naming.
    
    Args:
        track_ids: List of track IDs in the playlist
        seed_track_id: The original seed track ID
        output_dir: Directory to write playlist files
        formats: List of formats to export ("m3u", "csv")
        playlist_name: Custom playlist name (auto-generated if None)
    
    Returns:
        List of paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate playlist name if not provided
    if playlist_name is None:
        playlist_name = f"playlist_seed_{seed_track_id}_{len(track_ids)}_tracks"
    
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in playlist_name)
    
    created_files = []
    
    for fmt in formats:
        if fmt.lower() == "m3u":
            output_path = output_dir / f"{safe_name}.m3u"
            write_m3u(track_ids, output_path, title=playlist_name)
            created_files.append(output_path)
            
        elif fmt.lower() == "csv":
            output_path = output_dir / f"{safe_name}.csv"
            write_csv(track_ids, output_path)
            created_files.append(output_path)
            
        else:
            log.warning(f"Unknown export format: {fmt}")
    
    return created_files


if __name__ == "__main__":
    # Example usage
    example_track_ids = [2, 5, 10, 15, 20]  # Replace with actual track IDs
    seed_id = 2
    
    try:
        files = export_playlist(example_track_ids, seed_id, formats=["m3u", "csv"])
        print(f"Created playlist files: {files}")
    except Exception as e:
        print(f"Export failed: {e}")
