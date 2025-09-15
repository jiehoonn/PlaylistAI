"""
Utility functions for mapping FMA track IDs to file paths.

The FMA dataset stores audio files using six‑digit zero‑padded
identifiers.  The first three digits form a directory name, and the
full six digits (with `.mp3` extension) form the filename.  For
example:

* 2 -> 000/000002.mp3
* 1234 -> 001/001234.mp3

See FMA documentation for details (https://github.com/mdeff/fma#:~:text=Then%2C%20you%20got%20various%20sizes,encoded%20audio%20data).
"""
from pathlib import Path

def track_id_to_relpath(track_id: int) -> Path:
    """Return the relative path to an FMA MP3 given its numeric track ID."""
    try:
        tid_int = int(track_id)
    except (TypeError, ValueError):
        raise ValueError(f"track_id must be convertible to int, got {track_id!r}")
    if tid_int < 0:
        raise ValueError(f"track_id must be non‑negative, got {tid_int}")

    six = f"{tid_int:06d}"
    folder = six[:3]
    filename = f"{six}.mp3"
    return Path(folder) / filename