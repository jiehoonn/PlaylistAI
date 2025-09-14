from pathlib import Path
import os
from dotenv import load_dotenv

class Settings:
    def __init__(self) -> None:
        # Load environment variables from .env file
        load_dotenv()
        
        self.FMA_SMALL_DIR = Path(os.getenv("FMA_SMALL_DIR", "")).expanduser().resolve()
        self.FMA_METADATA_DIR = Path(os.getenv("FMA_METADATA_DIR", "")).expanduser().resolve()
        if not self.FMA_SMALL_DIR.is_dir():
            raise FileNotFoundError(f"FMA_SMALL_DIR not found: {self.FMA_SMALL_DIR}")
        if not self.FMA_METADATA_DIR.is_dir():
            raise FileNotFoundError(f"FMA_METADATA_DIR not found: {self.FMA_METADATA_DIR}")

settings = Settings()