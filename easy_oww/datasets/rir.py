"""
Room Impulse Response (RIR) downloader for acoustic simulation
"""
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


class RIRDownloader:
    """Downloads Room Impulse Response files for acoustic augmentation"""

    # Using Hugging Face dataset instead of direct MIT download
    # The MIT website URLs are no longer accessible
    HF_DATASET = "davidscripka/MIT_environmental_impulse_responses"

    # Estimated size for 270 files at 16kHz
    SIZE_GB = 0.01  # Dataset is only ~8MB, not 2GB as initially estimated

    def __init__(self, dest_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize RIR downloader

        Args:
            dest_dir: Destination directory
            cache_dir: Cache directory for Hugging Face datasets (optional)
        """
        self.dest_dir = Path(dest_dir) / 'mit_rir'
        self.dest_dir.mkdir(parents=True, exist_ok=True)

        # Use cache_dir if provided, otherwise use dest_dir parent's cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(dest_dir).parent / '.cache' / 'huggingface'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self) -> List[Path]:
        """
        Download all RIR files from Hugging Face

        Returns:
            List of downloaded file paths
        """
        print(f"Downloading MIT Room Impulse Responses from Hugging Face (~{self.SIZE_GB}GB)...")
        downloaded = []

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required to download RIR files. "
                "Install it with: pip install datasets"
            )

        try:
            # Load dataset from Hugging Face
            # Cache to the specified directory instead of ~/.cache/huggingface
            print(f"Loading dataset from {self.HF_DATASET}...")
            print(f"Cache directory: {self.cache_dir}")
            dataset = load_dataset(self.HF_DATASET, split="train", cache_dir=str(self.cache_dir))

            print(f"Dataset loaded with {len(dataset)} examples")

            # Save each audio file
            for idx, example in enumerate(tqdm(dataset, desc="Downloading RIR files")):
                try:
                    audio_data = example["audio"]

                    # Audio data contains 'array' and 'sampling_rate'
                    audio_array = audio_data["array"]
                    sample_rate = audio_data["sampling_rate"]

                    # Save as WAV file
                    output_path = self.dest_dir / f"rir_{idx:03d}.wav"

                    # Convert to numpy array and save
                    sf.write(output_path, audio_array, sample_rate)
                    downloaded.append(output_path)
                except Exception as e:
                    print(f"Warning: Failed to process example {idx}: {e}")
                    continue

            print(f"✓ Downloaded {len(downloaded)} RIR files to: {self.dest_dir}")
            return downloaded

        except Exception as e:
            import traceback
            print(f"Failed to download RIR dataset: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            raise

    def is_cached(self) -> bool:
        """Check if RIR files are already downloaded"""
        if not self.dest_dir.exists():
            return False

        # Check if we have at least 200 files (minimum for good coverage)
        existing_files = list(self.dest_dir.glob('*.wav'))
        return len(existing_files) >= 200

    def count_cached_files(self) -> int:
        """Count number of cached RIR files"""
        if not self.dest_dir.exists():
            return 0
        return len(list(self.dest_dir.glob('*.wav')))
