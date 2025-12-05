"""
Room Impulse Response (RIR) downloader for acoustic simulation
"""
import os
import requests
from pathlib import Path
from typing import List
from tqdm import tqdm


class RIRDownloader:
    """Downloads Room Impulse Response files for acoustic augmentation"""

    # MIT Acoustical Reverberation Scene Statistics Survey
    BASE_URL = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/"

    # Sample of RIR files (in practice, you'd want all 271 files)
    # This is a subset for demonstration
    MANIFEST = [
        "Audio/3-4-04_KEMAR_Omni/kemar_1.wav",
        "Audio/3-4-04_KEMAR_Omni/kemar_2.wav",
        "Audio/3-4-04_KEMAR_Omni/kemar_3.wav",
        # ... (full list would include all 271 files)
    ]

    SIZE_GB = 2

    def __init__(self, dest_dir: str):
        """
        Initialize RIR downloader

        Args:
            dest_dir: Destination directory
        """
        self.dest_dir = Path(dest_dir) / 'mit_rir'
        self.dest_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, relative_path: str) -> Path:
        """
        Download a single RIR file

        Args:
            relative_path: Relative path in the MIT RIR dataset

        Returns:
            Path to downloaded file
        """
        url = f"{self.BASE_URL}{relative_path}"
        filename = Path(relative_path).name
        output_path = self.dest_dir / filename

        if output_path.exists():
            return output_path

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return output_path

    def download_all(self) -> List[Path]:
        """
        Download all RIR files

        Returns:
            List of downloaded file paths
        """
        print(f"Downloading MIT Room Impulse Responses (~{self.SIZE_GB}GB)...")
        downloaded = []

        for file_path in self.MANIFEST:
            try:
                path = self.download_file(file_path)
                downloaded.append(path)
            except Exception as e:
                print(f"Failed to download {file_path}: {e}")

        print(f"✓ Downloaded {len(downloaded)} RIR files to: {self.dest_dir}")
        return downloaded

    def is_cached(self) -> bool:
        """Check if RIR files are already downloaded"""
        if not self.dest_dir.exists():
            return False

        # Check if we have at least some files
        existing_files = list(self.dest_dir.glob('*.wav'))
        return len(existing_files) >= len(self.MANIFEST) * 0.8  # 80% threshold

    def count_cached_files(self) -> int:
        """Count number of cached RIR files"""
        if not self.dest_dir.exists():
            return 0
        return len(list(self.dest_dir.glob('*.wav')))
