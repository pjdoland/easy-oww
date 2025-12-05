"""
FSD50K sound events dataset downloader
"""
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


class FSD50kDownloader:
    """Downloads FSD50K dataset for background sound augmentation"""

    ZENODO_RECORD = "4060432"
    FILES = [
        ("FSD50K.dev_audio.zip", "https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip"),
        ("FSD50K.eval_audio.zip", "https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip"),
    ]

    SIZE_GB = 30

    def __init__(self, dest_dir: str):
        """
        Initialize FSD50K downloader

        Args:
            dest_dir: Destination directory
        """
        self.dest_dir = Path(dest_dir) / 'fsd50k'
        self.dest_dir.mkdir(parents=True, exist_ok=True)

    def download_and_extract(self, filename: str, url: str) -> Path:
        """
        Download and extract a zip file

        Args:
            filename: Name of the file
            url: Download URL

        Returns:
            Path to extracted directory
        """
        zip_path = self.dest_dir / filename

        # Download if not exists
        if not zip_path.exists():
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        # Extract
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dest_dir)

        # Cleanup zip file
        zip_path.unlink()

        return self.dest_dir

    def download_all(self) -> Path:
        """
        Download and extract all FSD50K files

        Returns:
            Path to FSD50K directory
        """
        print(f"Downloading FSD50K dataset (~{self.SIZE_GB}GB)...")

        for filename, url in self.FILES:
            try:
                self.download_and_extract(filename, url)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")

        print(f"✓ FSD50K downloaded to: {self.dest_dir}")
        return self.dest_dir

    def is_cached(self) -> bool:
        """Check if FSD50K is already downloaded"""
        if not self.dest_dir.exists():
            return False

        # Check for extracted directories
        dev_dir = self.dest_dir / 'FSD50K.dev_audio'
        eval_dir = self.dest_dir / 'FSD50K.eval_audio'

        return dev_dir.exists() and eval_dir.exists()

    def count_cached_files(self) -> int:
        """Count number of cached audio files"""
        if not self.is_cached():
            return 0

        count = 0
        for audio_dir in [self.dest_dir / 'FSD50K.dev_audio', self.dest_dir / 'FSD50K.eval_audio']:
            if audio_dir.exists():
                count += len(list(audio_dir.glob('*.wav')))

        return count
