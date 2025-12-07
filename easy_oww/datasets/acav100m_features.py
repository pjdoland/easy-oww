"""
ACAV100M pre-computed features downloader
"""
import os
import requests
from pathlib import Path
from typing import Optional, Callable
from tqdm import tqdm


class ACAV100MDownloader:
    """Downloads ACAV100M pre-computed features"""

    # URLs for ACAV100M features on Hugging Face
    TRAIN_URL = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    VAL_URL = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"

    # Actual file sizes from HuggingFace
    TRAIN_SIZE_GB = 17.3
    VAL_SIZE_GB = 0.185  # 185 MB

    # Expected file sizes in bytes for validation
    TRAIN_SIZE_BYTES = 17300000000  # ~17.3 GB
    VAL_SIZE_BYTES = 185000000  # ~185 MB

    # Allow 1% tolerance for file size validation
    SIZE_TOLERANCE = 0.01

    def __init__(self, dest_dir: str):
        """
        Initialize downloader

        Args:
            dest_dir: Destination directory for downloads
        """
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)

    def download_file(
        self,
        url: str,
        output_path: Path,
        resume: bool = True,
        progress_callback: Optional[Callable] = None
    ):
        """
        Download file with resume capability

        Args:
            url: URL to download
            output_path: Output file path
            resume: Enable resume capability
            progress_callback: Optional callback for progress updates
        """
        # Check if partial download exists
        resume_byte_pos = 0
        if resume and output_path.exists():
            resume_byte_pos = output_path.stat().st_size
            mode = 'ab'
        else:
            mode = 'wb'

        # Set up headers for resume
        headers = {}
        if resume_byte_pos > 0:
            headers['Range'] = f'bytes={resume_byte_pos}-'

        # Make request
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        # Get total size
        if 'content-range' in response.headers:
            # Resume response: content-range: bytes start-end/total
            total_size = int(response.headers['content-range'].split('/')[-1])
        elif 'content-length' in response.headers:
            total_size = int(response.headers['content-length']) + resume_byte_pos
        else:
            total_size = None

        # Download with progress bar
        with open(output_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=resume_byte_pos,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)
                        if progress_callback:
                            progress_callback(size)

    def download_training_features(self, resume: bool = True) -> Path:
        """
        Download training features

        Args:
            resume: Enable resume capability

        Returns:
            Path to downloaded file
        """
        output_path = self.dest_dir / 'acav100m_train.npy'

        if output_path.exists() and not resume:
            raise FileExistsError(f"File already exists: {output_path}")

        print(f"Downloading ACAV100M training features (~{self.TRAIN_SIZE_GB}GB)...")
        print(f"URL: {self.TRAIN_URL}")

        self.download_file(self.TRAIN_URL, output_path, resume=resume)

        print(f"✓ Downloaded to: {output_path}")
        return output_path

    def download_validation_features(self, resume: bool = True) -> Path:
        """
        Download validation features

        Args:
            resume: Enable resume capability

        Returns:
            Path to downloaded file
        """
        output_path = self.dest_dir / 'acav100m_val.npy'

        if output_path.exists() and not resume:
            raise FileExistsError(f"File already exists: {output_path}")

        print(f"Downloading ACAV100M validation features (~{self.VAL_SIZE_GB}GB)...")
        print(f"URL: {self.VAL_URL}")

        self.download_file(self.VAL_URL, output_path, resume=resume)

        print(f"✓ Downloaded to: {output_path}")
        return output_path

    def download_all(self, resume: bool = True) -> tuple:
        """
        Download both training and validation features

        Args:
            resume: Enable resume capability

        Returns:
            Tuple of (train_path, val_path)
        """
        train_path = self.download_training_features(resume=resume)
        val_path = self.download_validation_features(resume=resume)
        return train_path, val_path

    def _validate_file_size(self, file_path: Path, expected_size: int) -> bool:
        """
        Validate downloaded file size

        Args:
            file_path: Path to file
            expected_size: Expected size in bytes

        Returns:
            True if file size is within tolerance
        """
        if not file_path.exists():
            return False

        actual_size = file_path.stat().st_size
        min_size = expected_size * (1 - self.SIZE_TOLERANCE)
        max_size = expected_size * (1 + self.SIZE_TOLERANCE)

        return min_size <= actual_size <= max_size

    def is_training_cached(self) -> bool:
        """Check if training features are already downloaded and complete"""
        file_path = self.dest_dir / 'acav100m_train.npy'
        return self._validate_file_size(file_path, self.TRAIN_SIZE_BYTES)

    def is_validation_cached(self) -> bool:
        """Check if validation features are already downloaded and complete"""
        file_path = self.dest_dir / 'acav100m_val.npy'
        return self._validate_file_size(file_path, self.VAL_SIZE_BYTES)

    def are_all_cached(self) -> bool:
        """Check if both training and validation are downloaded and complete"""
        return self.is_training_cached() and self.is_validation_cached()

    def get_training_path(self) -> Path:
        """Get path to training features file"""
        return self.dest_dir / 'acav100m_train.npy'

    def get_validation_path(self) -> Path:
        """Get path to validation features file"""
        return self.dest_dir / 'acav100m_val.npy'
