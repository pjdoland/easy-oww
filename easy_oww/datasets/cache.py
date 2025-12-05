"""
Cache management for downloaded datasets
"""
import json
import os
import hashlib
from pathlib import Path
from typing import Optional, Dict


class CacheManager:
    """Manages dataset caching and integrity verification"""

    def __init__(self, cache_dir: str):
        """
        Initialize cache manager

        Args:
            cache_dir: Directory to store cache metadata
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.cache_dir / 'manifest.json'
        self.manifest = self.load_manifest()

    def load_manifest(self) -> Dict:
        """
        Load cache manifest from disk

        Returns:
            Dictionary of cached datasets
        """
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {}

    def save_manifest(self):
        """Save cache manifest to disk"""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def is_cached(self, dataset_name: str) -> bool:
        """
        Check if dataset is cached and valid

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if dataset is cached and valid
        """
        if dataset_name not in self.manifest:
            return False

        entry = self.manifest[dataset_name]
        filepath = entry['path']

        # Check file exists
        if not os.path.exists(filepath):
            return False

        # Check file size matches (quick check)
        if os.path.getsize(filepath) != entry['size']:
            return False

        return True

    def add_entry(
        self,
        dataset_name: str,
        filepath: str,
        checksum: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add dataset to cache manifest

        Args:
            dataset_name: Name of the dataset
            filepath: Path to downloaded file
            checksum: Optional checksum for verification
            metadata: Optional additional metadata
        """
        import time

        self.manifest[dataset_name] = {
            'path': filepath,
            'size': os.path.getsize(filepath),
            'checksum': checksum,
            'downloaded_at': time.time(),
            'metadata': metadata or {}
        }
        self.save_manifest()

    def remove_entry(self, dataset_name: str):
        """
        Remove dataset from cache manifest

        Args:
            dataset_name: Name of the dataset
        """
        if dataset_name in self.manifest:
            del self.manifest[dataset_name]
            self.save_manifest()

    def verify_checksum(self, filepath: str, expected_checksum: str, algorithm: str = 'sha256') -> bool:
        """
        Verify file checksum

        Args:
            filepath: Path to file
            expected_checksum: Expected checksum value
            algorithm: Hash algorithm (default: sha256)

        Returns:
            True if checksum matches
        """
        hash_func = hashlib.new(algorithm)

        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)

        actual_checksum = hash_func.hexdigest()
        return actual_checksum == expected_checksum

    def get_entry(self, dataset_name: str) -> Optional[Dict]:
        """
        Get cache entry for dataset

        Args:
            dataset_name: Name of the dataset

        Returns:
            Cache entry dictionary or None
        """
        return self.manifest.get(dataset_name)

    def list_cached(self) -> Dict:
        """
        List all cached datasets

        Returns:
            Dictionary of cached datasets
        """
        return self.manifest.copy()
