"""
FSD50K sound events dataset downloader
"""
import os
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm


class FSD50kDownloader:
    """Downloads FSD50K dataset for background sound augmentation using Hugging Face"""

    # Using Hugging Face dataset instead of direct Zenodo download
    # The Zenodo zip files are very large and prone to corruption
    HF_DATASET = "Fhrozen/FSD50k"

    SIZE_GB = 30

    def __init__(self, dest_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize FSD50K downloader

        Args:
            dest_dir: Destination directory
            cache_dir: Cache directory for Hugging Face datasets (optional)
        """
        self.dest_dir = Path(dest_dir) / 'fsd50k'
        self.dest_dir.mkdir(parents=True, exist_ok=True)

        # Use cache_dir if provided, otherwise use dest_dir parent's cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(dest_dir).parent / '.cache' / 'huggingface'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self) -> Path:
        """
        Download FSD50K dataset using Hugging Face datasets library

        Returns:
            Path to FSD50K directory

        Raises:
            ImportError: If datasets library is not installed
            RuntimeError: If download fails
        """
        print(f"Downloading FSD50K dataset from Hugging Face (~{self.SIZE_GB}GB)...")
        print("Note: This may take a while. The dataset will be cached by Hugging Face.")

        try:
            from datasets import load_dataset
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required to download FSD50K. "
                "Install it with: pip install datasets soundfile"
            )

        try:
            # Load dataset from Hugging Face
            # Cache to the specified directory instead of ~/.cache/huggingface
            print(f"Loading dataset from {self.HF_DATASET}...")
            print(f"Cache directory: {self.cache_dir}")
            dataset = load_dataset(self.HF_DATASET, cache_dir=str(self.cache_dir))

            # Create directories for dev and eval audio
            dev_dir = self.dest_dir / 'FSD50K.dev_audio'
            eval_dir = self.dest_dir / 'FSD50K.eval_audio'
            dev_dir.mkdir(exist_ok=True)
            eval_dir.mkdir(exist_ok=True)

            # Process dev split
            if 'dev' in dataset:
                print(f"Saving dev audio files to {dev_dir}...")
                for idx, example in enumerate(tqdm(dataset['dev'], desc="Dev files")):
                    if 'audio' in example:
                        audio_data = example['audio']
                        audio_array = audio_data['array']
                        sample_rate = audio_data['sampling_rate']

                        # Get original filename or create one
                        filename = example.get('file_name', f'dev_{idx:05d}.wav')
                        if not filename.endswith('.wav'):
                            filename = f"{filename}.wav"

                        output_path = dev_dir / filename
                        sf.write(output_path, audio_array, sample_rate)

            # Process eval split
            if 'eval' in dataset:
                print(f"Saving eval audio files to {eval_dir}...")
                for idx, example in enumerate(tqdm(dataset['eval'], desc="Eval files")):
                    if 'audio' in example:
                        audio_data = example['audio']
                        audio_array = audio_data['array']
                        sample_rate = audio_data['sampling_rate']

                        # Get original filename or create one
                        filename = example.get('file_name', f'eval_{idx:05d}.wav')
                        if not filename.endswith('.wav'):
                            filename = f"{filename}.wav"

                        output_path = eval_dir / filename
                        sf.write(output_path, audio_array, sample_rate)

            print(f"✓ FSD50K downloaded to: {self.dest_dir}")
            return self.dest_dir

        except Exception as e:
            import traceback
            print(f"Failed to download FSD50K dataset: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            raise RuntimeError(f"FSD50K download failed: {e}")

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
