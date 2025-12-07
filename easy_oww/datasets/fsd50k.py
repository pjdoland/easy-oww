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

    # Limit number of files to download (we don't need all 50k+ files)
    MAX_FILES_PER_SPLIT = 2000

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
        print("Files will be extracted directly to WAV format for use as background noise.")

        try:
            from datasets import load_dataset
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required to download FSD50K. "
                "Install it with: pip install datasets soundfile"
            )

        try:
            # Load dataset from Hugging Face in streaming mode
            # This avoids downloading all 50k+ files upfront
            print(f"\n[1/3] Loading dataset from {self.HF_DATASET}...")
            print(f"Cache directory: {self.cache_dir}")
            print(f"Downloading up to {self.MAX_FILES_PER_SPLIT} files per split (streaming mode)")

            dataset = load_dataset(
                self.HF_DATASET,
                cache_dir=str(self.cache_dir),
                streaming=True  # Stream instead of downloading everything
            )

            # Show dataset structure
            print(f"\nDataset splits: {list(dataset.keys())}")

            # Create directories for dev and eval audio
            dev_dir = self.dest_dir / 'dev'
            eval_dir = self.dest_dir / 'eval'
            dev_dir.mkdir(exist_ok=True, parents=True)
            eval_dir.mkdir(exist_ok=True, parents=True)

            # Process dev split
            print(f"\n[2/3] Extracting dev split audio (max {self.MAX_FILES_PER_SPLIT} files)...")
            if 'train' in dataset:  # FSD50K uses 'train' split for dev
                split_name = 'train'
                target_dir = dev_dir
            elif 'dev' in dataset:
                split_name = 'dev'
                target_dir = dev_dir
            else:
                print("Warning: No train/dev split found")
                split_name = None

            if split_name:
                saved_count = 0
                split_iter = iter(dataset[split_name])

                with tqdm(total=self.MAX_FILES_PER_SPLIT, desc=f"{split_name} files") as pbar:
                    while saved_count < self.MAX_FILES_PER_SPLIT:
                        try:
                            example = next(split_iter)

                            if 'audio' in example:
                                audio_data = example['audio']
                                audio_array = audio_data['array']
                                sample_rate = audio_data['sampling_rate']

                                # Try different field names for filename
                                filename = None
                                for field in ['fname', 'file_name', 'filename', 'id']:
                                    if field in example and example[field]:
                                        filename = str(example[field])
                                        break

                                if not filename:
                                    filename = f'{split_name}_{saved_count:06d}.wav'
                                elif not filename.endswith('.wav'):
                                    filename = f"{filename}.wav"

                                output_path = target_dir / filename
                                sf.write(output_path, audio_array, sample_rate)
                                saved_count += 1
                                pbar.update(1)
                        except StopIteration:
                            print(f"\nReached end of {split_name} split")
                            break
                        except Exception as e:
                            print(f"\nWarning: Failed to process file: {e}")
                            continue

                print(f"✓ Saved {saved_count} dev files")

            # Process eval/test split
            print(f"\n[3/3] Extracting eval split audio (max {self.MAX_FILES_PER_SPLIT} files)...")
            if 'test' in dataset:  # FSD50K uses 'test' split for eval
                split_name = 'test'
                target_dir = eval_dir
            elif 'eval' in dataset:
                split_name = 'eval'
                target_dir = eval_dir
            elif 'validation' in dataset:
                split_name = 'validation'
                target_dir = eval_dir
            else:
                print("Warning: No test/eval/validation split found")
                split_name = None

            if split_name:
                saved_count = 0
                split_iter = iter(dataset[split_name])

                with tqdm(total=self.MAX_FILES_PER_SPLIT, desc=f"{split_name} files") as pbar:
                    while saved_count < self.MAX_FILES_PER_SPLIT:
                        try:
                            example = next(split_iter)

                            if 'audio' in example:
                                audio_data = example['audio']
                                audio_array = audio_data['array']
                                sample_rate = audio_data['sampling_rate']

                                # Try different field names for filename
                                filename = None
                                for field in ['fname', 'file_name', 'filename', 'id']:
                                    if field in example and example[field]:
                                        filename = str(example[field])
                                        break

                                if not filename:
                                    filename = f'{split_name}_{saved_count:06d}.wav'
                                elif not filename.endswith('.wav'):
                                    filename = f"{filename}.wav"

                                output_path = target_dir / filename
                                sf.write(output_path, audio_array, sample_rate)
                                saved_count += 1
                                pbar.update(1)
                        except StopIteration:
                            print(f"\nReached end of {split_name} split")
                            break
                        except Exception as e:
                            print(f"\nWarning: Failed to process file: {e}")
                            continue

                print(f"✓ Saved {saved_count} eval files")

            # Count total files
            total_files = self.count_cached_files()
            print(f"\n✓ FSD50K download complete!")
            print(f"  Total files: {total_files}")
            print(f"  Location: {self.dest_dir}")
            return self.dest_dir

        except Exception as e:
            import traceback
            print(f"\n✗ Failed to download FSD50K dataset: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            raise RuntimeError(f"FSD50K download failed: {e}")

    def is_cached(self) -> bool:
        """Check if FSD50K is already downloaded"""
        if not self.dest_dir.exists():
            return False

        # Check if we have actual audio files (not just empty directories)
        file_count = self.count_cached_files()
        return file_count > 0

    def count_cached_files(self) -> int:
        """Count number of cached audio files"""
        if not self.dest_dir.exists():
            return 0

        count = 0
        # Check new structure
        for audio_dir in [self.dest_dir / 'dev', self.dest_dir / 'eval']:
            if audio_dir.exists():
                count += len(list(audio_dir.glob('*.wav')))

        # Also check old structure for backwards compatibility
        if count == 0:
            for audio_dir in [self.dest_dir / 'FSD50K.dev_audio', self.dest_dir / 'FSD50K.eval_audio']:
                if audio_dir.exists():
                    count += len(list(audio_dir.glob('*.wav')))

        return count
