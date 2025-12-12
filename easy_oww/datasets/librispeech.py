"""
LibriSpeech dataset downloader for negative speech samples
"""
from pathlib import Path
from typing import List, Optional
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

console = Console()
logger = logging.getLogger(__name__)


class LibriSpeechDataset:
    """Downloads and manages LibriSpeech dataset for negative samples"""

    def __init__(self, datasets_dir: str, cache_dir: str):
        """
        Initialize LibriSpeech dataset

        Args:
            datasets_dir: Directory to store datasets
            cache_dir: Cache directory for downloads
        """
        self.datasets_dir = Path(datasets_dir)
        self.cache_dir = Path(cache_dir)
        self.speech_dir = self.datasets_dir / 'librispeech'
        self.speech_dir.mkdir(parents=True, exist_ok=True)

    def is_cached(self) -> bool:
        """Check if LibriSpeech is already downloaded"""
        # Check if we have at least 1000 audio files
        if not self.speech_dir.exists():
            return False
        wav_files = list(self.speech_dir.glob('**/*.wav'))
        return len(wav_files) >= 1000

    def download_subset(self, config: str = 'clean', split: str = 'validation', max_samples: int = 2000) -> Path:
        """
        Download a subset of LibriSpeech dataset

        Args:
            config: Which config to download ('clean', 'other', or 'all')
            split: Which split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to download

        Returns:
            Path to speech directory
        """
        try:
            from datasets import load_dataset
            import soundfile as sf
            import numpy as np
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            console.print("[red]Error:[/red] Install required packages: pip install datasets soundfile")
            return self.speech_dir

        console.print(f"\n[bold]Downloading LibriSpeech ({config}/{split})...[/bold]")
        console.print(f"  This will download ~{max_samples} speech samples for negative training")

        try:
            # Load dataset from Hugging Face
            with console.status("[bold green]Loading LibriSpeech dataset..."):
                dataset = load_dataset('librispeech_asr', config, split=split, streaming=True)

            # Download and save audio files
            subset_dir = self.speech_dir / f"{config}_{split}"
            subset_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"Downloading speech samples", total=max_samples)

                for i, sample in enumerate(dataset):
                    if count >= max_samples:
                        break

                    try:
                        # Get audio data
                        audio = sample['audio']
                        array = audio['array']
                        sample_rate = audio['sampling_rate']

                        # Resample to 16kHz if needed
                        if sample_rate != 16000:
                            import scipy.signal as signal
                            num_samples = int(len(array) * 16000 / sample_rate)
                            array = signal.resample(array, num_samples)
                            sample_rate = 16000

                        # Save to file
                        output_path = subset_dir / f"speech_{count:05d}.wav"
                        sf.write(output_path, array, sample_rate)

                        count += 1
                        progress.update(task, advance=1)

                    except Exception as e:
                        logger.warning(f"Failed to process sample {i}: {e}")
                        continue

            console.print(f"  [green]✓[/green] Downloaded {count} speech samples to {subset_dir}")
            return self.speech_dir

        except Exception as e:
            logger.error(f"Failed to download LibriSpeech: {e}")
            console.print(f"[red]Error downloading LibriSpeech:[/red] {e}")
            return self.speech_dir

    def download_all(self, max_samples: int = 5000) -> Path:
        """
        Download multiple subsets of LibriSpeech for variety

        Args:
            max_samples: Total maximum samples to download

        Returns:
            Path to speech directory
        """
        if self.is_cached():
            console.print("[green]✓[/green] LibriSpeech already downloaded")
            return self.speech_dir

        console.print("\n[bold cyan]Downloading speech dataset for negative samples[/bold cyan]")
        console.print(f"  Target: {max_samples} speech samples")

        # Download from multiple splits for diversity
        samples_per_split = max_samples // 3

        # Download validation split (clean speech)
        self.download_subset(config='clean', split='validation', max_samples=samples_per_split)

        # Download test split (clean speech)
        self.download_subset(config='clean', split='test', max_samples=samples_per_split)

        # Download train split (clean speech)
        self.download_subset(config='clean', split='train', max_samples=samples_per_split)

        total_files = len(list(self.speech_dir.glob('**/*.wav')))
        console.print(f"\n[green]✓[/green] LibriSpeech ready: {total_files} speech samples available")

        return self.speech_dir

    def get_speech_files(self) -> List[Path]:
        """
        Get all downloaded speech files

        Returns:
            List of paths to speech files
        """
        return list(self.speech_dir.glob('**/*.wav'))
