"""
Dinner Party Corpus (DiPCo) dataset downloader for false positive rate testing
"""
from pathlib import Path
from typing import List
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()
logger = logging.getLogger(__name__)


class DiPCoDataset:
    """Downloads and manages DiPCo dataset for false positive rate evaluation"""

    HF_DATASET = "benjamin-paine/dinner-party-corpus"
    HF_CONFIG = "mixed-channel"
    SIZE_GB = 1.5  # Approximate for far-field audio

    def __init__(self, datasets_dir: str, cache_dir: str):
        """
        Initialize DiPCo dataset

        Args:
            datasets_dir: Directory to store datasets
            cache_dir: Cache directory for downloads
        """
        self.datasets_dir = Path(datasets_dir)
        self.cache_dir = Path(cache_dir)
        self.dipco_dir = self.datasets_dir / 'dipco'
        self.dipco_dir.mkdir(parents=True, exist_ok=True)

    def is_cached(self) -> bool:
        """Check if DiPCo is already downloaded"""
        count = 0
        for _ in self.dipco_dir.glob('**/*.wav'):
            count += 1
            if count >= 10:
                return True
        return False

    def download_all(self) -> Path:
        """
        Download DiPCo far-field audio for FP rate testing

        Returns:
            Path to DiPCo directory
        """
        if self.is_cached():
            console.print("[green]✓[/green] DiPCo already downloaded")
            return self.dipco_dir

        try:
            from datasets import load_dataset
            import soundfile as sf
            import numpy as np
            import librosa
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            console.print("[red]Error:[/red] Install required packages: pip install datasets soundfile librosa")
            return self.dipco_dir

        console.print("\n[bold cyan]Downloading Dinner Party Corpus (DiPCo)...[/bold cyan]")
        console.print("  ~5.5 hours of far-field conversational speech for FP rate testing")

        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

            def _write_sample(index, audio_array, sample_rate):
                if sample_rate != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                output_path = self.dipco_dir / f"dipco_{index:05d}.wav"
                sf.write(output_path, audio_array, 16000, subtype='PCM_16')

            with console.status("[bold green]Downloading DiPCo dataset..."):
                dataset = load_dataset(
                    self.HF_DATASET,
                    self.HF_CONFIG,
                    split="test",
                )

            console.print(f"  Downloaded {len(dataset)} samples, writing WAV files...")

            count = 0
            next_index = 0
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console
            ) as progress:
                task = progress.add_task("Writing DiPCo audio", total=len(dataset))

                def _drain(futures, pending):
                    nonlocal count
                    for f in futures:
                        try:
                            f.result()
                            count += 1
                        except Exception as e:
                            logger.warning(f"Failed to write sample {pending[f]}: {e}")
                        progress.update(task, advance=1)
                        del pending[f]

                max_workers = 4
                max_pending = max_workers * 2
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    pending = {}
                    for sample in dataset:
                        while len(pending) >= max_pending:
                            done, _ = wait(pending, return_when=FIRST_COMPLETED)
                            _drain(done, pending)

                        try:
                            audio = sample['audio']
                            array = np.asarray(audio['array'], dtype=np.float32)
                            sample_rate = audio['sampling_rate']
                            pending[executor.submit(_write_sample, next_index, array, sample_rate)] = next_index
                            next_index += 1
                        except Exception as e:
                            logger.warning(f"Failed to read sample: {e}")
                            progress.update(task, advance=1)

                    _drain(as_completed(pending), pending)

            total_duration = self.get_total_duration_hours()
            console.print(f"\n[green]✓[/green] DiPCo ready: {count} files, {total_duration:.1f} hours of audio")
            return self.dipco_dir

        except Exception as e:
            logger.error(f"Failed to download DiPCo: {e}")
            console.print(f"[red]Error downloading DiPCo:[/red] {e}")
            return self.dipco_dir

    def get_audio_files(self) -> List[Path]:
        """Get all downloaded audio files sorted by name"""
        return sorted(self.dipco_dir.glob('**/*.wav'))

    def get_total_duration_hours(self) -> float:
        """Calculate total duration of all audio files in hours"""
        try:
            import soundfile as sf
        except ImportError:
            return 0.0

        total_seconds = 0.0
        for f in self.dipco_dir.glob('**/*.wav'):
            try:
                info = sf.info(str(f))
                total_seconds += info.duration
            except Exception:
                continue

        return total_seconds / 3600.0
