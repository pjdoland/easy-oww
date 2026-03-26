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
        if not self.dipco_dir.exists():
            return False
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
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            console.print("[red]Error:[/red] Install required packages: pip install datasets soundfile")
            return self.dipco_dir

        console.print("\n[bold cyan]Downloading Dinner Party Corpus (DiPCo)...[/bold cyan]")
        console.print("  ~5.5 hours of far-field conversational speech for FP rate testing")

        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

            def _write_sample(index, audio_array, sample_rate):
                """Resample if needed and write a single audio sample to disk."""
                if sample_rate != 16000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                output_path = self.dipco_dir / f"dipco_{index:05d}.wav"
                sf.write(output_path, audio_array, 16000, subtype='PCM_16')
                return index

            with console.status("[bold green]Loading DiPCo dataset..."):
                dataset = load_dataset(
                    self.HF_DATASET,
                    "mixed-channel",
                    split="test",
                    streaming=True
                )

            count = 0
            max_workers = 4
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Downloading DiPCo audio", total=None)

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    pending = set()
                    next_index = 0
                    max_pending = max_workers * 2  # Cap queued work to bound memory

                    def _drain_completed():
                        nonlocal count
                        done = {f for f in pending if f.done()}
                        for f in done:
                            pending.discard(f)
                            try:
                                f.result()
                                count += 1
                            except Exception as e:
                                logger.warning(f"Failed to process sample: {e}")

                    for i, sample in enumerate(dataset):
                        while len(pending) >= max_pending:
                            done_set, pending_set = wait(pending, return_when=FIRST_COMPLETED)
                            pending.clear()
                            pending.update(pending_set)
                            for f in done_set:
                                try:
                                    f.result()
                                    count += 1
                                except Exception as e:
                                    logger.warning(f"Failed to process sample: {e}")

                        try:
                            audio = sample['audio']
                            array = np.array(audio['array'], dtype=np.float32)
                            sample_rate = audio['sampling_rate']
                            future = executor.submit(_write_sample, next_index, array, sample_rate)
                            pending.add(future)
                            next_index += 1
                        except Exception as e:
                            logger.warning(f"Failed to read sample {i}: {e}")
                            continue

                        _drain_completed()
                        progress.update(task, description=f"Downloading DiPCo audio ({count} saved)")

                    for f in as_completed(pending):
                        try:
                            f.result()
                            count += 1
                        except Exception as e:
                            logger.warning(f"Failed to process sample: {e}")
                    progress.update(task, description=f"Downloading DiPCo audio ({count} saved)")

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
        for f in self.get_audio_files():
            try:
                info = sf.info(str(f))
                total_seconds += info.duration
            except Exception:
                continue

        return total_seconds / 3600.0
