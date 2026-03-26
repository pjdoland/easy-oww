"""
False positive rate evaluation using continuous audio streams
"""
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table

from easy_oww.testing.detector import ModelDetector

console = Console()


@dataclass
class FalsePositiveRateResult:
    """Results from a false positive rate evaluation"""
    total_duration_hours: float = 0.0
    total_false_positives: int = 0
    false_positives_per_hour: float = 0.0
    threshold: float = 0.5
    detections: List[Dict] = field(default_factory=list)
    per_file_results: List[Dict] = field(default_factory=list)

    def display(self):
        """Display results summary"""
        # Header
        fp_color = "green" if self.false_positives_per_hour < 0.5 else "yellow" if self.false_positives_per_hour < 1.0 else "red"

        console.print(Panel.fit(
            f"[bold]False Positive Rate Test Results[/bold]\n\n"
            f"Total audio:         {self.total_duration_hours:.2f} hours\n"
            f"Threshold:           {self.threshold}\n"
            f"False positives:     {self.total_false_positives}\n"
            f"FP/hour:             [{fp_color}]{self.false_positives_per_hour:.2f}[/{fp_color}]",
            title="DiPCo Evaluation"
        ))

        # Per-file breakdown if there are detections
        if self.per_file_results:
            table = Table(title="Per-File Breakdown")
            table.add_column("File", style="cyan")
            table.add_column("Duration", style="white")
            table.add_column("False Positives", style="yellow")

            for result in self.per_file_results:
                duration_str = f"{result['duration_seconds']:.1f}s"
                fp_str = str(result['false_positives'])
                if result['false_positives'] > 0:
                    fp_str = f"[red]{fp_str}[/red]"
                table.add_row(result['file'], duration_str, fp_str)

            console.print(table)

    def save(self, path: Path):
        """Save results to JSON"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        console.print(f"\nResults saved to: {path}")


def evaluate_false_positive_rate(
    detector: ModelDetector,
    audio_files: List[Path],
    threshold: float = 0.5,
    debounce_seconds: float = 2.0,
) -> FalsePositiveRateResult:
    """
    Evaluate false positive rate by streaming audio through the model.

    Args:
        detector: ModelDetector instance (model will be loaded if needed)
        audio_files: List of audio file paths to evaluate
        threshold: Detection confidence threshold
        debounce_seconds: Minimum seconds between detections to count as separate
    """
    import soundfile as sf

    if not detector.model_loaded:
        detector.load_model()

    result = FalsePositiveRateResult(threshold=threshold)
    total_seconds = 0.0
    total_fps = 0

    chunk_size = 1280
    frame_duration = chunk_size / 16000.0
    model_name = list(detector.model.models.keys())[0]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        console=console
    ) as progress:
        task = progress.add_task("Evaluating FP rate", total=len(audio_files))

        for audio_path in audio_files:
            try:
                audio, sr = sf.read(str(audio_path), dtype='int16')

                if sr != 16000:
                    import scipy.signal as signal
                    num_samples = int(len(audio) * 16000 / sr)
                    audio = signal.resample(audio, num_samples).astype(np.int16)

                if audio.ndim > 1:
                    audio = audio.mean(axis=1).astype(np.int16)

                file_duration = len(audio) / 16000.0
                total_seconds += file_duration

                if hasattr(detector.model, 'reset'):
                    detector.model.reset()

                predictions = detector.model.predict_clip(
                    audio, padding=0, chunk_size=chunk_size
                )

                file_fps = 0
                last_detection_time = -debounce_seconds  # Allow first detection

                for frame_idx, pred in enumerate(predictions):
                    score = pred.get(model_name, 0.0)
                    frame_time = frame_idx * frame_duration

                    if score >= threshold and (frame_time - last_detection_time) >= debounce_seconds:
                        file_fps += 1
                        last_detection_time = frame_time
                        result.detections.append({
                            'file': audio_path.name,
                            'timestamp_seconds': round(frame_time, 2),
                            'score': round(float(score), 4)
                        })

                total_fps += file_fps
                result.per_file_results.append({
                    'file': audio_path.name,
                    'duration_seconds': round(file_duration, 1),
                    'false_positives': file_fps
                })

            except Exception as e:
                console.print(f"  [yellow]Skipped {audio_path.name}: {e}[/yellow]")
                continue

            progress.update(task, advance=1)

    result.total_duration_hours = total_seconds / 3600.0
    result.total_false_positives = total_fps
    if result.total_duration_hours > 0:
        result.false_positives_per_hour = total_fps / result.total_duration_hours

    return result
