"""
Interactive UI for audio recording
"""
import time
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
import questionary

from easy_oww.audio.recorder import AudioRecorder
from easy_oww.audio.validator import AudioValidator
from easy_oww.utils.logger import get_logger

console = Console()
logger = get_logger()


class RecordingUI:
    """Interactive UI for recording wake word samples"""

    def __init__(
        self,
        output_dir: Path,
        sample_rate: int = 16000
    ):
        """
        Initialize recording UI

        Args:
            output_dir: Directory to save recordings
            sample_rate: Audio sample rate
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

        self.recorder = AudioRecorder(sample_rate=sample_rate)
        self.validator = AudioValidator(sample_rate=sample_rate)

    def select_microphone(self) -> Optional[int]:
        """
        Let user select microphone device

        Returns:
            Selected device index or None for default
        """
        devices = self.recorder.list_devices()

        if not devices:
            console.print("[red]Error:[/red] No input devices found")
            return None

        if len(devices) == 1:
            device = devices[0]
            console.print(f"[green]Using:[/green] {device['name']}")
            return device['index']

        # Show available devices
        console.print("\n[bold]Available microphones:[/bold]")
        table = Table()
        table.add_column("Index", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Channels", style="yellow")

        choices = []
        for device in devices:
            table.add_row(
                str(device['index']),
                device['name'],
                str(device['channels'])
            )
            choices.append(f"{device['index']}: {device['name']}")

        console.print(table)

        # Let user select
        try:
            choice = questionary.select(
                "Select microphone:",
                choices=choices
            ).ask()

            if choice is None:
                return None

            device_idx = int(choice.split(':')[0])
            return device_idx

        except KeyboardInterrupt:
            return None

    def test_microphone(self, device: Optional[int] = None) -> bool:
        """
        Test microphone with user

        Args:
            device: Device index

        Returns:
            True if microphone is working
        """
        console.print("\n[bold cyan]Testing microphone...[/bold cyan]")
        console.print("Please speak normally for 2 seconds...")

        time.sleep(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Recording...", total=None)

            success, message, audio = self.recorder.test_microphone(
                duration=2.0,
                device=device
            )

            progress.remove_task(task)

        if success:
            console.print(f"[green]✓[/green] {message}")
            return True
        else:
            console.print(f"[red]✗[/red] {message}")
            return False

    def record_sample(
        self,
        sample_number: int,
        device: Optional[int] = None,
        duration: float = 2.0
    ) -> Optional[Path]:
        """
        Record a single sample

        Args:
            sample_number: Sample number (for filename)
            device: Device index
            duration: Recording duration in seconds

        Returns:
            Path to saved file or None if failed
        """
        output_path = self.output_dir / f"sample_{sample_number:04d}.wav"

        console.print(f"\n[bold]Recording sample {sample_number}[/bold]")
        console.print(f"Speak your wake word in [cyan]{duration}[/cyan] seconds...")

        # Countdown
        for i in range(3, 0, -1):
            console.print(f"  {i}...", end="\r")
            time.sleep(1)

        console.print("  🎤 Recording!", end="\r")

        try:
            # Record
            audio = self.recorder.record_duration(duration, device)

            console.print("  ✓ Recorded!  ")

            # Validate
            validation = self.validator.validate_audio(audio)

            if validation['valid']:
                # Save
                self.recorder.save_wav(audio, output_path)
                console.print(f"[green]✓[/green] Sample saved: {output_path.name}")

                # Show warnings if any
                if validation['warnings']:
                    for warning in validation['warnings']:
                        console.print(f"[yellow]⚠[/yellow] {warning}")

                return output_path

            else:
                # Show issues
                console.print(f"[red]✗[/red] Sample quality issues:")
                for issue in validation['issues']:
                    console.print(f"  • {issue}")

                # Ask if user wants to keep it anyway
                if Confirm.ask("Keep this sample anyway?", default=False):
                    self.recorder.save_wav(audio, output_path)
                    console.print(f"[yellow]⚠[/yellow] Sample saved (with issues): {output_path.name}")
                    return output_path
                else:
                    return None

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            logger.exception("Recording failed")
            return None

    def record_session(
        self,
        count: int,
        device: Optional[int] = None,
        duration: float = 2.0
    ) -> List[Path]:
        """
        Record multiple samples in a session

        Args:
            count: Number of samples to record
            device: Device index
            duration: Recording duration per sample

        Returns:
            List of successfully recorded file paths
        """
        console.print(Panel.fit(
            f"[bold cyan]Recording Session[/bold cyan]\n\n"
            f"You will record [bold]{count}[/bold] samples of your wake word.\n"
            f"Each recording will be [bold]{duration}[/bold] seconds long.\n\n"
            f"[yellow]Tips:[/yellow]\n"
            f"  • Vary your tone and speed\n"
            f"  • Try different distances from mic\n"
            f"  • Include natural variations\n"
            f"  • Stay consistent with pronunciation",
            title="Recording Instructions"
        ))

        if not Confirm.ask("\nReady to start?", default=True):
            console.print("[yellow]Recording cancelled[/yellow]")
            return []

        recorded_files = []
        sample_num = 1

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Recording samples (0/{count})",
                total=count
            )

            while len(recorded_files) < count:
                # Record sample
                file_path = self.record_sample(sample_num, device, duration)

                if file_path:
                    recorded_files.append(file_path)
                    progress.update(
                        task,
                        advance=1,
                        description=f"Recording samples ({len(recorded_files)}/{count})"
                    )
                else:
                    # Ask if user wants to retry or skip
                    console.print("\n[yellow]Options:[/yellow]")
                    console.print("  r - Retry this sample")
                    console.print("  s - Skip this sample")
                    console.print("  q - Quit session")

                    choice = Prompt.ask(
                        "What would you like to do?",
                        choices=['r', 's', 'q'],
                        default='r'
                    )

                    if choice == 'q':
                        console.print("[yellow]Session ended early[/yellow]")
                        break
                    elif choice == 's':
                        progress.update(
                            task,
                            advance=1,
                            description=f"Recording samples ({len(recorded_files)}/{count})"
                        )

                sample_num += 1

                # Brief pause between recordings
                if len(recorded_files) < count:
                    time.sleep(0.5)

        return recorded_files

    def show_session_summary(self, recorded_files: List[Path]):
        """
        Show summary of recording session

        Args:
            recorded_files: List of recorded file paths
        """
        console.print("\n" + "=" * 60)
        console.print(Panel.fit(
            f"[bold green]Recording Session Complete![/bold green]\n\n"
            f"Successfully recorded [bold]{len(recorded_files)}[/bold] samples\n"
            f"Saved to: {self.output_dir}",
            title="Session Summary"
        ))

        if recorded_files:
            # Validate all samples
            console.print("\n[bold]Sample Quality Report:[/bold]")

            validation_results = self.validator.batch_validate(recorded_files)

            table = Table()
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Total Samples", str(validation_results['total']))
            table.add_row(
                "Valid Samples",
                f"[green]{validation_results['valid']}[/green]"
            )
            if validation_results['invalid'] > 0:
                table.add_row(
                    "Invalid Samples",
                    f"[red]{validation_results['invalid']}[/red]"
                )

            # Calculate average metrics
            all_metrics = [
                r['metrics']
                for r in validation_results['results'].values()
                if r['metrics']
            ]

            if all_metrics:
                avg_duration = sum(m.get('duration', 0) for m in all_metrics) / len(all_metrics)
                avg_level = sum(m.get('level_db', 0) for m in all_metrics) / len(all_metrics)

                table.add_row("Avg Duration", f"{avg_duration:.2f}s")
                table.add_row("Avg Level", f"{avg_level:.1f} dB")

            console.print(table)

            # Show files with issues
            files_with_issues = [
                (path, result)
                for path, result in validation_results['results'].items()
                if not result['valid']
            ]

            if files_with_issues:
                console.print("\n[yellow]Samples with issues:[/yellow]")
                for path, result in files_with_issues:
                    console.print(f"  • {Path(path).name}")
                    for issue in result['issues']:
                        console.print(f"    - {issue}")


def run_recording_session(
    output_dir: Path,
    count: int = 20,
    duration: float = 2.0,
    sample_rate: int = 16000
) -> List[Path]:
    """
    Run a complete recording session

    Args:
        output_dir: Directory to save recordings
        count: Number of samples to record
        duration: Duration per sample in seconds
        sample_rate: Audio sample rate

    Returns:
        List of successfully recorded file paths
    """
    ui = RecordingUI(output_dir, sample_rate)

    # Select microphone
    device = ui.select_microphone()
    if device is None:
        console.print("[red]No microphone selected[/red]")
        return []

    # Test microphone
    if not ui.test_microphone(device):
        if not Confirm.ask("Continue anyway?", default=False):
            console.print("[yellow]Recording cancelled[/yellow]")
            return []

    # Record samples
    recorded_files = ui.record_session(count, device, duration)

    # Show summary
    ui.show_session_summary(recorded_files)

    return recorded_files
