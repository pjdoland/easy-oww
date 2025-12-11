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
        duration: float = 3.0
    ) -> Optional[Path]:
        """
        Record a single sample with playback and accept/re-record option

        Args:
            sample_number: Sample number (for filename)
            device: Device index
            duration: Recording duration in seconds

        Returns:
            Path to saved file or None if failed
        """
        output_path = self.output_dir / f"sample_{sample_number:04d}.wav"

        while True:  # Loop until user accepts or cancels
            console.print(f"\n[bold]Recording sample {sample_number}[/bold]")
            console.print(f"Get ready to speak ({duration:.1f} seconds)...")

            # Countdown with clear visual feedback
            console.print("\n[yellow]Starting in:[/yellow]")
            for i in range(3, 0, -1):
                console.print(f"  [bold yellow]{i}[/bold yellow]")
                time.sleep(1)

            console.print("\n[bold green]🎤 SPEAK NOW![/bold green]")

            # Small delay to let user start speaking
            time.sleep(0.3)

            try:
                # Record
                audio = self.recorder.record_duration(duration, device)
                console.print("  ✓ Recorded!  ")

                # Validate
                validation = self.validator.validate_audio(audio)

                # Show quality info
                if validation['warnings']:
                    for warning in validation['warnings']:
                        console.print(f"[yellow]⚠[/yellow] {warning}")

                if not validation['valid']:
                    console.print(f"[red]✗[/red] Sample quality issues:")
                    for issue in validation['issues']:
                        console.print(f"  • {issue}")

                # Playback the recording
                console.print("\n[cyan]Playing back your recording...[/cyan]")
                time.sleep(0.5)
                self.recorder.playback_audio(audio, device)

                # Ask user what to do
                console.print("\n[bold]What would you like to do?[/bold]")
                console.print("  a - Accept and continue")
                console.print("  r - Re-record this sample")
                console.print("  s - Skip this sample")

                choice = Prompt.ask(
                    "\nYour choice",
                    choices=['a', 'r', 's'],
                    default='a'
                )

                if choice == 'a':
                    # Accept - save and return
                    self.recorder.save_wav(audio, output_path)
                    if validation['valid']:
                        console.print(f"[green]✓[/green] Sample accepted: {output_path.name}")
                    else:
                        console.print(f"[yellow]⚠[/yellow] Sample accepted (with quality issues): {output_path.name}")
                    return output_path

                elif choice == 'r':
                    # Re-record - loop again
                    console.print("[cyan]Re-recording...[/cyan]")
                    continue

                elif choice == 's':
                    # Skip - return None
                    console.print("[yellow]Sample skipped[/yellow]")
                    return None

            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                logger.exception("Recording failed")

                # Ask if user wants to retry
                if Confirm.ask("Retry recording?", default=True):
                    continue
                else:
                    return None

    def record_session(
        self,
        count: int,
        device: Optional[int] = None,
        duration: float = 3.0
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
            f"You will record [bold]{count}[/bold] samples.\n"
            f"Each recording will be [bold]{duration:.1f}[/bold] seconds long.\n\n"
            f"After each recording, you'll hear a playback and can choose to:\n"
            f"  • Accept and continue\n"
            f"  • Re-record the sample\n"
            f"  • Skip the sample\n\n"
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
                # Temporarily stop progress display for recording
                progress.stop()

                # Record sample (includes playback and accept/re-record logic)
                file_path = self.record_sample(sample_num, device, duration)

                # Restart progress display
                progress.start()

                if file_path:
                    # Sample was accepted
                    recorded_files.append(file_path)
                    progress.update(
                        task,
                        advance=1,
                        description=f"Recording samples ({len(recorded_files)}/{count})"
                    )
                    sample_num += 1
                else:
                    # Sample was skipped - move to next sample number
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
    duration: float = 3.0,
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


def run_negative_recording_session(
    output_dir: Path,
    count: int = 20,
    duration: float = 3.0,
    sample_rate: int = 16000,
    examples: Optional[List[str]] = None
) -> List[Path]:
    """
    Run a recording session for negative/adversarial samples

    Args:
        output_dir: Directory to save recordings
        count: Number of samples to record
        duration: Duration per sample in seconds
        sample_rate: Audio sample rate
        examples: Optional list of example phrases to record

    Returns:
        List of successfully recorded file paths
    """
    console.print(Panel.fit(
        "[bold cyan]Negative Sample Recording[/bold cyan]\n\n"
        "You will now record NEGATIVE samples - these are phrases that should NOT\n"
        "trigger your wake word detection. This helps reduce false positives.\n\n"
        "[yellow]Examples of good negative samples:[/yellow]\n"
        "  • Similar-sounding phrases (\"hey system\" vs \"hey assistant\")\n"
        "  • Common phrases in your environment\n"
        "  • Words that rhyme with your wake word\n"
        "  • Partial wake words (\"hey\" or \"assistant\" alone)\n\n"
        "Recording these helps your model learn what NOT to respond to!",
        title="Adversarial Recording"
    ))

    # Show example phrases if provided
    if examples:
        console.print("\n[bold]Suggested phrases to record:[/bold]")
        for i, example in enumerate(examples, 1):
            console.print(f"  {i}. \"{example}\"")
        console.print()

    if not Confirm.ask("\nReady to start recording negative samples?", default=True):
        console.print("[yellow]Recording cancelled[/yellow]")
        return []

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

    # Record samples with custom instructions
    console.print(Panel.fit(
        f"[bold cyan]Recording Negative Samples[/bold cyan]\n\n"
        f"You will record [bold]{count}[/bold] negative samples.\n"
        f"Each recording will be [bold]{duration}[/bold] seconds long.\n\n"
        f"[yellow]Tips:[/yellow]\n"
        f"  • Vary the phrases you say\n"
        f"  • Use similar-sounding words to your wake word\n"
        f"  • Include common background phrases\n"
        f"  • Try different tones and speeds",
        title="Negative Recording Instructions"
    ))

    if not Confirm.ask("\nReady to start?", default=True):
        console.print("[yellow]Recording cancelled[/yellow]")
        return []

    recorded_files = ui.record_session(count, device, duration)

    # Show summary
    console.print("\n" + "=" * 60)
    console.print(Panel.fit(
        f"[bold green]Negative Recording Session Complete![/bold green]\n\n"
        f"Successfully recorded [bold]{len(recorded_files)}[/bold] negative samples\n"
        f"Saved to: {output_dir}",
        title="Session Summary"
    ))

    return recorded_files
