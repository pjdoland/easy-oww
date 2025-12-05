"""
Command implementations for easy-oww CLI
"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from easy_oww.utils.paths import PathManager
from easy_oww.utils.system import SystemChecker
from easy_oww.utils.logger import get_logger

console = Console()
logger = get_logger()


def setup_tts(paths, verbose=False):
    """
    Setup TTS system (Piper installation)

    Args:
        paths: PathManager instance
        verbose: Enable verbose output
    """
    from easy_oww.tts import PiperTTS
    from rich.prompt import Confirm

    piper = PiperTTS(paths.piper)

    if piper.is_installed():
        console.print("[green]✓[/green] Piper TTS already installed")
        return

    # Ask user if they want to install Piper
    console.print("\nPiper TTS is required for generating synthetic training samples.")

    if Confirm.ask("Install Piper TTS now?", default=True):
        try:
            piper.install()
            console.print("[green]✓[/green] Piper TTS installed successfully")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Piper installation failed: {e}")
            console.print("You can install it later manually")
            if verbose:
                import traceback
                traceback.print_exc()
    else:
        console.print("[yellow]⚠[/yellow] Skipped Piper installation")
        console.print("You'll need to install it manually for training")


def init_workspace(workspace_path=None, verbose=False):
    """
    Initialize easy-oww workspace

    Args:
        workspace_path: Custom workspace path
        verbose: Enable verbose output
    """
    console.print(Panel.fit(
        "[bold cyan]Welcome to easy-oww![/bold cyan]\n\n"
        "This tool will guide you through creating custom wake word models.\n\n"
        "The process involves:\n"
        "  1. Downloading required datasets (~50GB)\n"
        "  2. Recording your wake word samples\n"
        "  3. Training your custom model\n\n"
        "This will take approximately 2-4 hours depending on your hardware.",
        title="easy-oww Setup"
    ))

    # Initialize path manager
    paths = PathManager(workspace_path)

    console.print("\n[bold]Checking system requirements...[/bold]")

    # Check system requirements
    requirements = SystemChecker.check_all_requirements(str(paths.workspace))

    # Display results
    table = Table(title="System Requirements")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="white")

    # Python version
    python_status = "✓" if requirements['python']['valid'] else "✗"
    table.add_row(
        "Python Version",
        f"[{'green' if requirements['python']['valid'] else 'red'}]{python_status}[/]",
        f"{requirements['python']['version']} ({'OK' if requirements['python']['valid'] else 'Requires 3.7+'})"
    )

    # Disk space
    disk_status = "✓" if requirements['disk']['sufficient'] else "✗"
    table.add_row(
        "Disk Space",
        f"[{'green' if requirements['disk']['sufficient'] else 'yellow'}]{disk_status}[/]",
        f"{requirements['disk']['available_gb']}GB available ({'OK' if requirements['disk']['sufficient'] else 'Recommended: 60GB+'})"
    )

    # Memory
    mem_status = "✓" if requirements['memory']['sufficient'] else "✗"
    table.add_row(
        "RAM",
        f"[{'green' if requirements['memory']['sufficient'] else 'yellow'}]{mem_status}[/]",
        f"{requirements['memory']['available_gb']}GB available ({'OK' if requirements['memory']['sufficient'] else 'Recommended: 8GB+'})"
    )

    # GPU
    if requirements['gpu']['available']:
        table.add_row(
            "GPU",
            "[green]✓[/]",
            f"{requirements['gpu']['device_name']} detected (training will be faster)"
        )
    else:
        table.add_row(
            "GPU",
            "[yellow]○[/]",
            "No CUDA GPU detected (will use CPU - slower)"
        )

    console.print(table)

    # Check if we can proceed
    if not requirements['python']['valid']:
        console.print("\n[bold red]Error:[/bold red] Python 3.7 or higher is required")
        return

    # Create workspace structure
    console.print(f"\n[bold]Creating workspace at:[/bold] {paths.workspace}")
    paths.ensure_structure()

    # Setup TTS
    console.print("\n[bold]Setting up Text-to-Speech...[/bold]")
    setup_tts(paths, verbose)

    console.print("\n[green]✓ Workspace initialized successfully![/green]")
    console.print(f"\nWorkspace location: {paths.workspace}")
    console.print("\nNext steps:")
    console.print("  1. Download datasets: [cyan]easy-oww download[/cyan]")
    console.print("  2. Download TTS voices: [cyan]easy-oww download-voices[/cyan]")
    console.print("  3. Create a project: [cyan]easy-oww create <project_name>[/cyan]")


def download_datasets(workspace_path=None, required_only=False, verbose=False):
    """
    Download required datasets

    Args:
        workspace_path: Custom workspace path
        required_only: Download only required datasets
        verbose: Enable verbose output
    """
    from easy_oww.datasets.manager import DatasetManager

    paths = PathManager(workspace_path)
    paths.ensure_structure()

    console.print("\n[bold cyan]Dataset Download[/bold cyan]\n")

    # Initialize dataset manager
    manager = DatasetManager(
        str(paths.datasets),
        str(paths.cache)
    )

    try:
        manager.download_all(required_only=required_only)
    except KeyboardInterrupt:
        console.print("\n[yellow]Download interrupted[/yellow]")
        console.print("You can resume by running the same command again")
    except Exception as e:
        console.print(f"\n[red]Download failed: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()


def create_project(project_name, workspace_path=None, wake_word=None, samples=1000, steps=10000, verbose=False):
    """
    Create new wake word project

    Args:
        project_name: Name of the project
        workspace_path: Custom workspace path
        wake_word: Wake word/phrase
        samples: Number of training samples
        steps: Training steps
        verbose: Enable verbose output
    """
    from easy_oww.training import ConfigManager
    from easy_oww.tts import VoiceDownloader
    from rich.prompt import Prompt

    paths = PathManager(workspace_path)

    if paths.project_exists(project_name):
        console.print(f"[red]Error:[/red] Project '{project_name}' already exists")
        return

    console.print(f"\n[bold]Creating project:[/bold] {project_name}")

    # Get wake word if not provided
    if not wake_word:
        wake_word = Prompt.ask("Enter your wake word/phrase")

    # Create project structure
    paths.create_project_structure(project_name)

    # Create training configuration
    project_path = paths.get_project_path(project_name)
    config_manager = ConfigManager(project_path)

    # Get available voices
    voice_downloader = VoiceDownloader(paths.voices)
    installed_voices = voice_downloader.list_installed_voices()

    voice_names = [v['name'] for v in installed_voices[:3]] if installed_voices else []

    # Create config
    config = config_manager.create_default(
        project_name=project_name,
        wake_word=wake_word
    )

    # Update with parameters
    config.target_samples = samples
    config.max_steps = steps
    config.voices = voice_names

    config_manager.save(config)

    console.print(f"[green]✓ Project created successfully![/green]")
    console.print(f"\nProject location: {project_path}")
    console.print(f"Wake word: {wake_word}")

    if not voice_names:
        console.print("\n[yellow]⚠ No TTS voices installed[/yellow]")
        console.print("Download voices for synthetic sample generation:")
        console.print("  [cyan]easy-oww download-voices[/cyan]")

    console.print("\nNext steps:")
    console.print(f"  1. Record samples: [cyan]easy-oww record {project_name}[/cyan]")
    console.print(f"  2. Train model: [cyan]easy-oww train {project_name}[/cyan]")


def record_samples(project_name, workspace_path=None, count=20, verbose=False):
    """
    Record wake word samples

    Args:
        project_name: Name of the project
        workspace_path: Custom workspace path
        count: Number of samples to record
        verbose: Enable verbose output
    """
    from easy_oww.audio import run_recording_session

    paths = PathManager(workspace_path)

    if not paths.project_exists(project_name):
        console.print(f"[red]Error:[/red] Project '{project_name}' not found")
        console.print(f"Create it first with: [cyan]easy-oww create {project_name}[/cyan]")
        return

    # Get recordings directory for this project
    recordings_dir = paths.get_project_path(project_name) / 'recordings'

    console.print(f"\n[bold cyan]Recording samples for:[/bold cyan] {project_name}")
    console.print(f"[bold]Output directory:[/bold] {recordings_dir}\n")

    try:
        # Run recording session
        recorded_files = run_recording_session(
            output_dir=recordings_dir,
            count=count,
            duration=2.0,
            sample_rate=16000
        )

        if recorded_files:
            console.print(f"\n[green]✓ Successfully recorded {len(recorded_files)} samples![/green]")
            console.print(f"\nNext step:")
            console.print(f"  Train model: [cyan]easy-oww train {project_name}[/cyan]")
        else:
            console.print("\n[yellow]No samples were recorded[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Recording interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Recording failed: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()


def train_model(project_name, workspace_path=None, resume=False, verbose=False):
    """
    Train wake word model

    Args:
        project_name: Name of the project
        workspace_path: Custom workspace path
        resume: Resume from last checkpoint
        verbose: Enable verbose output
    """
    from easy_oww.training import run_training

    paths = PathManager(workspace_path)

    if not paths.project_exists(project_name):
        console.print(f"[red]Error:[/red] Project '{project_name}' not found")
        console.print(f"Create it first with: [cyan]easy-oww create {project_name}[/cyan]")
        return

    project_path = paths.get_project_path(project_name)

    try:
        # Run training pipeline
        run_training(
            project_path=project_path,
            workspace_path=paths.workspace,
            resume=resume,
            verbose=verbose
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Training failed: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()


def test_model(project_name, workspace_path=None, duration=60, verbose=False):
    """
    Test trained model

    Args:
        project_name: Name of the project
        workspace_path: Custom workspace path
        duration: Test duration in seconds
        verbose: Enable verbose output
    """
    from easy_oww.testing import run_realtime_test, ModelDetector, evaluate_model_on_dataset
    from rich.prompt import Confirm, Prompt

    paths = PathManager(workspace_path)

    if not paths.project_exists(project_name):
        console.print(f"[red]Error:[/red] Project '{project_name}' not found")
        return

    project_path = paths.get_project_path(project_name)
    models_dir = project_path / 'models'

    # Find model file
    if not models_dir.exists():
        console.print(f"[red]Error:[/red] No models directory found")
        console.print(f"Train a model first: [cyan]easy-oww train {project_name}[/cyan]")
        return

    model_files = list(models_dir.glob('*.onnx'))

    if not model_files:
        console.print(f"[red]Error:[/red] No trained models found")
        console.print(f"Train a model first: [cyan]easy-oww train {project_name}[/cyan]")
        return

    # Select model if multiple
    if len(model_files) > 1:
        console.print("\n[bold]Available models:[/bold]")
        for i, model_file in enumerate(model_files, 1):
            console.print(f"  {i}. {model_file.name}")

        choice = Prompt.ask("Select model", choices=[str(i) for i in range(1, len(model_files) + 1)])
        model_path = model_files[int(choice) - 1]
    else:
        model_path = model_files[0]

    console.print(f"\n[bold]Testing model:[/bold] {model_path.name}")

    # Ask test type
    console.print("\n[bold]Test options:[/bold]")
    console.print("  1. Real-time microphone test")
    console.print("  2. Evaluate on test clips")
    console.print("  3. Both")

    test_choice = Prompt.ask("Select test type", choices=['1', '2', '3'], default='1')

    try:
        # Real-time test
        if test_choice in ['1', '3']:
            threshold = 0.5
            if Confirm.ask("\nCustomize detection threshold?", default=False):
                threshold_str = Prompt.ask("Enter threshold (0.0-1.0)", default="0.5")
                threshold = float(threshold_str)

            run_realtime_test(
                model_path=model_path,
                duration=duration,
                threshold=threshold
            )

        # Clip evaluation
        if test_choice in ['2', '3']:
            clips_dir = project_path / 'clips'

            if not clips_dir.exists():
                console.print("\n[yellow]No clips directory found, skipping evaluation[/yellow]")
            else:
                console.print("\n[bold cyan]Evaluating on test clips...[/bold cyan]")

                # Load clips
                positive_clips = list((clips_dir / 'positive').glob('*.wav'))
                negative_clips = list((clips_dir / 'negative').glob('*.wav'))

                if not positive_clips and not negative_clips:
                    console.print("[yellow]No test clips found[/yellow]")
                else:
                    # Limit to subset for faster testing
                    max_samples = 100
                    positive_clips = positive_clips[:max_samples]
                    negative_clips = negative_clips[:max_samples]

                    # Create detector
                    detector = ModelDetector(model_path=model_path)

                    # Evaluate
                    tracker = evaluate_model_on_dataset(
                        detector=detector,
                        positive_clips=positive_clips,
                        negative_clips=negative_clips
                    )

                    # Display results
                    tracker.display_summary()

                    # Save results
                    results_path = project_path / 'test_results.json'
                    tracker.save_results(results_path)

    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Test failed: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()


def list_projects(workspace_path=None):
    """
    List all projects

    Args:
        workspace_path: Custom workspace path
    """
    paths = PathManager(workspace_path)

    if not paths.projects.exists():
        console.print("[yellow]No projects found[/yellow]")
        console.print(f"Create one with: [cyan]easy-oww create <project_name>[/cyan]")
        return

    projects = [p for p in paths.projects.iterdir() if p.is_dir()]

    if not projects:
        console.print("[yellow]No projects found[/yellow]")
        console.print(f"Create one with: [cyan]easy-oww create <project_name>[/cyan]")
        return

    table = Table(title="Projects")
    table.add_column("Project Name", style="cyan")
    table.add_column("Location", style="white")

    for project in sorted(projects):
        table.add_row(project.name, str(project))

    console.print(table)


def download_voices(workspace_path=None, language='en_US', count=2, verbose=False):
    """
    Download TTS voice models

    Args:
        workspace_path: Custom workspace path
        language: Language code (e.g., 'en_US', 'en_GB')
        count: Number of voices to download
        verbose: Enable verbose output
    """
    from easy_oww.tts import VoiceDownloader, PiperTTS
    from rich.prompt import Confirm

    paths = PathManager(workspace_path)

    # Check if Piper is installed
    piper = PiperTTS(paths.piper)
    if not piper.is_installed():
        console.print("[yellow]⚠[/yellow] Piper TTS is not installed")
        if Confirm.ask("Install Piper TTS now?", default=True):
            try:
                piper.install()
                console.print("[green]✓[/green] Piper TTS installed")
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to install Piper: {e}")
                return
        else:
            console.print("[yellow]Cancelled voice download[/yellow]")
            return

    # Initialize voice downloader
    downloader = VoiceDownloader(paths.voices)

    # Show available voices
    available = downloader.list_available_voices(language)

    if not available:
        console.print(f"[red]Error:[/red] No voices available for language: {language}")
        console.print("\nSupported languages: en_US, en_GB, es_ES, fr_FR, de_DE")
        return

    console.print(f"\n[bold]Available voices for {language}:[/bold]")
    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Quality", style="yellow")
    table.add_column("Size", style="magenta")

    for voice in available[:count]:
        table.add_row(
            voice['name'],
            voice['description'],
            voice['quality'],
            f"{voice['size_mb']} MB"
        )

    console.print(table)

    # Confirm download
    total_size = sum(v['size_mb'] for v in available[:count])
    console.print(f"\nTotal download size: ~{total_size} MB")

    if not Confirm.ask("Download these voices?", default=True):
        console.print("[yellow]Download cancelled[/yellow]")
        return

    # Download voices
    try:
        downloaded = downloader.download_recommended_voices(language, count)

        if downloaded:
            console.print(f"\n[green]✓ Successfully downloaded {len(downloaded)} voice(s)[/green]")

            # Test first voice
            console.print("\n[bold]Testing voice...[/bold]")
            success, message = piper.test_voice(downloaded[0])
            if success:
                console.print(f"[green]✓[/green] {message}")
            else:
                console.print(f"[yellow]⚠[/yellow] {message}")

        else:
            console.print("[yellow]No voices were downloaded[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] Download failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def list_voices(workspace_path=None):
    """
    List installed TTS voices

    Args:
        workspace_path: Custom workspace path
    """
    from easy_oww.tts import VoiceDownloader

    paths = PathManager(workspace_path)
    downloader = VoiceDownloader(paths.voices)

    installed = downloader.list_installed_voices()

    if not installed:
        console.print("[yellow]No voices installed[/yellow]")
        console.print("Download voices with: [cyan]easy-oww download-voices[/cyan]")
        return

    console.print(f"\n[bold]Installed Voices ({len(installed)}):[/bold]")
    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Language", style="white")
    table.add_column("Size", style="magenta")

    for voice in installed:
        table.add_row(
            voice['name'],
            voice.get('language', 'unknown'),
            f"{voice['size_mb']:.1f} MB"
        )

    console.print(table)
