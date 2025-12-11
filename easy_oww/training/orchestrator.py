"""
Training orchestration - ties together the complete training pipeline
"""
from pathlib import Path
from typing import Optional, Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from easy_oww.training.config import TrainingConfig, ConfigManager
from easy_oww.training.clips import ClipGenerator
from easy_oww.training.augmentation import AudioAugmenter
from easy_oww.tts import PiperTTS, VoiceDownloader
from easy_oww.utils.logger import get_logger

logger = get_logger()
console = Console()


class TrainingOrchestrator:
    """Orchestrates the complete wake word training pipeline"""

    def __init__(
        self,
        project_path: Path,
        workspace_path: Path
    ):
        """
        Initialize training orchestrator

        Args:
            project_path: Path to project directory
            workspace_path: Path to workspace directory
        """
        self.project_path = Path(project_path)
        self.workspace_path = Path(workspace_path)

        # Initialize managers
        self.config_manager = ConfigManager(self.project_path)

        # Paths
        self.datasets_dir = self.workspace_path / 'datasets'
        self.piper_dir = self.workspace_path / 'piper-sample-generator'
        self.voices_dir = self.workspace_path / 'voices'

    def run(self, resume: bool = False, verbose: bool = False, force: bool = False):
        """
        Run complete training pipeline

        Args:
            resume: Resume from last checkpoint
            verbose: Enable verbose output
            force: Force full retrain (regenerate all clips and features)
        """
        console.print(Panel.fit(
            "[bold cyan]Wake Word Training Pipeline[/bold cyan]\n\n"
            "This will train a custom wake word model using:\n"
            "  1. Your recorded samples\n"
            "  2. Generated synthetic samples (TTS)\n"
            "  3. Audio augmentation (RIR, noise)\n"
            "  4. OpenWakeWord model training",
            title="Training"
        ))

        # Load configuration
        config = self._load_config()

        # Display training plan
        self._display_training_plan(config)

        # Validate configuration
        self._validate_config(config)

        # Phase 1: Generate clips
        console.print("\n[bold cyan]Phase 1: Clip Generation[/bold cyan]")
        self._generate_clips(config, force)

        # Phase 2: Augment clips
        console.print("\n[bold cyan]Phase 2: Audio Augmentation[/bold cyan]")
        self._augment_clips(config, force)

        # Phase 3: Train model
        console.print("\n[bold cyan]Phase 3: Model Training[/bold cyan]")
        self._train_model(config, resume, force)

        # Complete
        console.print("\n[green]✓ Training complete![/green]")
        self._display_completion_summary(config)

    def _load_config(self) -> TrainingConfig:
        """
        Load training configuration

        Returns:
            TrainingConfig instance

        Raises:
            RuntimeError: If config doesn't exist
        """
        if not self.config_manager.exists():
            raise RuntimeError(
                f"Training config not found. "
                f"Create it first with project setup."
            )

        config = self.config_manager.load()
        logger.info(f"Loaded config for project: {config.project_name}")

        return config

    def _display_training_plan(self, config: TrainingConfig):
        """
        Display training plan to user

        Args:
            config: Training configuration
        """
        summary = config.get_summary()

        table = Table(title="Training Plan")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")

        for key, value in summary.items():
            table.add_row(key, str(value))

        console.print("\n")
        console.print(table)

    def _validate_config(self, config: TrainingConfig):
        """
        Validate training configuration

        Args:
            config: Training configuration

        Raises:
            RuntimeError: If validation fails
        """
        is_valid, issues = config.validate()

        if not is_valid:
            console.print("\n[red]Configuration validation failed:[/red]")
            for issue in issues:
                console.print(f"  • {issue}")
            raise RuntimeError("Invalid training configuration")

        console.print("\n[green]✓[/green] Configuration validated")

    def _clips_exist(self, config: TrainingConfig) -> bool:
        """Check if clips have already been generated"""
        clips_dir = Path(config.clips_dir)
        positive_dir = clips_dir / 'positive'
        negative_dir = clips_dir / 'negative'

        if not positive_dir.exists() or not negative_dir.exists():
            return False

        positive_count = len(list(positive_dir.glob('*.wav')))
        negative_count = len(list(negative_dir.glob('*.wav')))

        # Require at least some clips to be present
        return positive_count > 0 and negative_count > 0

    def _augmented_clips_exist(self, config: TrainingConfig) -> bool:
        """Check if augmented clips have already been generated"""
        if not config.use_augmentation:
            return True  # Skip check if augmentation is disabled

        clips_dir = Path(config.clips_dir)
        augmented_dir = clips_dir / 'positive_augmented'

        if not augmented_dir.exists():
            return False

        augmented_count = len(list(augmented_dir.glob('*.wav')))
        return augmented_count > 0

    def _generate_clips(self, config: TrainingConfig, force: bool = False):
        """
        Generate training clips

        Args:
            config: Training configuration
            force: Force regeneration even if clips exist
        """
        # Check if clips already exist
        if not force and self._clips_exist(config):
            console.print("[green]✓[/green] Clips already generated, skipping...")

            # Show counts
            clips_dir = Path(config.clips_dir)
            positive_count = len(list((clips_dir / 'positive').glob('*.wav')))
            negative_count = len(list((clips_dir / 'negative').glob('*.wav')))
            console.print(f"  Positive clips: {positive_count}")
            console.print(f"  Negative clips: {negative_count}")
            return

        if force:
            console.print("[yellow]⚠[/yellow] Force flag set - regenerating all clips...")

        # Check for negative recordings directory
        negative_recordings_dir = self.project_path / 'recordings_negative'

        clip_generator = ClipGenerator(
            recordings_dir=Path(config.recordings_dir),
            clips_dir=Path(config.clips_dir),
            sample_rate=config.sample_rate,
            target_duration_ms=config.clip_duration_ms,
            negative_recordings_dir=negative_recordings_dir if negative_recordings_dir.exists() else None
        )

        # Process real recordings
        real_clips = clip_generator.process_real_recordings()

        if len(real_clips) < config.real_samples:
            console.print(
                f"[yellow]⚠[/yellow] Only found {len(real_clips)} real recordings "
                f"(expected {config.real_samples})"
            )

        # Process negative recordings (adversarial samples)
        negative_real_clips = clip_generator.process_negative_recordings()
        if negative_real_clips:
            console.print(f"[green]✓[/green] Added {len(negative_real_clips)} adversarial samples from recordings")

        # Generate synthetic clips
        if config.synthetic_samples > 0:
            # Check Piper and voices
            # Pass workspace path - PiperTTS will look for voices in workspace/voices
            piper = PiperTTS(self.workspace_path)
            if not piper.is_installed():
                raise RuntimeError("Piper TTS is not installed. Run 'easy-oww init' first.")

            # Get voice models
            voice_models = []
            for voice_name in config.voices:
                voice_path = piper.get_voice(voice_name)
                if voice_path:
                    voice_models.append(voice_path)
                else:
                    console.print(f"[yellow]⚠[/yellow] Voice not found: {voice_name}")

            if not voice_models:
                raise RuntimeError(
                    "No voice models available. "
                    "Download voices with 'easy-oww download-voices'"
                )

            console.print(f"Using {len(voice_models)} voices for generation")

            # Generate synthetic clips
            synthetic_clips = clip_generator.generate_synthetic_clips(
                wake_word=config.wake_word,
                voice_models=voice_models,
                piper=piper,
                count=config.synthetic_samples
            )

        # Generate negative clips
        # Use FSD50K dataset for negative samples
        fsd50k_dir = self.datasets_dir / 'fsd50k'
        if fsd50k_dir.exists():
            negative_clips = clip_generator.generate_negative_clips(
                negative_audio_dir=fsd50k_dir,
                count=config.target_samples  # Equal number of negatives
            )
        else:
            console.print("[yellow]⚠[/yellow] FSD50K dataset not found, skipping negative samples")
            console.print("Download with: [cyan]easy-oww download[/cyan]")

        # Verify clips
        console.print("\n[bold]Verifying clips...[/bold]")
        verification = clip_generator.verify_clips()

        for clip_type in ['positive', 'negative']:
            valid = verification[clip_type]['valid']
            invalid = verification[clip_type]['invalid']
            console.print(f"  {clip_type.capitalize()}: {valid} valid, {invalid} invalid")

            if invalid > 0 and invalid < 10:
                for issue in verification[clip_type]['issues'][:5]:
                    console.print(f"    • {issue}")

    def _augment_clips(self, config: TrainingConfig, force: bool = False):
        """
        Apply audio augmentation to clips

        Args:
            config: Training configuration
            force: Force regeneration even if augmented clips exist
        """
        if not config.use_augmentation:
            console.print("[yellow]Augmentation disabled, skipping[/yellow]")
            return

        # Check if augmented clips already exist
        if not force and self._augmented_clips_exist(config):
            console.print("[green]✓[/green] Augmented clips already generated, skipping...")
            clips_dir = Path(config.clips_dir)
            augmented_count = len(list((clips_dir / 'positive_augmented').glob('*.wav')))
            console.print(f"  Augmented clips: {augmented_count}")
            return

        if force:
            console.print("[yellow]⚠[/yellow] Force flag set - regenerating augmented clips...")

        # Initialize augmenter
        rir_dir = self.datasets_dir / 'mit_rir'
        noise_dir = self.datasets_dir / 'fsd50k'

        # Check dataset availability
        console.print("\n[bold]Checking augmentation datasets:[/bold]")

        rir_available = rir_dir.exists()
        noise_available = noise_dir.exists()

        if rir_available:
            rir_count = len(list(rir_dir.rglob('*.wav')))
            console.print(f"  [green]✓[/green] RIR dataset: {rir_count} files")
        else:
            console.print(f"  [yellow]✗[/yellow] RIR dataset: Not found at {rir_dir}")
            console.print(f"      Download with: [cyan]easy-oww download --required-only[/cyan]")

        if noise_available:
            # Count noise files in subdirectories
            noise_count = 0
            for subdir in ['dev', 'eval', 'FSD50K.dev_audio', 'FSD50K.eval_audio']:
                subdir_path = noise_dir / subdir
                if subdir_path.exists():
                    noise_count += len(list(subdir_path.glob('*.wav')))

            if noise_count > 0:
                console.print(f"  [green]✓[/green] Noise dataset (FSD50K): {noise_count} files")
            else:
                console.print(f"  [yellow]✗[/yellow] Noise dataset: Directory exists but no audio files found")
                console.print(f"      Download with: [cyan]easy-oww download[/cyan] (without --required-only)")
        else:
            console.print(f"  [yellow]✗[/yellow] Noise dataset: Not found at {noise_dir}")
            console.print(f"      Download with: [cyan]easy-oww download[/cyan] (without --required-only)")

        if not rir_available and not noise_available:
            console.print("\n[yellow]⚠ Warning:[/yellow] No augmentation datasets available!")
            console.print("  Augmentation will only apply pitch/time/volume changes.")
            console.print("  For better results, download datasets with: [cyan]easy-oww download[/cyan]")

        augmenter = AudioAugmenter(
            rir_dir=rir_dir if rir_available else None,
            noise_dir=noise_dir if noise_available else None,
            sample_rate=config.sample_rate
        )

        # Get positive clips
        positive_dir = Path(config.clips_dir) / 'positive'
        positive_clips = list(positive_dir.glob('*.wav'))

        if not positive_clips:
            console.print("[yellow]⚠[/yellow] No positive clips to augment")
            return

        # Augment clips
        augmented_dir = Path(config.clips_dir) / 'positive_augmented'
        augmented_clips = augmenter.augment_clips(
            input_clips=positive_clips,
            output_dir=augmented_dir,
            augmentations_per_clip=2,  # 2 variations per clip
            rir_prob=config.rir_probability,
            noise_prob=config.noise_probability
        )

        console.print(f"[green]✓[/green] Augmented {len(positive_clips)} clips into {len(augmented_clips)} variations")

    def _train_model(self, config: TrainingConfig, resume: bool = False, force: bool = False):
        """
        Train wake word model

        Args:
            config: Training configuration
            resume: Resume from checkpoint
            force: Force full retrain (regenerate features)
        """
        from easy_oww.training.full_trainer import train_full_model

        console.print("\n[bold]Phase 3: Model Training[/bold]")

        # Get clip and output directories
        clips_dir = Path(config.clips_dir)
        models_dir = Path(config.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Count available clips
        positive_dir = clips_dir / 'positive'
        positive_aug_dir = clips_dir / 'positive_augmented'
        negative_dir = clips_dir / 'negative'

        total_positive = len(list(positive_dir.glob('*.wav'))) + len(list(positive_aug_dir.glob('*.wav')))
        total_negative = len(list(negative_dir.glob('*.wav')))

        console.print(f"\nTraining data:")
        console.print(f"  Positive samples: {total_positive}")
        console.print(f"  Negative samples: {total_negative}")
        console.print(f"  Total: {total_positive + total_negative}")

        if total_positive < 50:
            console.print("[yellow]⚠[/yellow] Low number of positive samples, model may not perform well")
            console.print("    Consider recording more samples or generating more synthetic clips")

        if total_negative < total_positive:
            console.print("[yellow]⚠[/yellow] Fewer negative than positive samples")
            console.print("    Model will generate adversarial negatives to compensate")

        # Train full model
        try:
            model_path = train_full_model(
                project_name=config.project_name,
                wake_word=config.wake_word,
                clips_dir=clips_dir,
                output_dir=models_dir,
                workspace_dir=self.workspace_path,
                model_type="dnn",
                layer_size=128,
                steps=5000,
                target_fp_per_hour=1.0,  # More aggressive - reduce false positives
                augmentation_rounds=2,
                batch_size=128,
                max_negative_weight=10,  # Balance: not too high (collapse) or too low (false positives)
                force=force
            )

            console.print(f"\n[green]✓[/green] Model saved to: {model_path}")

            # Store model path in config for testing
            config.model_path = str(model_path)
            self.config_manager.save(config)

        except Exception as e:
            logger.exception("Model training failed")
            raise RuntimeError(f"Failed to train model: {e}")

    def _display_completion_summary(self, config: TrainingConfig):
        """
        Display training completion summary

        Args:
            config: Training configuration
        """
        model_path = getattr(config, 'model_path', 'Not trained')

        console.print(Panel.fit(
            f"[bold green]Training Complete![/bold green]\n\n"
            f"Project: {config.project_name}\n"
            f"Wake Word: {config.wake_word}\n\n"
            f"Model: {model_path}\n"
            f"Clips: {config.clips_dir}",
            title="Success"
        ))

        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  1. Test the model: [cyan]easy-oww test {config.project_name}[/cyan]")
        console.print(f"  2. Review generated clips: {config.clips_dir}")
        console.print(f"\n[dim]Model location: {model_path}[/dim]")


def run_training(
    project_path: Path,
    workspace_path: Path,
    resume: bool = False,
    verbose: bool = False,
    force: bool = False
):
    """
    Run training pipeline

    Args:
        project_path: Path to project directory
        workspace_path: Path to workspace directory
        resume: Resume from checkpoint
        verbose: Enable verbose output
        force: Force full retrain (regenerate all clips and features)
    """
    orchestrator = TrainingOrchestrator(project_path, workspace_path)
    orchestrator.run(resume=resume, verbose=verbose, force=force)
