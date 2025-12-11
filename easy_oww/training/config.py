"""
Training configuration management
"""
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from easy_oww.utils.logger import get_logger

logger = get_logger()


@dataclass
class TrainingConfig:
    """Training configuration for wake word model"""

    # Project settings
    wake_word: str
    project_name: str

    # Sample settings
    target_samples: int = 1000  # Total samples to generate
    real_samples: int = 20  # Expected real recordings
    synthetic_samples: int = 980  # Generated with TTS

    # Training settings
    max_steps: int = 10000
    batch_size: int = 512
    learning_rate: float = 0.001
    early_stopping_patience: int = 5

    # Audio settings
    sample_rate: int = 16000
    clip_duration_ms: int = 3000  # Target clip duration (3 seconds)

    # Augmentation settings
    use_augmentation: bool = True
    augmentation_probability: float = 0.8
    noise_probability: float = 0.5
    rir_probability: float = 0.5

    # Voice settings
    voices: list = None  # List of voice model names to use
    min_voices: int = 2

    # Feature extraction settings
    feature_type: str = "melspectrogram"  # or "mfcc"
    n_mels: int = 40
    n_fft: int = 512
    hop_length: int = 160

    # Model settings
    model_type: str = "cnn"  # Model architecture
    embedding_size: int = 96

    # Paths (set dynamically)
    recordings_dir: Optional[str] = None
    clips_dir: Optional[str] = None
    features_dir: Optional[str] = None
    models_dir: Optional[str] = None

    def __post_init__(self):
        """Post initialization processing"""
        if self.voices is None:
            self.voices = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def save(self, path: Path):
        """
        Save configuration to YAML file

        Args:
            path: Path to save config file
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config to {path}")

    @classmethod
    def load(cls, path: Path) -> 'TrainingConfig':
        """
        Load configuration from YAML file

        Args:
            path: Path to config file

        Returns:
            TrainingConfig instance
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def create_default(
        cls,
        project_name: str,
        wake_word: str,
        project_path: Path
    ) -> 'TrainingConfig':
        """
        Create default configuration for project

        Args:
            project_name: Name of the project
            wake_word: Wake word or phrase
            project_path: Path to project directory

        Returns:
            TrainingConfig instance
        """
        config = cls(
            project_name=project_name,
            wake_word=wake_word,
            recordings_dir=str(project_path / 'recordings'),
            clips_dir=str(project_path / 'clips'),
            features_dir=str(project_path / 'features'),
            models_dir=str(project_path / 'models')
        )

        return config

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check required fields
        if not self.wake_word:
            issues.append("Wake word is required")

        if not self.project_name:
            issues.append("Project name is required")

        # Check sample counts
        if self.target_samples < 100:
            issues.append("Target samples should be at least 100")

        if self.real_samples < 10:
            issues.append("Real samples should be at least 10")

        # Check training settings
        if self.max_steps < 1000:
            issues.append("Max steps should be at least 1000")

        if self.batch_size < 32:
            issues.append("Batch size should be at least 32")

        if self.learning_rate <= 0 or self.learning_rate > 1:
            issues.append("Learning rate should be between 0 and 1")

        # Check audio settings
        if self.sample_rate not in [8000, 16000, 22050, 44100]:
            issues.append("Sample rate should be 8000, 16000, 22050, or 44100")

        if self.clip_duration_ms < 500 or self.clip_duration_ms > 3000:
            issues.append("Clip duration should be between 500 and 3000 ms")

        # Check augmentation settings
        if self.augmentation_probability < 0 or self.augmentation_probability > 1:
            issues.append("Augmentation probability should be between 0 and 1")

        # Check voice settings
        if len(self.voices) < self.min_voices:
            issues.append(f"At least {self.min_voices} voices required, found {len(self.voices)}")

        return len(issues) == 0, issues

    def update(self, **kwargs):
        """
        Update configuration parameters

        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")

    def get_synthetic_samples_per_voice(self) -> int:
        """
        Calculate synthetic samples per voice

        Returns:
            Number of samples to generate per voice
        """
        if not self.voices:
            return 0

        return self.synthetic_samples // len(self.voices)

    def estimate_training_time(self) -> float:
        """
        Estimate total training time in minutes

        Returns:
            Estimated time in minutes
        """
        # Rough estimates based on typical hardware
        clip_generation_time = (self.synthetic_samples / 100) * 2  # ~2 min per 100 samples
        augmentation_time = (self.target_samples / 100) * 1  # ~1 min per 100 samples
        feature_extraction_time = (self.target_samples / 100) * 0.5  # ~0.5 min per 100 samples
        training_time = (self.max_steps / 1000) * 3  # ~3 min per 1000 steps

        total = clip_generation_time + augmentation_time + feature_extraction_time + training_time

        return total

    def get_summary(self) -> Dict[str, Any]:
        """
        Get human-readable configuration summary

        Returns:
            Dictionary with summary information
        """
        return {
            'Project': self.project_name,
            'Wake Word': self.wake_word,
            'Total Samples': self.target_samples,
            'Real Samples': self.real_samples,
            'Synthetic Samples': self.synthetic_samples,
            'Voices': len(self.voices),
            'Max Training Steps': self.max_steps,
            'Estimated Time': f"{self.estimate_training_time():.0f} minutes",
            'Augmentation': 'Enabled' if self.use_augmentation else 'Disabled',
            'Sample Rate': f"{self.sample_rate} Hz",
            'Clip Duration': f"{self.clip_duration_ms} ms"
        }


class ConfigManager:
    """Manages training configurations for projects"""

    CONFIG_FILENAME = 'config.yaml'

    def __init__(self, project_path: Path):
        """
        Initialize config manager

        Args:
            project_path: Path to project directory
        """
        self.project_path = Path(project_path)
        self.config_path = self.project_path / self.CONFIG_FILENAME

    def exists(self) -> bool:
        """
        Check if config file exists

        Returns:
            True if config exists
        """
        return self.config_path.exists()

    def load(self) -> TrainingConfig:
        """
        Load configuration

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If config doesn't exist
        """
        if not self.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        return TrainingConfig.load(self.config_path)

    def save(self, config: TrainingConfig):
        """
        Save configuration

        Args:
            config: TrainingConfig to save
        """
        config.save(self.config_path)

    def create_default(
        self,
        project_name: str,
        wake_word: str
    ) -> TrainingConfig:
        """
        Create and save default configuration

        Args:
            project_name: Name of the project
            wake_word: Wake word or phrase

        Returns:
            Created TrainingConfig instance
        """
        config = TrainingConfig.create_default(
            project_name=project_name,
            wake_word=wake_word,
            project_path=self.project_path
        )

        self.save(config)
        return config

    def update(self, **kwargs):
        """
        Update and save configuration

        Args:
            **kwargs: Parameters to update
        """
        config = self.load()
        config.update(**kwargs)
        self.save(config)

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate current configuration

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        config = self.load()
        return config.validate()
