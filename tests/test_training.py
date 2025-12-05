"""
Tests for training functionality
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
from scipy.io import wavfile
from unittest.mock import Mock, patch

from easy_oww.training.config import TrainingConfig, ConfigManager
from easy_oww.training.clips import ClipGenerator
from easy_oww.training.augmentation import AudioAugmenter


class TestTrainingConfig:
    """Tests for TrainingConfig class"""

    def test_init(self):
        """Test config initialization"""
        config = TrainingConfig(
            project_name="test_project",
            wake_word="hey assistant"
        )
        assert config.project_name == "test_project"
        assert config.wake_word == "hey assistant"
        assert config.sample_rate == 16000
        assert config.target_samples == 1000

    def test_to_dict(self):
        """Test config to dictionary conversion"""
        config = TrainingConfig(
            project_name="test",
            wake_word="test word"
        )
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['project_name'] == "test"
        assert config_dict['wake_word'] == "test word"

    def test_save_and_load(self):
        """Test saving and loading config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'

            # Create and save config
            config = TrainingConfig(
                project_name="test",
                wake_word="test word",
                max_steps=5000
            )
            config.save(config_path)

            # Load config
            loaded = TrainingConfig.load(config_path)
            assert loaded.project_name == "test"
            assert loaded.wake_word == "test word"
            assert loaded.max_steps == 5000

    def test_create_default(self):
        """Test default config creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig.create_default(
                project_name="test",
                wake_word="hey test",
                project_path=Path(tmpdir)
            )
            assert config.project_name == "test"
            assert config.wake_word == "hey test"
            assert config.recordings_dir is not None

    def test_validate_valid(self):
        """Test validation of valid config"""
        config = TrainingConfig(
            project_name="test",
            wake_word="test word",
            voices=['voice1', 'voice2']
        )
        is_valid, issues = config.validate()
        assert is_valid
        assert len(issues) == 0

    def test_validate_invalid(self):
        """Test validation of invalid config"""
        config = TrainingConfig(
            project_name="",  # Invalid: empty
            wake_word="test",
            target_samples=50,  # Invalid: too few
            voices=[]  # Invalid: not enough voices
        )
        is_valid, issues = config.validate()
        assert not is_valid
        assert len(issues) > 0

    def test_estimate_training_time(self):
        """Test training time estimation"""
        config = TrainingConfig(
            project_name="test",
            wake_word="test",
            target_samples=500,
            max_steps=5000
        )
        time_estimate = config.estimate_training_time()
        assert time_estimate > 0
        assert isinstance(time_estimate, float)


class TestConfigManager:
    """Tests for ConfigManager class"""

    def test_init(self):
        """Test config manager initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(Path(tmpdir))
            assert manager.project_path == Path(tmpdir)
            assert not manager.exists()

    def test_create_and_load(self):
        """Test creating and loading config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(Path(tmpdir))

            # Create default config
            config = manager.create_default("test", "test word")
            assert manager.exists()

            # Load config
            loaded = manager.load()
            assert loaded.project_name == "test"
            assert loaded.wake_word == "test word"

    def test_update(self):
        """Test updating config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(Path(tmpdir))
            manager.create_default("test", "test word")

            # Update config
            manager.update(max_steps=15000, batch_size=256)

            # Load and verify
            config = manager.load()
            assert config.max_steps == 15000
            assert config.batch_size == 256


class TestClipGenerator:
    """Tests for ClipGenerator class"""

    def test_init(self):
        """Test clip generator initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ClipGenerator(
                recordings_dir=Path(tmpdir) / 'recordings',
                clips_dir=Path(tmpdir) / 'clips'
            )
            assert generator.recordings_dir.name == 'recordings'
            assert generator.clips_dir.exists()
            assert generator.positive_dir.exists()
            assert generator.negative_dir.exists()

    def test_prepare_clip(self):
        """Test clip preparation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ClipGenerator(
                recordings_dir=Path(tmpdir) / 'recordings',
                clips_dir=Path(tmpdir) / 'clips',
                target_duration_ms=1000
            )

            # Create test audio (too long)
            audio = np.random.randint(-1000, 1000, 32000, dtype=np.int16)

            # Prepare clip
            prepared = generator._prepare_clip(audio)

            # Check length
            assert len(prepared) == generator.target_samples
            assert prepared.dtype == np.int16

    def test_get_clip_counts(self):
        """Test getting clip counts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ClipGenerator(
                recordings_dir=Path(tmpdir) / 'recordings',
                clips_dir=Path(tmpdir) / 'clips'
            )

            # Create some dummy clips
            for i in range(5):
                clip_path = generator.positive_dir / f"clip_{i}.wav"
                audio = np.zeros(16000, dtype=np.int16)
                wavfile.write(str(clip_path), 16000, audio)

            counts = generator.get_clip_counts()
            assert counts['positive'] == 5
            assert counts['negative'] == 0
            assert counts['total'] == 5


class TestAudioAugmenter:
    """Tests for AudioAugmenter class"""

    def test_init(self):
        """Test augmenter initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            augmenter = AudioAugmenter(
                rir_dir=Path(tmpdir) / 'rir',
                noise_dir=Path(tmpdir) / 'noise'
            )
            assert augmenter.sample_rate == 16000
            assert len(augmenter.rir_files) == 0
            assert len(augmenter.noise_files) == 0

    def test_apply_volume_change(self):
        """Test volume change"""
        augmenter = AudioAugmenter()

        # Create test audio
        audio = np.ones(1000, dtype=np.int16) * 1000

        # Apply volume change
        augmented = augmenter.apply_volume_change(audio, gain_db=6)

        # Check that volume increased
        assert np.mean(np.abs(augmented)) > np.mean(np.abs(audio))
        assert augmented.dtype == np.int16

    def test_apply_pitch_shift(self):
        """Test pitch shifting"""
        augmenter = AudioAugmenter()

        # Create test audio
        audio = np.random.randint(-1000, 1000, 1000, dtype=np.int16)

        # Apply pitch shift
        augmented = augmenter.apply_pitch_shift(audio, semitones=2)

        # Check that audio changed
        assert len(augmented) == len(audio)
        assert augmented.dtype == np.int16

    def test_apply_time_stretch(self):
        """Test time stretching"""
        augmenter = AudioAugmenter()

        # Create test audio
        audio = np.random.randint(-1000, 1000, 1000, dtype=np.int16)

        # Apply time stretch
        augmented = augmenter.apply_time_stretch(audio, rate=1.1)

        # Check that audio changed but length preserved
        assert len(augmented) == len(audio)
        assert augmented.dtype == np.int16

    def test_augment_no_files(self):
        """Test augmentation without RIR/noise files"""
        augmenter = AudioAugmenter()

        # Create test audio
        audio = np.random.randint(-1000, 1000, 1000, dtype=np.int16)

        # Apply augmentation
        augmented = augmenter.augment(audio)

        # Should still work (just without RIR/noise)
        assert len(augmented) == len(audio)
        assert augmented.dtype == np.int16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
