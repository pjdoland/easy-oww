"""
Tests for TTS functionality
"""
import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock

from easy_oww.tts.piper import PiperTTS
from easy_oww.tts.voices import VoiceDownloader
from easy_oww.tts.generator import SampleGenerator


class TestPiperTTS:
    """Tests for PiperTTS class"""

    def test_init(self):
        """Test PiperTTS initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            piper = PiperTTS(Path(tmpdir))
            assert piper.install_dir == Path(tmpdir)
            assert piper.voices_dir.exists()

    def test_get_binary_path(self):
        """Test binary path detection"""
        with tempfile.TemporaryDirectory() as tmpdir:
            piper = PiperTTS(Path(tmpdir))
            binary_path = piper._get_binary_path()
            assert binary_path.parent == Path(tmpdir)
            assert 'piper' in binary_path.name.lower()

    def test_is_installed_false(self):
        """Test installation check when not installed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            piper = PiperTTS(Path(tmpdir))
            assert not piper.is_installed()

    def test_get_download_url(self):
        """Test download URL generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            piper = PiperTTS(Path(tmpdir))
            url, archive = piper.get_download_url()
            assert url.startswith('https://')
            assert 'piper' in archive.lower()
            assert any(ext in archive for ext in ['.tar.gz', '.zip'])

    def test_list_installed_voices_empty(self):
        """Test listing voices when none installed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            piper = PiperTTS(Path(tmpdir))
            voices = piper.list_installed_voices()
            assert isinstance(voices, list)
            assert len(voices) == 0

    def test_get_voice_not_found(self):
        """Test getting voice that doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            piper = PiperTTS(Path(tmpdir))
            voice = piper.get_voice('nonexistent')
            assert voice is None


class TestVoiceDownloader:
    """Tests for VoiceDownloader class"""

    def test_init(self):
        """Test VoiceDownloader initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = VoiceDownloader(Path(tmpdir))
            assert downloader.voices_dir == Path(tmpdir)
            assert downloader.voices_dir.exists()

    def test_get_voice_url(self):
        """Test voice URL generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = VoiceDownloader(Path(tmpdir))
            model_url, config_url = downloader.get_voice_url('en_US-lessac-medium')

            assert model_url.endswith('.onnx')
            assert config_url.endswith('.onnx.json')
            assert 'en_US-lessac-medium' in model_url

    def test_list_available_voices(self):
        """Test listing available voices"""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = VoiceDownloader(Path(tmpdir))

            # Test specific language
            voices = downloader.list_available_voices('en_US')
            assert isinstance(voices, list)
            assert len(voices) > 0
            assert all('name' in v for v in voices)

            # Test all languages
            all_voices = downloader.list_available_voices()
            assert len(all_voices) >= len(voices)

    def test_list_installed_voices_empty(self):
        """Test listing installed voices when none exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = VoiceDownloader(Path(tmpdir))
            installed = downloader.list_installed_voices()
            assert isinstance(installed, list)
            assert len(installed) == 0

    def test_is_voice_installed_false(self):
        """Test checking for voice that isn't installed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = VoiceDownloader(Path(tmpdir))
            assert not downloader.is_voice_installed('en_US-lessac-medium')

    def test_get_total_size(self):
        """Test calculating total download size"""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = VoiceDownloader(Path(tmpdir))
            size = downloader.get_total_size(['en_US-lessac-medium'])
            assert size > 0
            assert isinstance(size, int)

    def test_verify_voice_missing(self):
        """Test verifying missing voice"""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = VoiceDownloader(Path(tmpdir))
            success, message = downloader.verify_voice('nonexistent')
            assert not success
            assert 'missing' in message.lower()


class TestSampleGenerator:
    """Tests for SampleGenerator class"""

    def test_init(self):
        """Test SampleGenerator initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_piper = Mock()
            generator = SampleGenerator(mock_piper, Path(tmpdir))
            assert generator.piper == mock_piper
            assert generator.output_dir == Path(tmpdir)
            assert generator.sample_rate == 16000

    def test_generate_variations(self):
        """Test generating text variations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_piper = Mock()
            generator = SampleGenerator(mock_piper, Path(tmpdir))

            variations = generator.generate_variations("hey assistant", count=10)
            assert len(variations) == 10
            assert all(isinstance(v, str) for v in variations)
            assert any('hey assistant' in v for v in variations)

    def test_estimate_generation_time(self):
        """Test generation time estimation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_piper = Mock()
            generator = SampleGenerator(mock_piper, Path(tmpdir))

            time_estimate = generator.estimate_generation_time(100, voices=2)
            assert time_estimate > 0
            assert isinstance(time_estimate, float)

    @patch('easy_oww.tts.generator.SampleGenerator.generate_samples')
    def test_generate_multi_voice(self, mock_generate):
        """Test multi-voice generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_piper = Mock()
            generator = SampleGenerator(mock_piper, Path(tmpdir))

            # Mock generate_samples to return fake paths
            mock_generate.return_value = [
                Path(tmpdir) / 'sample_0000.wav',
                Path(tmpdir) / 'sample_0001.wav'
            ]

            voice_models = [
                Path(tmpdir) / 'voice1.onnx',
                Path(tmpdir) / 'voice2.onnx'
            ]

            # Create dummy voice files
            for vm in voice_models:
                vm.touch()

            results = generator.generate_multi_voice(
                "test word",
                voice_models,
                samples_per_voice=2
            )

            assert isinstance(results, dict)
            assert len(results) == 2

    def test_validate_generated_samples_empty(self):
        """Test validating empty sample list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_piper = Mock()
            generator = SampleGenerator(mock_piper, Path(tmpdir))

            results = generator.validate_generated_samples([])
            assert results['total'] == 0
            assert results['valid'] == 0
            assert results['invalid'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
