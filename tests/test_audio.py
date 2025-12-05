"""
Tests for audio recording functionality
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
from easy_oww.audio.recorder import AudioRecorder
from easy_oww.audio.validator import AudioValidator


class TestAudioRecorder:
    """Tests for AudioRecorder class"""

    def test_init(self):
        """Test recorder initialization"""
        recorder = AudioRecorder()
        assert recorder.sample_rate == 16000
        assert recorder.channels == 1
        assert recorder.dtype == 'int16'
        assert not recorder.recording

    def test_list_devices(self):
        """Test device listing"""
        recorder = AudioRecorder()
        devices = recorder.list_devices()
        assert isinstance(devices, list)

    def test_get_default_device(self):
        """Test getting default device"""
        recorder = AudioRecorder()
        device = recorder.get_default_device()
        # Device may be None if no input devices available
        if device is not None:
            assert 'index' in device
            assert 'name' in device

    def test_get_audio_level(self):
        """Test audio level calculation"""
        recorder = AudioRecorder()

        # Test silence
        silence = np.zeros(1000, dtype=np.int16)
        level = recorder.get_audio_level(silence)
        assert level < -90  # Very quiet

        # Test loud signal
        loud = np.ones(1000, dtype=np.int16) * 10000
        level = recorder.get_audio_level(loud)
        assert level > -40  # Louder

    def test_save_wav(self):
        """Test WAV file saving"""
        recorder = AudioRecorder()

        # Create test audio
        audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.wav'
            recorder.save_wav(audio, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0


class TestAudioValidator:
    """Tests for AudioValidator class"""

    def test_init(self):
        """Test validator initialization"""
        validator = AudioValidator()
        assert validator.sample_rate == 16000
        assert validator.min_duration == 0.5
        assert validator.max_duration == 3.0

    def test_validate_audio_duration(self):
        """Test duration validation"""
        validator = AudioValidator()

        # Too short
        short_audio = np.zeros(4000, dtype=np.int16)  # 0.25s at 16kHz
        result = validator.validate_audio(short_audio)
        assert not result['valid']
        assert any('short' in issue.lower() for issue in result['issues'])

        # Good duration
        good_audio = np.zeros(16000, dtype=np.int16)  # 1s at 16kHz
        result = validator.validate_audio(good_audio)
        # May still be invalid due to silence, but duration should be OK
        assert not any('short' in issue.lower() for issue in result['issues'])

    def test_validate_audio_level(self):
        """Test audio level validation"""
        validator = AudioValidator()

        # Too quiet
        quiet_audio = np.random.randint(-10, 10, 16000, dtype=np.int16)
        result = validator.validate_audio(quiet_audio)
        assert not result['valid']
        assert any('quiet' in issue.lower() or 'silence' in issue.lower() for issue in result['issues'])

        # Good level
        good_audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
        result = validator.validate_audio(good_audio)
        # Check level metric exists
        assert 'level_db' in result['metrics']

    def test_validate_clipping(self):
        """Test clipping detection"""
        validator = AudioValidator()

        # Clipped audio
        clipped = np.ones(16000, dtype=np.int16) * 32000
        result = validator.validate_audio(clipped)
        assert not result['valid']
        assert any('clip' in issue.lower() for issue in result['issues'])
        assert result['metrics']['clipping']

    def test_batch_validate(self):
        """Test batch validation"""
        validator = AudioValidator()

        # Create test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create some test WAV files
            from scipy.io import wavfile

            files = []
            for i in range(3):
                file_path = tmpdir / f'test_{i}.wav'
                audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
                wavfile.write(str(file_path), 16000, audio)
                files.append(file_path)

            # Validate batch
            result = validator.batch_validate(files)
            assert result['total'] == 3
            assert 'valid' in result
            assert 'invalid' in result
            assert len(result['results']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
