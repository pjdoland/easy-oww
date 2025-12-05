"""
Audio quality validation for wake word samples
"""
import numpy as np
from typing import Dict, List, Optional
from scipy.io import wavfile
from pathlib import Path
from easy_oww.utils.logger import get_logger

logger = get_logger()


class AudioValidator:
    """Validates audio quality for wake word training"""

    def __init__(
        self,
        sample_rate: int = 16000,
        min_duration: float = 0.5,
        max_duration: float = 3.0,
        min_level_db: float = -50.0,
        max_level_db: float = -10.0,
        silence_threshold_db: float = -60.0
    ):
        """
        Initialize audio validator

        Args:
            sample_rate: Expected sample rate in Hz
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            min_level_db: Minimum acceptable audio level in dB
            max_level_db: Maximum acceptable audio level in dB
            silence_threshold_db: Threshold for silence detection in dB
        """
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_level_db = min_level_db
        self.max_level_db = max_level_db
        self.silence_threshold_db = silence_threshold_db

    def validate_audio(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Validate audio quality

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate (uses default if None)

        Returns:
            Dictionary with validation results:
                - valid: bool
                - issues: List of issue descriptions
                - warnings: List of warning descriptions
                - metrics: Dictionary of audio metrics
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        issues = []
        warnings = []
        metrics = {}

        # Calculate duration
        duration = len(audio) / sample_rate
        metrics['duration'] = duration

        if duration < self.min_duration:
            issues.append(f"Duration too short ({duration:.2f}s < {self.min_duration}s)")
        elif duration > self.max_duration:
            issues.append(f"Duration too long ({duration:.2f}s > {self.max_duration}s)")

        # Calculate RMS level
        rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
        if rms > 0:
            level_db = 20 * np.log10(rms / 32768.0)
        else:
            level_db = -96.0

        metrics['level_db'] = level_db

        if level_db < self.min_level_db:
            issues.append(f"Audio too quiet ({level_db:.1f} dB < {self.min_level_db:.1f} dB)")
        elif level_db > self.max_level_db:
            issues.append(f"Audio too loud ({level_db:.1f} dB > {self.max_level_db:.1f} dB)")
        elif level_db < self.min_level_db + 10:
            warnings.append(f"Audio level is low ({level_db:.1f} dB)")

        # Check for clipping
        max_value = np.max(np.abs(audio))
        clipping_threshold = 32000  # Close to int16 max
        if max_value >= clipping_threshold:
            clip_percentage = np.sum(np.abs(audio) >= clipping_threshold) / len(audio) * 100
            issues.append(f"Audio clipping detected ({clip_percentage:.1f}% of samples)")
            metrics['clipping'] = True
            metrics['clip_percentage'] = clip_percentage
        else:
            metrics['clipping'] = False

        # Check for silence
        silence_frames = self._detect_silence(audio, sample_rate)
        silence_percentage = len(silence_frames) / len(audio) * 100
        metrics['silence_percentage'] = silence_percentage

        if silence_percentage > 80:
            issues.append(f"Too much silence ({silence_percentage:.1f}%)")
        elif silence_percentage > 60:
            warnings.append(f"High silence level ({silence_percentage:.1f}%)")

        # Check for DC offset
        dc_offset = np.mean(audio.astype(np.float32))
        metrics['dc_offset'] = dc_offset

        if abs(dc_offset) > 1000:
            warnings.append(f"DC offset detected ({dc_offset:.0f})")

        # Check signal-to-noise ratio estimate
        snr = self._estimate_snr(audio, silence_frames)
        metrics['snr_db'] = snr

        if snr < 10:
            issues.append(f"Low signal-to-noise ratio ({snr:.1f} dB)")
        elif snr < 20:
            warnings.append(f"Moderate signal-to-noise ratio ({snr:.1f} dB)")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'metrics': metrics
        }

    def validate_file(self, file_path: Path) -> Dict[str, any]:
        """
        Validate audio file

        Args:
            file_path: Path to audio file

        Returns:
            Validation results dictionary
        """
        try:
            sample_rate, audio = wavfile.read(str(file_path))

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Convert to int16 if needed
            if audio.dtype != np.int16:
                audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)

            return self.validate_audio(audio, sample_rate)

        except Exception as e:
            logger.error(f"Failed to validate file {file_path}: {e}")
            return {
                'valid': False,
                'issues': [f"Failed to read file: {e}"],
                'warnings': [],
                'metrics': {}
            }

    def _detect_silence(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Detect silence frames in audio

        Args:
            audio: Audio data
            sample_rate: Sample rate

        Returns:
            Boolean array indicating silence frames
        """
        # Calculate frame energy
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop

        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            rms = np.sqrt(np.mean(np.square(frame.astype(np.float32))))
            if rms > 0:
                db = 20 * np.log10(rms / 32768.0)
            else:
                db = -96.0
            energy.append(db)

        # Expand energy back to sample level
        silence_frames = np.zeros(len(audio), dtype=bool)
        for i, e in enumerate(energy):
            start = i * hop_length
            end = min(start + frame_length, len(audio))
            if e < self.silence_threshold_db:
                silence_frames[start:end] = True

        return silence_frames

    def _estimate_snr(
        self,
        audio: np.ndarray,
        silence_frames: np.ndarray
    ) -> float:
        """
        Estimate signal-to-noise ratio

        Args:
            audio: Audio data
            silence_frames: Boolean array indicating silence frames

        Returns:
            Estimated SNR in dB
        """
        if len(silence_frames) == 0 or not np.any(silence_frames):
            return 40.0  # Assume good SNR if no silence detected

        # Noise level from silence regions
        noise = audio[silence_frames].astype(np.float32)
        if len(noise) > 0:
            noise_rms = np.sqrt(np.mean(np.square(noise)))
        else:
            noise_rms = 1.0

        # Signal level from non-silence regions
        signal = audio[~silence_frames].astype(np.float32)
        if len(signal) > 0:
            signal_rms = np.sqrt(np.mean(np.square(signal)))
        else:
            signal_rms = 1.0

        # Calculate SNR
        if noise_rms > 0 and signal_rms > 0:
            snr = 20 * np.log10(signal_rms / noise_rms)
        else:
            snr = 0.0

        return max(0.0, min(60.0, snr))  # Clamp to reasonable range

    def batch_validate(
        self,
        audio_files: List[Path]
    ) -> Dict[str, any]:
        """
        Validate multiple audio files

        Args:
            audio_files: List of audio file paths

        Returns:
            Dictionary with batch validation results:
                - total: Total number of files
                - valid: Number of valid files
                - invalid: Number of invalid files
                - results: Dictionary mapping file paths to validation results
        """
        results = {}
        valid_count = 0
        invalid_count = 0

        for file_path in audio_files:
            result = self.validate_file(file_path)
            results[str(file_path)] = result

            if result['valid']:
                valid_count += 1
            else:
                invalid_count += 1

        return {
            'total': len(audio_files),
            'valid': valid_count,
            'invalid': invalid_count,
            'results': results
        }
