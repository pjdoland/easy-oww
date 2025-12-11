"""
Audio augmentation for robust wake word training
"""
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from scipy.io import wavfile
from scipy import signal
from rich.progress import Progress
from rich.console import Console

from easy_oww.utils.logger import get_logger

logger = get_logger()
console = Console()


class AudioAugmenter:
    """Applies audio augmentations for training robustness"""

    def __init__(
        self,
        rir_dir: Optional[Path] = None,
        noise_dir: Optional[Path] = None,
        sample_rate: int = 16000
    ):
        """
        Initialize audio augmenter

        Args:
            rir_dir: Directory with room impulse response files
            noise_dir: Directory with noise audio files
            sample_rate: Audio sample rate
        """
        self.rir_dir = Path(rir_dir) if rir_dir else None
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.sample_rate = sample_rate

        # Load RIR files
        self.rir_files = []
        if self.rir_dir and self.rir_dir.exists():
            self.rir_files = list(self.rir_dir.rglob('*.wav'))
            if self.rir_files:
                logger.info(f"Loaded {len(self.rir_files)} RIR files from {self.rir_dir}")
            else:
                logger.warning(f"RIR directory exists but no .wav files found in {self.rir_dir}")
        else:
            logger.info("No RIR directory provided - RIR augmentation will be skipped")

        # Load noise files
        self.noise_files = []
        if self.noise_dir and self.noise_dir.exists():
            for ext in ['*.wav', '*.mp3', '*.flac']:
                self.noise_files.extend(list(self.noise_dir.rglob(ext)))
            if self.noise_files:
                logger.info(f"Loaded {len(self.noise_files)} noise files from {self.noise_dir}")
            else:
                logger.warning(f"Noise directory exists but no audio files found in {self.noise_dir}")
        else:
            logger.info("No noise directory provided - noise augmentation will be skipped")

        # Cache for loaded noise segments
        self.noise_cache = []
        self.noise_cache_size = 100

    def apply_rir(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random room impulse response

        Args:
            audio: Input audio

        Returns:
            Audio with RIR applied
        """
        if not self.rir_files:
            logger.debug("No RIR files available - skipping RIR augmentation")
            return audio

        # Select random RIR
        rir_path = np.random.choice(self.rir_files)

        try:
            # Load RIR
            rir_sr, rir = wavfile.read(str(rir_path))

            # Convert to mono if needed
            if len(rir.shape) > 1:
                rir = rir.mean(axis=1)

            # Resample RIR if needed
            if rir_sr != self.sample_rate:
                rir = self._resample(rir, rir_sr, self.sample_rate)

            # Normalize RIR
            rir = rir.astype(np.float32)
            rir = rir / np.max(np.abs(rir))

            # Apply convolution
            audio_float = audio.astype(np.float32)
            augmented = signal.convolve(audio_float, rir, mode='same')

            # Normalize to prevent clipping
            if np.max(np.abs(augmented)) > 0:
                augmented = augmented / np.max(np.abs(augmented)) * 0.9

            # Convert back to int16
            augmented = (augmented * 32767).astype(np.int16)

            return augmented

        except Exception as e:
            logger.warning(f"Failed to apply RIR from {rir_path.name}: {e}")
            return audio

    def add_noise(
        self,
        audio: np.ndarray,
        snr_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Add background noise to audio

        Args:
            audio: Input audio
            snr_db: Signal-to-noise ratio in dB (if None, random between 5-20)

        Returns:
            Audio with noise added
        """
        if not self.noise_files:
            logger.debug("No noise files available - skipping noise augmentation")
            return audio

        # Random SNR if not specified
        if snr_db is None:
            snr_db = np.random.uniform(5, 20)

        try:
            # Get noise segment
            noise = self._get_noise_segment(len(audio))

            if noise is None:
                return audio

            # Calculate noise scaling factor
            audio_power = np.mean(audio.astype(np.float32) ** 2)
            noise_power = np.mean(noise.astype(np.float32) ** 2)

            if noise_power == 0:
                return audio

            # Calculate scale factor for target SNR
            snr_linear = 10 ** (snr_db / 10)
            scale = np.sqrt(audio_power / (snr_linear * noise_power))

            # Add noise
            audio_float = audio.astype(np.float32)
            noise_float = noise.astype(np.float32) * scale
            augmented = audio_float + noise_float

            # Normalize to prevent clipping
            if np.max(np.abs(augmented)) > 32767:
                augmented = augmented / np.max(np.abs(augmented)) * 32767 * 0.9

            augmented = augmented.astype(np.int16)

            return augmented

        except Exception as e:
            logger.warning(f"Failed to add noise: {e}")
            return audio

    def apply_pitch_shift(
        self,
        audio: np.ndarray,
        semitones: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply pitch shifting

        Args:
            audio: Input audio
            semitones: Semitones to shift (if None, random between -2 and 2)

        Returns:
            Pitch-shifted audio
        """
        if semitones is None:
            semitones = np.random.uniform(-2, 2)

        if abs(semitones) < 0.1:
            return audio

        try:
            # Calculate stretch factor
            factor = 2 ** (semitones / 12)

            # Resample for pitch shift
            num_samples = int(len(audio) / factor)
            shifted = signal.resample(audio, num_samples)

            # Trim or pad to original length
            if len(shifted) > len(audio):
                start = (len(shifted) - len(audio)) // 2
                shifted = shifted[start:start + len(audio)]
            elif len(shifted) < len(audio):
                pad_length = len(audio) - len(shifted)
                shifted = np.pad(shifted, (0, pad_length), mode='constant')

            return shifted.astype(np.int16)

        except Exception as e:
            logger.warning(f"Failed to apply pitch shift: {e}")
            return audio

    def apply_time_stretch(
        self,
        audio: np.ndarray,
        rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply time stretching (speed change)

        Args:
            audio: Input audio
            rate: Stretch rate (if None, random between 0.9 and 1.1)

        Returns:
            Time-stretched audio
        """
        if rate is None:
            rate = np.random.uniform(0.9, 1.1)

        if abs(rate - 1.0) < 0.01:
            return audio

        try:
            # Resample for time stretch
            num_samples = int(len(audio) * rate)
            stretched = signal.resample(audio, num_samples)

            # Trim or pad to original length
            if len(stretched) > len(audio):
                start = (len(stretched) - len(audio)) // 2
                stretched = stretched[start:start + len(audio)]
            elif len(stretched) < len(audio):
                pad_length = len(audio) - len(stretched)
                stretched = np.pad(stretched, (0, pad_length), mode='constant')

            return stretched.astype(np.int16)

        except Exception as e:
            logger.warning(f"Failed to apply time stretch: {e}")
            return audio

    def apply_volume_change(
        self,
        audio: np.ndarray,
        gain_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply volume change

        Args:
            audio: Input audio
            gain_db: Gain in dB (if None, random between -6 and 6)

        Returns:
            Volume-adjusted audio
        """
        if gain_db is None:
            gain_db = np.random.uniform(-6, 6)

        if abs(gain_db) < 0.1:
            return audio

        # Convert dB to linear scale
        gain_linear = 10 ** (gain_db / 20)

        # Apply gain
        audio_float = audio.astype(np.float32) * gain_linear

        # Clip to prevent overflow
        audio_float = np.clip(audio_float, -32767, 32767)

        return audio_float.astype(np.int16)

    def augment(
        self,
        audio: np.ndarray,
        rir_prob: float = 0.5,
        noise_prob: float = 0.5,
        pitch_prob: float = 0.3,
        stretch_prob: float = 0.3,
        volume_prob: float = 0.5
    ) -> np.ndarray:
        """
        Apply random augmentations

        Args:
            audio: Input audio
            rir_prob: Probability of applying RIR
            noise_prob: Probability of adding noise
            pitch_prob: Probability of pitch shifting
            stretch_prob: Probability of time stretching
            volume_prob: Probability of volume change

        Returns:
            Augmented audio
        """
        augmented = audio.copy()

        # Apply RIR
        if np.random.random() < rir_prob:
            augmented = self.apply_rir(augmented)

        # Add noise
        if np.random.random() < noise_prob:
            augmented = self.add_noise(augmented)

        # Pitch shift
        if np.random.random() < pitch_prob:
            augmented = self.apply_pitch_shift(augmented)

        # Time stretch
        if np.random.random() < stretch_prob:
            augmented = self.apply_time_stretch(augmented)

        # Volume change
        if np.random.random() < volume_prob:
            augmented = self.apply_volume_change(augmented)

        return augmented

    def augment_clips(
        self,
        input_clips: List[Path],
        output_dir: Path,
        augmentations_per_clip: int = 3,
        **augment_kwargs
    ) -> List[Path]:
        """
        Augment a list of clips

        Args:
            input_clips: List of input clip paths
            output_dir: Output directory for augmented clips
            augmentations_per_clip: Number of augmented versions per clip
            **augment_kwargs: Arguments for augment()

        Returns:
            List of augmented clip paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[bold]Augmenting {len(input_clips)} clips...[/bold]")

        augmented_clips = []
        total_augmentations = len(input_clips) * augmentations_per_clip

        with Progress(console=console) as progress:
            task = progress.add_task("Augmenting...", total=total_augmentations)

            for clip_idx, clip_path in enumerate(input_clips):
                try:
                    # Load clip
                    sample_rate, audio = wavfile.read(str(clip_path))

                    # Generate augmented versions
                    for aug_idx in range(augmentations_per_clip):
                        # Apply augmentations
                        augmented = self.augment(audio, **augment_kwargs)

                        # Save
                        output_path = output_dir / f"{clip_path.stem}_aug{aug_idx}.wav"
                        wavfile.write(str(output_path), sample_rate, augmented)

                        augmented_clips.append(output_path)
                        progress.update(task, advance=1)

                except Exception as e:
                    logger.warning(f"Failed to augment {clip_path.name}: {e}")
                    progress.update(task, advance=augmentations_per_clip)

        console.print(f"[green]✓[/green] Generated {len(augmented_clips)} augmented clips")
        return augmented_clips

    def _get_noise_segment(self, length: int) -> Optional[np.ndarray]:
        """
        Get random noise segment of specified length

        Args:
            length: Desired length in samples

        Returns:
            Noise segment or None if unavailable
        """
        if not self.noise_files:
            return None

        # Try to get from cache first
        if len(self.noise_cache) < self.noise_cache_size:
            # Load new noise segment
            noise_path = np.random.choice(self.noise_files)

            try:
                noise_sr, noise = wavfile.read(str(noise_path))

                # Convert to mono
                if len(noise.shape) > 1:
                    noise = noise.mean(axis=1)

                # Resample if needed
                if noise_sr != self.sample_rate:
                    noise = self._resample(noise, noise_sr, self.sample_rate)

                # Add to cache
                self.noise_cache.append(noise)

            except Exception as e:
                logger.warning(f"Failed to load noise from {noise_path.name}: {e}")
                return None

        # Get random noise from cache
        if not self.noise_cache:
            return None

        # Use random.choice instead of np.random.choice for list of arrays
        import random
        noise = random.choice(self.noise_cache)

        # Extract random segment
        if len(noise) >= length:
            start = np.random.randint(0, len(noise) - length + 1)
            segment = noise[start:start + length]
        else:
            # Repeat noise if too short
            repeats = (length // len(noise)) + 1
            repeated = np.tile(noise, repeats)
            segment = repeated[:length]

        return segment

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio

        Args:
            audio: Audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        ratio = target_sr / orig_sr
        num_samples = int(len(audio) * ratio)
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(audio.dtype)
