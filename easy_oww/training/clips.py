"""
Clip generation and preparation for training
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from scipy.io import wavfile
import shutil
from rich.progress import Progress
from rich.console import Console

from easy_oww.tts import PiperTTS, SampleGenerator
from easy_oww.utils.logger import get_logger

logger = get_logger()
console = Console()


class ClipGenerator:
    """Generates and prepares audio clips for training"""

    def __init__(
        self,
        recordings_dir: Path,
        clips_dir: Path,
        sample_rate: int = 16000,
        target_duration_ms: int = 1000
    ):
        """
        Initialize clip generator

        Args:
            recordings_dir: Directory with real recordings
            clips_dir: Directory to save processed clips
            sample_rate: Target sample rate
            target_duration_ms: Target clip duration in milliseconds
        """
        self.recordings_dir = Path(recordings_dir)
        self.clips_dir = Path(clips_dir)
        self.sample_rate = sample_rate
        self.target_duration_ms = target_duration_ms
        self.target_samples = int((target_duration_ms / 1000.0) * sample_rate)

        # Create subdirectories
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.positive_dir = self.clips_dir / 'positive'
        self.negative_dir = self.clips_dir / 'negative'
        self.positive_dir.mkdir(exist_ok=True)
        self.negative_dir.mkdir(exist_ok=True)

    def process_real_recordings(self) -> List[Path]:
        """
        Process real recordings into training clips

        Returns:
            List of processed clip paths
        """
        if not self.recordings_dir.exists():
            logger.warning(f"Recordings directory not found: {self.recordings_dir}")
            return []

        recording_files = list(self.recordings_dir.glob('*.wav'))

        if not recording_files:
            logger.warning("No recordings found")
            return []

        console.print(f"\n[bold]Processing {len(recording_files)} real recordings...[/bold]")

        processed_clips = []

        with Progress(console=console) as progress:
            task = progress.add_task("Processing recordings...", total=len(recording_files))

            for i, recording_path in enumerate(recording_files):
                try:
                    # Load audio
                    sample_rate, audio = wavfile.read(str(recording_path))

                    # Convert to mono if needed
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1).astype(audio.dtype)

                    # Resample if needed
                    if sample_rate != self.sample_rate:
                        audio = self._resample(audio, sample_rate, self.sample_rate)

                    # Normalize and trim/pad to target duration
                    audio = self._prepare_clip(audio)

                    # Save to positive clips
                    output_path = self.positive_dir / f"real_{i:04d}.wav"
                    wavfile.write(str(output_path), self.sample_rate, audio)

                    processed_clips.append(output_path)

                except Exception as e:
                    logger.warning(f"Failed to process {recording_path.name}: {e}")

                progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Processed {len(processed_clips)} recordings")
        return processed_clips

    def generate_synthetic_clips(
        self,
        wake_word: str,
        voice_models: List[Path],
        piper: PiperTTS,
        count: int = 500
    ) -> List[Path]:
        """
        Generate synthetic clips using TTS

        Args:
            wake_word: Wake word text
            voice_models: List of voice model paths
            piper: PiperTTS instance
            count: Number of clips to generate

        Returns:
            List of generated clip paths
        """
        console.print(f"\n[bold]Generating {count} synthetic clips...[/bold]")

        # Create temporary directory for raw TTS output
        temp_dir = self.clips_dir / 'temp_tts'
        temp_dir.mkdir(exist_ok=True)

        try:
            # Generate samples
            generator = SampleGenerator(
                piper=piper,
                output_dir=temp_dir,
                sample_rate=self.sample_rate
            )

            raw_samples = generator.generate_mixed_samples(
                wake_word=wake_word,
                voice_models=voice_models,
                total_count=count,
                prefix="synthetic"
            )

            # Process generated samples into clips
            console.print("\n[bold]Processing synthetic samples...[/bold]")
            processed_clips = []

            with Progress(console=console) as progress:
                task = progress.add_task("Processing...", total=len(raw_samples))

                for i, sample_path in enumerate(raw_samples):
                    try:
                        # Load audio
                        sample_rate, audio = wavfile.read(str(sample_path))

                        # Prepare clip (normalize, trim/pad)
                        audio = self._prepare_clip(audio)

                        # Save to positive clips
                        output_path = self.positive_dir / f"synth_{i:04d}.wav"
                        wavfile.write(str(output_path), self.sample_rate, audio)

                        processed_clips.append(output_path)

                    except Exception as e:
                        logger.warning(f"Failed to process {sample_path.name}: {e}")

                    progress.update(task, advance=1)

            console.print(f"[green]✓[/green] Generated {len(processed_clips)} synthetic clips")
            return processed_clips

        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def generate_negative_clips(
        self,
        negative_audio_dir: Path,
        count: int = 1000
    ) -> List[Path]:
        """
        Generate negative (non-wake-word) clips from audio datasets

        Args:
            negative_audio_dir: Directory with negative audio samples
            count: Number of negative clips to generate

        Returns:
            List of negative clip paths
        """
        console.print(f"\n[bold]Generating {count} negative clips...[/bold]")

        if not negative_audio_dir.exists():
            logger.warning(f"Negative audio directory not found: {negative_audio_dir}")
            return []

        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(list(negative_audio_dir.rglob(ext)))

        if not audio_files:
            logger.warning("No negative audio files found")
            return []

        logger.info(f"Found {len(audio_files)} negative audio files")

        negative_clips = []

        with Progress(console=console) as progress:
            task = progress.add_task("Generating negatives...", total=count)

            attempts = 0
            max_attempts = count * 3  # Try up to 3x the target count

            while len(negative_clips) < count and attempts < max_attempts:
                attempts += 1

                # Randomly select an audio file
                audio_file = np.random.choice(audio_files)

                try:
                    # Load audio
                    sample_rate, audio = wavfile.read(str(audio_file))

                    # Convert to mono
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1).astype(audio.dtype)

                    # Resample if needed
                    if sample_rate != self.sample_rate:
                        audio = self._resample(audio, sample_rate, self.sample_rate)

                    # Extract random segment
                    if len(audio) >= self.target_samples:
                        start = np.random.randint(0, len(audio) - self.target_samples)
                        segment = audio[start:start + self.target_samples]

                        # Prepare clip
                        segment = self._prepare_clip(segment)

                        # Save
                        output_path = self.negative_dir / f"negative_{len(negative_clips):04d}.wav"
                        wavfile.write(str(output_path), self.sample_rate, segment)

                        negative_clips.append(output_path)
                        progress.update(task, advance=1)

                except Exception as e:
                    logger.debug(f"Failed to process negative sample from {audio_file.name}: {e}")
                    continue

        console.print(f"[green]✓[/green] Generated {len(negative_clips)} negative clips")
        return negative_clips

    def _prepare_clip(self, audio: np.ndarray) -> np.ndarray:
        """
        Prepare audio clip (normalize, trim/pad to target duration)

        Args:
            audio: Audio data

        Returns:
            Prepared audio clip
        """
        # Ensure correct dtype
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio.astype(np.float32)
            audio = audio / np.max(np.abs(audio)) * 0.9  # Scale to 90% of max
            audio = (audio * 32767).astype(np.int16)

        # Trim or pad to target duration
        if len(audio) > self.target_samples:
            # Trim: center crop
            start = (len(audio) - self.target_samples) // 2
            audio = audio[start:start + self.target_samples]
        elif len(audio) < self.target_samples:
            # Pad with silence
            pad_length = self.target_samples - len(audio)
            pad_start = pad_length // 2
            pad_end = pad_length - pad_start
            audio = np.pad(audio, (pad_start, pad_end), mode='constant', constant_values=0)

        return audio

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate

        Args:
            audio: Audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        from scipy import signal

        # Calculate resampling ratio
        ratio = target_sr / orig_sr

        # Use scipy's resample
        num_samples = int(len(audio) * ratio)
        resampled = signal.resample(audio, num_samples)

        return resampled.astype(audio.dtype)

    def get_clip_counts(self) -> Dict[str, int]:
        """
        Get counts of generated clips

        Returns:
            Dictionary with clip counts
        """
        positive_clips = list(self.positive_dir.glob('*.wav'))
        negative_clips = list(self.negative_dir.glob('*.wav'))

        return {
            'positive': len(positive_clips),
            'negative': len(negative_clips),
            'total': len(positive_clips) + len(negative_clips)
        }

    def cleanup(self):
        """Clean up temporary files and directories"""
        temp_dir = self.clips_dir / 'temp_tts'
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def verify_clips(self) -> Dict[str, any]:
        """
        Verify generated clips

        Returns:
            Verification results
        """
        results = {
            'positive': {'valid': 0, 'invalid': 0, 'issues': []},
            'negative': {'valid': 0, 'invalid': 0, 'issues': []}
        }

        for clip_type, clip_dir in [('positive', self.positive_dir), ('negative', self.negative_dir)]:
            clips = list(clip_dir.glob('*.wav'))

            for clip_path in clips:
                try:
                    sample_rate, audio = wavfile.read(str(clip_path))

                    # Check sample rate
                    if sample_rate != self.sample_rate:
                        results[clip_type]['invalid'] += 1
                        results[clip_type]['issues'].append(
                            f"{clip_path.name}: Wrong sample rate ({sample_rate} Hz)"
                        )
                        continue

                    # Check duration
                    if len(audio) != self.target_samples:
                        results[clip_type]['invalid'] += 1
                        results[clip_type]['issues'].append(
                            f"{clip_path.name}: Wrong length ({len(audio)} samples)"
                        )
                        continue

                    # Check for silence
                    if np.max(np.abs(audio)) < 100:
                        results[clip_type]['invalid'] += 1
                        results[clip_type]['issues'].append(
                            f"{clip_path.name}: Too quiet or silent"
                        )
                        continue

                    results[clip_type]['valid'] += 1

                except Exception as e:
                    results[clip_type]['invalid'] += 1
                    results[clip_type]['issues'].append(f"{clip_path.name}: {e}")

        return results
