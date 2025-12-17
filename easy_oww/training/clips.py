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
        target_duration_ms: int = 1000,
        negative_recordings_dir: Optional[Path] = None
    ):
        """
        Initialize clip generator

        Args:
            recordings_dir: Directory with real recordings
            clips_dir: Directory to save processed clips
            sample_rate: Target sample rate
            target_duration_ms: Target clip duration in milliseconds
            negative_recordings_dir: Directory with negative/adversarial recordings
        """
        self.recordings_dir = Path(recordings_dir)
        self.clips_dir = Path(clips_dir)
        self.negative_recordings_dir = Path(negative_recordings_dir) if negative_recordings_dir else None
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
                    # Preserve content for positive clips (never cut off wake word)
                    audio = self._prepare_clip(audio, preserve_content=True)

                    # Save to positive clips
                    output_path = self.positive_dir / f"real_{i:04d}.wav"
                    wavfile.write(str(output_path), self.sample_rate, audio)

                    processed_clips.append(output_path)

                except Exception as e:
                    logger.warning(f"Failed to process {recording_path.name}: {e}")

                progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Processed {len(processed_clips)} recordings")
        return processed_clips

    def process_negative_recordings(self) -> List[Path]:
        """
        Process negative/adversarial recordings into training clips

        Returns:
            List of processed negative clip paths
        """
        if not self.negative_recordings_dir or not self.negative_recordings_dir.exists():
            logger.debug("No negative recordings directory found")
            return []

        recording_files = list(self.negative_recordings_dir.glob('*.wav'))

        if not recording_files:
            logger.debug("No negative recordings found")
            return []

        console.print(f"\n[bold]Processing {len(recording_files)} negative recordings...[/bold]")

        processed_clips = []

        with Progress(console=console) as progress:
            task = progress.add_task("Processing negative recordings...", total=len(recording_files))

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

                    # Save to negative clips
                    output_path = self.negative_dir / f"real_negative_{i:04d}.wav"
                    wavfile.write(str(output_path), self.sample_rate, audio)

                    processed_clips.append(output_path)

                except Exception as e:
                    logger.warning(f"Failed to process {recording_path.name}: {e}")

                progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Processed {len(processed_clips)} negative recordings")
        return processed_clips

    def generate_synthetic_clips(
        self,
        wake_word: str,
        voice_models: List[Path] = None,
        piper: PiperTTS = None,
        count: int = 500,
        use_openai: bool = True
    ) -> List[Path]:
        """
        Generate synthetic clips using TTS

        Args:
            wake_word: Wake word text
            voice_models: List of voice model paths (for Piper TTS)
            piper: PiperTTS instance (for Piper TTS)
            count: Number of clips to generate
            use_openai: Use OpenAI TTS if available (default: True)

        Returns:
            List of generated clip paths
        """
        import os
        from easy_oww.tts import OpenAITTS
        import random

        console.print(f"\n[bold]Generating {count} synthetic clips...[/bold]")

        # Check for existing synthetic clips and resume if needed
        existing_clips = list(self.positive_dir.glob('synth_*.wav'))
        existing_count = len(existing_clips)

        if existing_count >= count:
            console.print(f"[green]✓[/green] Already have {existing_count} synthetic clips (target: {count})")
            return existing_clips[:count]

        if existing_count > 0:
            console.print(f"[cyan]Found {existing_count} existing synthetic clips, generating {count - existing_count} more...[/cyan]")

        # Find the starting index (highest existing number + 1)
        start_index = 0
        if existing_clips:
            import re
            numbers = []
            for clip in existing_clips:
                match = re.search(r'synth_(\d+)\.wav', clip.name)
                if match:
                    numbers.append(int(match.group(1)))
            if numbers:
                start_index = max(numbers) + 1

        # Create temporary directory for raw TTS output
        temp_dir = self.clips_dir / 'temp_tts'
        temp_dir.mkdir(exist_ok=True)

        try:
            raw_samples = []
            clips_to_generate = count - existing_count

            # Try to use OpenAI TTS first if requested
            if use_openai and os.environ.get('OPENAI_API_KEY'):
                try:
                    console.print("[cyan]Using OpenAI GPT-4o-mini-TTS for synthetic samples[/cyan]")

                    tts = OpenAITTS()
                    voices = OpenAITTS.get_diverse_voices(count=10)

                    # Estimate cost
                    avg_chars_per_sample = len(wake_word) * 1.5  # Account for variations
                    total_chars = int(avg_chars_per_sample * clips_to_generate)
                    estimated_cost = tts.estimate_cost(total_chars)
                    console.print(f"  Estimated cost: ${estimated_cost:.4f} for {total_chars:,} characters")

                    # Generate text variations
                    from easy_oww.tts import SampleGenerator
                    generator = SampleGenerator(
                        piper=None,  # Not used for OpenAI
                        output_dir=temp_dir,
                        sample_rate=self.sample_rate
                    )

                    variations = generator.generate_variations(wake_word, clips_to_generate * 2)
                    random.shuffle(variations)

                    # Generate samples with OpenAI TTS in parallel
                    speed_variations = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    from rich.progress import Progress

                    # Function to generate a single clip
                    def generate_clip(i):
                        text = variations[i % len(variations)]
                        voice = voices[i % len(voices)]
                        speed = random.choice(speed_variations)
                        output_path = temp_dir / f"synthetic_{start_index + i:04d}.wav"

                        try:
                            if tts.generate_speech(text, voice, output_path, self.sample_rate, speed):
                                return output_path
                        except Exception as e:
                            logger.warning(f"Failed to generate sample {i}: {e}")
                        return None

                    # Use ThreadPoolExecutor for parallel I/O-bound TTS calls
                    # Limit to 10 workers to avoid overwhelming the API
                    with Progress(console=console) as progress:
                        task = progress.add_task("Generating with OpenAI TTS...", total=clips_to_generate)

                        with ThreadPoolExecutor(max_workers=10) as executor:
                            # Submit all tasks
                            futures = [executor.submit(generate_clip, i) for i in range(clips_to_generate)]

                            # Collect results as they complete
                            for future in as_completed(futures):
                                result = future.result()
                                if result is not None:
                                    raw_samples.append(result)
                                progress.update(task, advance=1)

                    # Report usage stats
                    usage_stats = tts.get_usage_stats()
                    console.print(f"  [cyan]TTS Usage:[/cyan] {usage_stats['total_requests']:,} requests, "
                                 f"{usage_stats['total_characters']:,} characters")
                    console.print(f"  [cyan]Cost:[/cyan] ${usage_stats['estimated_cost_usd']:.4f}")

                except Exception as e:
                    console.print(f"[yellow]OpenAI TTS failed: {e}[/yellow]")
                    console.print(f"[yellow]Falling back to Piper TTS[/yellow]")
                    use_openai = False
            else:
                use_openai = False

            # Fall back to Piper TTS if OpenAI not available or failed
            if not use_openai:
                if not piper or not voice_models:
                    raise ValueError("Piper TTS instance and voice models required when OpenAI TTS not available")

                console.print("[cyan]Using Piper TTS for synthetic samples[/cyan]")

                # Generate samples with Piper
                generator = SampleGenerator(
                    piper=piper,
                    output_dir=temp_dir,
                    sample_rate=self.sample_rate
                )

                raw_samples = generator.generate_mixed_samples(
                    wake_word=wake_word,
                    voice_models=voice_models,
                    total_count=clips_to_generate,
                    prefix=f"synthetic_{start_index:04d}"
                )

            # Process generated samples into clips
            console.print("\n[bold]Processing synthetic samples...[/bold]")
            processed_clips = list(existing_clips)  # Start with existing clips

            with Progress(console=console) as progress:
                task = progress.add_task("Processing...", total=len(raw_samples))

                for i, sample_path in enumerate(raw_samples):
                    try:
                        # Load audio
                        sample_rate, audio = wavfile.read(str(sample_path))

                        # Apply random time stretch for speed variation (70% of clips)
                        # This compensates for older Piper versions without length_scale
                        if np.random.random() < 0.7:
                            # Wider range for more noticeable variation
                            stretch_rate = np.random.uniform(0.75, 1.25)
                            if abs(stretch_rate - 1.0) > 0.05:
                                # Resample for time stretch
                                num_samples = int(len(audio) * stretch_rate)
                                from scipy import signal as scipy_signal
                                audio = scipy_signal.resample(audio, num_samples).astype(np.int16)

                        # Prepare clip (normalize, trim/pad)
                        # Preserve content for positive clips (never cut off wake word)
                        audio = self._prepare_clip(audio, preserve_content=True)

                        # Save to positive clips with correct index
                        output_path = self.positive_dir / f"synth_{start_index + i:04d}.wav"
                        wavfile.write(str(output_path), self.sample_rate, audio)

                        processed_clips.append(output_path)

                    except Exception as e:
                        logger.warning(f"Failed to process {sample_path.name}: {e}")

                    progress.update(task, advance=1)

            console.print(f"[green]✓[/green] Total {len(processed_clips)} synthetic clips ({existing_count} existing + {len(processed_clips) - existing_count} new)")
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

    def _prepare_clip(self, audio: np.ndarray, preserve_content: bool = False) -> np.ndarray:
        """
        Prepare audio clip (normalize, trim/pad to target duration)

        Args:
            audio: Audio data
            preserve_content: If True, never trim audio (only pad). Use for wake word clips.

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
            if preserve_content:
                # For wake word clips, detect speech boundaries and crop intelligently
                # Find the loudest section (likely contains the wake word)
                audio_float = np.abs(audio.astype(np.float32))

                # Use a sliding window to find the loudest section
                window_size = self.target_samples
                if len(audio_float) >= window_size:
                    max_energy = -1
                    best_start = 0

                    # Check every possible position
                    for start_pos in range(len(audio_float) - window_size + 1):
                        window = audio_float[start_pos:start_pos + window_size]
                        energy = np.sum(window ** 2)

                        if energy > max_energy:
                            max_energy = energy
                            best_start = start_pos

                    # Extract the loudest section (most likely contains wake word)
                    audio = audio[best_start:best_start + self.target_samples]
                    logger.debug(f"Trimmed audio using energy-based crop at position {best_start}/{len(audio_float)}")
                else:
                    # Fallback to center crop if too short
                    start = (len(audio) - self.target_samples) // 2
                    audio = audio[start:start + self.target_samples]
            else:
                # For negative clips, use center crop
                start = (len(audio) - self.target_samples) // 2
                audio = audio[start:start + self.target_samples]
        elif len(audio) < self.target_samples:
            # Pad with silence (always safe)
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

    def cleanup_synthetic_clips(self):
        """
        Delete all synthetic clips but preserve real recordings.
        Used when force mode is enabled to regenerate synthetic samples.
        """
        from rich.console import Console
        console = Console()

        deleted_count = 0

        # Delete synthetic positive clips (prefixed with 'synth_')
        if self.positive_dir.exists():
            for clip_path in self.positive_dir.glob('synth_*.wav'):
                try:
                    clip_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {clip_path}: {e}")

        # Delete all negative clips (they're regenerated from datasets)
        # But preserve real negative recordings (prefixed with 'real_negative_')
        if self.negative_dir.exists():
            for clip_path in self.negative_dir.glob('*.wav'):
                # Skip real negative recordings
                if clip_path.name.startswith('real_negative_'):
                    continue
                try:
                    clip_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {clip_path}: {e}")

        # Delete adversarial negative clips
        adversarial_dir = self.clips_dir / 'adversarial_negative'
        if adversarial_dir.exists():
            for clip_path in adversarial_dir.glob('*.wav'):
                try:
                    clip_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {clip_path}: {e}")

        # Clean up augmented clips
        positive_aug_dir = self.clips_dir / 'positive_augmented'
        if positive_aug_dir.exists():
            shutil.rmtree(positive_aug_dir)
            positive_aug_dir.mkdir(exist_ok=True)

        if deleted_count > 0:
            console.print(f"[yellow]Deleted {deleted_count} synthetic clips (preserving real recordings)[/yellow]")

        return deleted_count

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
