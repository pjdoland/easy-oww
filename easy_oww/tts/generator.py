"""
Synthetic sample generation using TTS
"""
import random
from pathlib import Path
from typing import List, Optional, Dict
from rich.progress import Progress, TaskID
from rich.console import Console
import numpy as np

from easy_oww.tts.piper import PiperTTS
from easy_oww.utils.logger import get_logger

logger = get_logger()
console = Console()


class SampleGenerator:
    """Generates synthetic wake word samples using TTS"""

    # Variations to apply to wake word for natural speech
    PHRASE_VARIATIONS = [
        "{word}",  # Plain
        "{word}.",  # With period
        "{word}!",  # With exclamation
        "{word}?",  # With question
        "hey {word}",  # With prefix
        "ok {word}",  # Alternative prefix
        "{word} please",  # With suffix
    ]

    # Speed variations (if supported by TTS)
    SPEED_VARIATIONS = [0.9, 1.0, 1.1, 1.2]

    def __init__(
        self,
        piper: PiperTTS,
        output_dir: Path,
        sample_rate: int = 16000
    ):
        """
        Initialize sample generator

        Args:
            piper: PiperTTS instance
            output_dir: Directory to save generated samples
            sample_rate: Output sample rate
        """
        self.piper = piper
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

    def generate_variations(self, wake_word: str, count: int = 10) -> List[str]:
        """
        Generate text variations of wake word

        Args:
            wake_word: Base wake word or phrase
            count: Number of variations to generate

        Returns:
            List of text variations
        """
        variations = []
        templates = self.PHRASE_VARIATIONS.copy()

        while len(variations) < count:
            # Cycle through templates
            template = templates[len(variations) % len(templates)]
            text = template.format(word=wake_word)

            # Add slight variations
            if random.random() > 0.7:
                # Sometimes add a pause (comma)
                text = text.replace(wake_word, f"{wake_word},")

            variations.append(text)

        return variations[:count]

    def generate_samples(
        self,
        wake_word: str,
        voice_model: Path,
        count: int = 100,
        prefix: str = "synthetic",
        show_progress: bool = True
    ) -> List[Path]:
        """
        Generate synthetic samples for wake word

        Args:
            wake_word: Wake word or phrase
            voice_model: Path to voice model
            count: Number of samples to generate
            prefix: Filename prefix
            show_progress: Show progress bar

        Returns:
            List of generated file paths
        """
        logger.info(f"Generating {count} synthetic samples for '{wake_word}'")

        # Generate text variations
        # Create more variations than needed to have variety
        variations = self.generate_variations(wake_word, count * 2)

        # Shuffle for randomness
        random.shuffle(variations)

        generated_files = []
        failed_count = 0

        if show_progress:
            with Progress(console=console) as progress:
                task = progress.add_task(
                    f"Generating samples...",
                    total=count
                )

                for i in range(count):
                    text = variations[i % len(variations)]
                    output_path = self.output_dir / f"{prefix}_{i:04d}.wav"

                    try:
                        success = self.piper.generate_speech(
                            text,
                            voice_model,
                            output_path,
                            self.sample_rate
                        )

                        if success:
                            generated_files.append(output_path)
                        else:
                            failed_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to generate sample {i}: {e}")
                        failed_count += 1

                    progress.update(
                        task,
                        advance=1,
                        description=f"Generated {len(generated_files)}/{count} samples"
                    )
        else:
            for i in range(count):
                text = variations[i % len(variations)]
                output_path = self.output_dir / f"{prefix}_{i:04d}.wav"

                try:
                    success = self.piper.generate_speech(
                        text,
                        voice_model,
                        output_path,
                        self.sample_rate
                    )

                    if success:
                        generated_files.append(output_path)
                    else:
                        failed_count += 1

                except Exception as e:
                    logger.warning(f"Failed to generate sample {i}: {e}")
                    failed_count += 1

        if failed_count > 0:
            logger.warning(f"Failed to generate {failed_count} samples")

        logger.info(f"Successfully generated {len(generated_files)} samples")
        return generated_files

    def generate_multi_voice(
        self,
        wake_word: str,
        voice_models: List[Path],
        samples_per_voice: int = 50,
        prefix: str = "synthetic"
    ) -> Dict[str, List[Path]]:
        """
        Generate samples using multiple voices

        Args:
            wake_word: Wake word or phrase
            voice_models: List of voice model paths
            samples_per_voice: Number of samples per voice
            prefix: Filename prefix

        Returns:
            Dictionary mapping voice names to generated file lists
        """
        console.print(f"\n[bold]Generating samples with {len(voice_models)} voices...[/bold]")

        results = {}

        for i, voice_model in enumerate(voice_models):
            voice_name = voice_model.stem

            console.print(f"\n[cyan]Voice {i+1}/{len(voice_models)}:[/cyan] {voice_name}")

            # Create voice-specific subdirectory
            voice_dir = self.output_dir / voice_name
            voice_dir.mkdir(exist_ok=True)

            # Temporary change output directory
            original_dir = self.output_dir
            self.output_dir = voice_dir

            try:
                generated = self.generate_samples(
                    wake_word,
                    voice_model,
                    count=samples_per_voice,
                    prefix=f"{prefix}_{voice_name}",
                    show_progress=True
                )

                results[voice_name] = generated

            except Exception as e:
                logger.error(f"Failed to generate samples with {voice_name}: {e}")
                results[voice_name] = []

            finally:
                # Restore output directory
                self.output_dir = original_dir

        # Summary
        total_generated = sum(len(files) for files in results.values())
        console.print(f"\n[green]✓ Generated {total_generated} total samples across {len(voice_models)} voices[/green]")

        return results

    def generate_mixed_samples(
        self,
        wake_word: str,
        voice_models: List[Path],
        total_count: int = 500,
        prefix: str = "mixed"
    ) -> List[Path]:
        """
        Generate samples distributed across multiple voices

        Args:
            wake_word: Wake word or phrase
            voice_models: List of voice model paths
            total_count: Total number of samples to generate
            prefix: Filename prefix

        Returns:
            List of all generated file paths
        """
        if not voice_models:
            raise ValueError("No voice models provided")

        samples_per_voice = total_count // len(voice_models)
        remainder = total_count % len(voice_models)

        console.print(f"\n[bold]Generating {total_count} samples across {len(voice_models)} voices[/bold]")
        console.print(f"~{samples_per_voice} samples per voice")

        all_files = []

        for i, voice_model in enumerate(voice_models):
            voice_name = voice_model.stem

            # Add remainder samples to first voices
            count = samples_per_voice + (1 if i < remainder else 0)

            console.print(f"\n[cyan]Voice {i+1}/{len(voice_models)}:[/cyan] {voice_name} ({count} samples)")

            try:
                generated = self.generate_samples(
                    wake_word,
                    voice_model,
                    count=count,
                    prefix=f"{prefix}_{i:02d}",
                    show_progress=True
                )

                all_files.extend(generated)

            except Exception as e:
                logger.error(f"Failed to generate samples with {voice_name}: {e}")
                continue

        console.print(f"\n[green]✓ Generated {len(all_files)} total samples[/green]")
        return all_files

    def estimate_generation_time(
        self,
        count: int,
        voices: int = 1,
        avg_seconds_per_sample: float = 2.0
    ) -> float:
        """
        Estimate time to generate samples

        Args:
            count: Number of samples
            voices: Number of voices
            avg_seconds_per_sample: Average generation time per sample

        Returns:
            Estimated time in seconds
        """
        # TTS generation is relatively fast
        # Estimate ~0.5-2 seconds per sample depending on text length
        return count * voices * avg_seconds_per_sample

    def validate_generated_samples(
        self,
        samples: List[Path],
        min_duration: float = 0.3,
        max_duration: float = 3.0
    ) -> Dict[str, any]:
        """
        Validate generated samples

        Args:
            samples: List of sample paths
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds

        Returns:
            Validation results dictionary
        """
        from scipy.io import wavfile

        valid_samples = []
        invalid_samples = []
        issues = []

        for sample_path in samples:
            try:
                sample_rate, audio = wavfile.read(str(sample_path))

                duration = len(audio) / sample_rate

                if duration < min_duration:
                    invalid_samples.append(sample_path)
                    issues.append(f"{sample_path.name}: too short ({duration:.2f}s)")
                elif duration > max_duration:
                    invalid_samples.append(sample_path)
                    issues.append(f"{sample_path.name}: too long ({duration:.2f}s)")
                elif len(audio) == 0:
                    invalid_samples.append(sample_path)
                    issues.append(f"{sample_path.name}: empty file")
                else:
                    valid_samples.append(sample_path)

            except Exception as e:
                invalid_samples.append(sample_path)
                issues.append(f"{sample_path.name}: {e}")

        return {
            'total': len(samples),
            'valid': len(valid_samples),
            'invalid': len(invalid_samples),
            'valid_samples': valid_samples,
            'invalid_samples': invalid_samples,
            'issues': issues
        }
