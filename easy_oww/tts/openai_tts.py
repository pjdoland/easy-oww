"""
OpenAI TTS integration for high-quality synthetic sample generation
"""
import os
from pathlib import Path
from typing import Optional, List
import wave
import numpy as np
from easy_oww.utils.logger import get_logger

logger = get_logger()


class OpenAITTS:
    """Manages OpenAI TTS for speech generation"""

    # Available GPT-4o-mini TTS voices
    # These voices are high quality and cost-effective
    AVAILABLE_VOICES = [
        "alloy",    # Neutral, balanced
        "echo",     # Male, clear
        "fable",    # British accent, warm
        "onyx",     # Deep male voice
        "nova",     # Female, energetic
        "shimmer",  # Female, soft
        "ash",      # Male, conversational (new)
        "ballad",   # Female, expressive (new)
        "coral",    # Female, warm (new)
        "sage",     # Male, professional (new)
        "verse",    # Male, narrative (new)
    ]

    # Pricing: ~$0.10 per 1M characters for gpt-4o-mini-tts
    COST_PER_MILLION_CHARS = 0.10

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI TTS

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        # Track usage for cost estimation
        self.total_characters = 0
        self.total_requests = 0

    def is_installed(self) -> bool:
        """
        Check if OpenAI package is installed

        Returns:
            True if openai package is available
        """
        try:
            import openai
            return True
        except ImportError:
            return False

    def generate_speech(
        self,
        text: str,
        voice: str = "alloy",
        output_path: Path = None,
        sample_rate: int = 16000,
        speed: float = 1.0
    ) -> bool:
        """
        Generate speech from text using OpenAI TTS

        Args:
            text: Text to convert to speech
            voice: Voice name (one of AVAILABLE_VOICES)
            output_path: Output WAV file path
            sample_rate: Output sample rate (will resample from 24kHz)
            speed: Speaking rate (0.25 to 4.0, 1.0 = normal)

        Returns:
            True if generation successful

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_installed():
            raise RuntimeError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        if voice not in self.AVAILABLE_VOICES:
            raise ValueError(
                f"Invalid voice '{voice}'. Available: {', '.join(self.AVAILABLE_VOICES)}"
            )

        # Clamp speed to valid range
        speed = max(0.25, min(4.0, speed))

        try:
            from openai import OpenAI
            import io
            import wave
            import numpy as np
            import time

            client = OpenAI(api_key=self.api_key)

            # Track usage
            self.total_characters += len(text)
            self.total_requests += 1

            # Generate speech using PCM format with retry logic for rate limits
            # PCM gives us raw audio data that we can process directly
            max_retries = 5
            base_delay = 0.5  # Start with 500ms delay

            for attempt in range(max_retries):
                try:
                    response = client.audio.speech.create(
                        model="gpt-4o-mini-tts",  # Most cost-effective model
                        voice=voice,
                        input=text,
                        speed=speed,
                        response_format="pcm"  # Raw PCM audio data
                    )
                    break  # Success, exit retry loop

                except Exception as e:
                    error_str = str(e)

                    # Check if it's a rate limit error (429)
                    if "429" in error_str or "rate_limit" in error_str.lower():
                        if attempt < max_retries - 1:
                            # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s
                            delay = base_delay * (2 ** attempt)
                            from easy_oww.utils.logger import get_logger
                            logger = get_logger()
                            logger.debug(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                        else:
                            # Final attempt failed
                            raise RuntimeError(f"Speech generation failed after {max_retries} attempts: {e}")
                    else:
                        # Non-rate-limit error, don't retry
                        raise RuntimeError(f"Speech generation failed: {e}")

            # Get PCM audio data
            # OpenAI returns 24kHz, 16-bit, mono PCM by default
            pcm_data = response.content

            # Convert PCM bytes to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)

            # Resample if needed (from 24kHz to target sample_rate)
            source_rate = 24000  # OpenAI TTS outputs at 24kHz
            if source_rate != sample_rate:
                from scipy import signal
                num_samples = int(len(audio_array) * sample_rate / source_rate)
                audio_array = signal.resample(audio_array, num_samples).astype(np.int16)

            # Write as WAV file
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with wave.open(str(output_path), 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_array.tobytes())

                logger.debug(f"Generated speech: {output_path}")

            return True

        except ImportError as e:
            raise RuntimeError(f"Missing dependency: {e}")
        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            raise RuntimeError(f"Speech generation failed: {e}")

    def batch_generate(
        self,
        texts: List[str],
        voices: List[str],
        output_dir: Path,
        sample_rate: int = 16000,
        prefix: str = "sample",
        speed_variations: List[float] = None
    ) -> List[Path]:
        """
        Generate multiple speech samples with voice rotation

        Args:
            texts: List of texts to convert
            voices: List of voice names to rotate through
            output_dir: Output directory
            sample_rate: Output sample rate
            prefix: Filename prefix
            speed_variations: Optional list of speed values to vary

        Returns:
            List of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not voices:
            voices = ["alloy"]  # Default voice

        if speed_variations is None:
            speed_variations = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

        generated_files = []

        for i, text in enumerate(texts):
            output_path = output_dir / f"{prefix}_{i:04d}.wav"

            # Rotate through voices
            voice = voices[i % len(voices)]

            # Random speed variation
            speed = np.random.choice(speed_variations)

            try:
                if self.generate_speech(
                    text, voice, output_path, sample_rate, speed
                ):
                    generated_files.append(output_path)
            except Exception as e:
                logger.warning(f"Failed to generate sample {i}: {e}")
                continue

        return generated_files

    def estimate_cost(self, character_count: int) -> float:
        """
        Estimate cost for generating speech

        Args:
            character_count: Number of characters to synthesize

        Returns:
            Estimated cost in USD
        """
        return (character_count / 1_000_000) * self.COST_PER_MILLION_CHARS

    def get_usage_stats(self) -> dict:
        """
        Get TTS usage statistics

        Returns:
            Dictionary with usage stats and cost estimate
        """
        cost = self.estimate_cost(self.total_characters)

        return {
            'total_requests': self.total_requests,
            'total_characters': self.total_characters,
            'estimated_cost_usd': round(cost, 4),
            'cost_per_request': round(cost / max(1, self.total_requests), 6)
        }

    def reset_usage_stats(self):
        """Reset usage tracking counters"""
        self.total_characters = 0
        self.total_requests = 0

    @classmethod
    def get_voice_list(cls) -> List[str]:
        """
        Get list of available voices

        Returns:
            List of voice names
        """
        return cls.AVAILABLE_VOICES.copy()

    @classmethod
    def get_diverse_voices(cls, count: int = 8) -> List[str]:
        """
        Get a diverse set of voices

        Args:
            count: Number of voices to return (max 11)

        Returns:
            List of diverse voice names
        """
        # Return up to 'count' voices with good diversity
        # Prioritize variety in gender and tone
        diverse_set = [
            "alloy",    # Neutral
            "nova",     # Female energetic
            "echo",     # Male clear
            "shimmer",  # Female soft
            "onyx",     # Deep male
            "fable",    # British warm
            "ash",      # Male conversational
            "coral",    # Female warm
            "sage",     # Male professional
            "ballad",   # Female expressive
            "verse",    # Male narrative
        ]

        return diverse_set[:min(count, len(diverse_set))]
