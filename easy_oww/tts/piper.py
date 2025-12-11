"""
Piper TTS integration for synthetic sample generation
"""
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import json
from easy_oww.utils.logger import get_logger

logger = get_logger()


class PiperTTS:
    """Manages Piper TTS installation and usage"""

    # Piper is now available on PyPI as piper-tts
    # The old rhasspy/piper repo was archived and moved to OHF-Voice/piper1-gpl
    PIPER_PACKAGE = "piper-tts"
    PIPER_VERSION = "1.3.0"  # Latest stable version on PyPI

    def __init__(self, install_dir: Path):
        """
        Initialize Piper TTS manager

        Args:
            install_dir: Directory to install Piper (unused now, kept for compatibility)
        """
        self.install_dir = Path(install_dir)
        self.install_dir.mkdir(parents=True, exist_ok=True)

        # Piper is now a Python package, not a standalone binary
        # The binary is installed via pip in the system/venv
        self.voices_dir = self.install_dir / 'voices'
        self.voices_dir.mkdir(exist_ok=True)

    def is_installed(self) -> bool:
        """
        Check if Piper is installed as a Python package

        Returns:
            True if piper-tts package is installed
        """
        try:
            import importlib.util
            spec = importlib.util.find_spec("piper")
            return spec is not None
        except (ImportError, ModuleNotFoundError):
            return False

    def install(self, force: bool = False) -> bool:
        """
        Install Piper TTS using pip

        Args:
            force: Force reinstall even if already installed

        Returns:
            True if installation successful

        Raises:
            RuntimeError: If installation fails
        """
        if self.is_installed() and not force:
            logger.info("Piper already installed")
            return True

        logger.info("Installing Piper TTS via pip...")

        try:
            import sys

            # Install using pip
            cmd = [sys.executable, "-m", "pip", "install", self.PIPER_PACKAGE]

            if force:
                cmd.append("--force-reinstall")

            logger.debug(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for pip install
            )

            if result.returncode != 0:
                stderr = result.stderr
                raise RuntimeError(f"pip install failed: {stderr}")

            logger.info("Piper installed successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Piper installation timed out")
            raise RuntimeError("Piper installation timed out")
        except Exception as e:
            logger.error(f"Failed to install Piper: {e}")
            raise RuntimeError(f"Piper installation failed: {e}")

    def generate_speech(
        self,
        text: str,
        voice_model: Path,
        output_path: Path,
        sample_rate: int = 16000,
        length_scale: float = 1.0
    ) -> bool:
        """
        Generate speech from text using Piper Python package

        Args:
            text: Text to convert to speech
            voice_model: Path to voice model (.onnx file)
            output_path: Output WAV file path
            sample_rate: Output sample rate (default: 16000)
            length_scale: Speaking rate (1.0 = normal, <1.0 = faster, >1.0 = slower)

        Returns:
            True if generation successful

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_installed():
            raise RuntimeError("Piper is not installed. Install with: pip install piper-tts")

        if not voice_model.exists():
            raise RuntimeError(f"Voice model not found: {voice_model}")

        try:
            from piper import PiperVoice
            import wave

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Load voice model
            voice = PiperVoice.load(str(voice_model))

            # Generate speech with length_scale for speed variation
            # voice.synthesize() returns a generator that yields AudioChunk objects
            import numpy as np
            result = voice.synthesize(text, length_scale=length_scale)

            # Collect audio data from all chunks
            audio_data = []
            for audio_chunk in result:
                # AudioChunk has audio_int16_array attribute
                if hasattr(audio_chunk, 'audio_int16_array'):
                    audio_data.extend(audio_chunk.audio_int16_array)
                else:
                    # Fallback for older API
                    audio_data.extend(audio_chunk)

            # Convert to numpy array
            audio_array = np.array(audio_data, dtype=np.int16)

            # Write WAV file
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_array.tobytes())

            logger.debug(f"Generated speech: {output_path}")
            return True

        except ImportError as e:
            logger.error("Piper package not found")
            raise RuntimeError("Piper is not installed. Install with: pip install piper-tts")
        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            raise RuntimeError(f"Speech generation failed: {e}")

    def batch_generate(
        self,
        texts: List[str],
        voice_model: Path,
        output_dir: Path,
        sample_rate: int = 16000,
        prefix: str = "sample"
    ) -> List[Path]:
        """
        Generate multiple speech samples

        Args:
            texts: List of texts to convert
            voice_model: Path to voice model
            output_dir: Output directory
            sample_rate: Output sample rate
            prefix: Filename prefix

        Returns:
            List of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        for i, text in enumerate(texts):
            output_path = output_dir / f"{prefix}_{i:04d}.wav"

            try:
                if self.generate_speech(text, voice_model, output_path, sample_rate):
                    generated_files.append(output_path)
            except Exception as e:
                logger.warning(f"Failed to generate sample {i}: {e}")
                continue

        return generated_files

    def list_installed_voices(self) -> List[dict]:
        """
        List installed voice models

        Returns:
            List of voice model information dictionaries
        """
        voices = []

        for model_file in self.voices_dir.glob('**/*.onnx'):
            # Read config if available
            config_file = model_file.with_suffix('.onnx.json')
            config = {}

            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read config {config_file}: {e}")

            voices.append({
                'name': model_file.stem,
                'path': model_file,
                'config': config,
                'language': config.get('language', {}).get('code', 'unknown'),
                'quality': config.get('audio', {}).get('quality', 'unknown'),
                'sample_rate': config.get('audio', {}).get('sample_rate', 22050)
            })

        return voices

    def get_voice(self, name: str) -> Optional[Path]:
        """
        Get path to voice model by name

        Args:
            name: Voice name (without .onnx extension)

        Returns:
            Path to voice model or None if not found
        """
        voice_path = self.voices_dir / f"{name}.onnx"

        if voice_path.exists():
            return voice_path

        # Try searching in subdirectories
        matches = list(self.voices_dir.glob(f"**/{name}.onnx"))
        if matches:
            return matches[0]

        return None

    def test_voice(self, voice_model: Path) -> Tuple[bool, str]:
        """
        Test voice model with sample text

        Args:
            voice_model: Path to voice model

        Returns:
            Tuple of (success, message)
        """
        import tempfile

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / 'test.wav'

                self.generate_speech(
                    "This is a test of the text to speech system.",
                    voice_model,
                    output_path
                )

                if output_path.exists() and output_path.stat().st_size > 0:
                    return True, "Voice model working correctly"
                else:
                    return False, "Generated file is empty or missing"

        except Exception as e:
            return False, f"Voice test failed: {e}"
