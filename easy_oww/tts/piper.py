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

    # Piper GitHub releases
    PIPER_REPO = "rhasspy/piper"
    PIPER_VERSION = "2023.11.14-2"  # Latest stable version

    def __init__(self, install_dir: Path):
        """
        Initialize Piper TTS manager

        Args:
            install_dir: Directory to install Piper
        """
        self.install_dir = Path(install_dir)
        self.install_dir.mkdir(parents=True, exist_ok=True)

        self.piper_binary = self._get_binary_path()
        self.voices_dir = self.install_dir / 'voices'
        self.voices_dir.mkdir(exist_ok=True)

    def _get_binary_path(self) -> Path:
        """
        Get path to Piper binary based on platform

        Returns:
            Path to Piper binary
        """
        system = platform.system().lower()

        if system == 'darwin':
            # macOS
            if platform.machine() == 'arm64':
                binary_name = 'piper'  # Apple Silicon
            else:
                binary_name = 'piper'  # Intel
        elif system == 'linux':
            binary_name = 'piper'
        elif system == 'windows':
            binary_name = 'piper.exe'
        else:
            binary_name = 'piper'

        return self.install_dir / binary_name

    def is_installed(self) -> bool:
        """
        Check if Piper is installed

        Returns:
            True if Piper binary exists and is executable
        """
        if not self.piper_binary.exists():
            return False

        # Check if executable
        if platform.system() != 'Windows':
            return self.piper_binary.stat().st_mode & 0o111 != 0

        return True

    def get_download_url(self) -> Tuple[str, str]:
        """
        Get download URL for current platform

        Returns:
            Tuple of (download_url, archive_name)

        Raises:
            RuntimeError: If platform is not supported
        """
        system = platform.system().lower()
        machine = platform.machine().lower()

        base_url = f"https://github.com/{self.PIPER_REPO}/releases/download/{self.PIPER_VERSION}"

        if system == 'darwin':
            if 'arm' in machine or 'aarch64' in machine:
                archive = f"piper_macos_arm64.tar.gz"
            else:
                archive = f"piper_macos_x64.tar.gz"
        elif system == 'linux':
            if 'aarch64' in machine or 'arm64' in machine:
                archive = f"piper_linux_aarch64.tar.gz"
            elif 'arm' in machine:
                archive = f"piper_linux_armv7l.tar.gz"
            else:
                archive = f"piper_linux_x86_64.tar.gz"
        elif system == 'windows':
            archive = f"piper_windows_amd64.zip"
        else:
            raise RuntimeError(f"Unsupported platform: {system} {machine}")

        url = f"{base_url}/{archive}"
        return url, archive

    def install(self, force: bool = False) -> bool:
        """
        Install Piper TTS

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

        logger.info("Installing Piper TTS...")

        try:
            import requests
            import tarfile
            import zipfile

            # Get download URL
            url, archive_name = self.get_download_url()
            logger.debug(f"Downloading from: {url}")

            # Download
            response = requests.get(url, stream=True)
            response.raise_for_status()

            archive_path = self.install_dir / archive_name
            total_size = int(response.headers.get('content-length', 0))

            # Save archive
            with open(archive_path, 'wb') as f:
                if total_size > 0:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        logger.debug(f"Download progress: {progress:.1f}%")
                else:
                    f.write(response.content)

            logger.info("Extracting Piper...")

            # Extract based on archive type
            if archive_name.endswith('.tar.gz'):
                with tarfile.open(archive_path, 'r:gz') as tar:
                    # Extract to temporary location first
                    temp_dir = self.install_dir / 'temp_extract'
                    temp_dir.mkdir(exist_ok=True)
                    tar.extractall(temp_dir)

                    # Find piper binary in extracted files
                    piper_files = list(temp_dir.rglob('piper'))
                    if piper_files:
                        # Move binary to install directory
                        shutil.move(str(piper_files[0]), str(self.piper_binary))
                    else:
                        raise RuntimeError("Piper binary not found in archive")

                    # Clean up
                    shutil.rmtree(temp_dir)

            elif archive_name.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    temp_dir = self.install_dir / 'temp_extract'
                    temp_dir.mkdir(exist_ok=True)
                    zip_ref.extractall(temp_dir)

                    # Find piper binary
                    piper_files = list(temp_dir.rglob('piper.exe'))
                    if piper_files:
                        shutil.move(str(piper_files[0]), str(self.piper_binary))
                    else:
                        raise RuntimeError("Piper binary not found in archive")

                    # Clean up
                    shutil.rmtree(temp_dir)

            # Clean up archive
            archive_path.unlink()

            # Make executable on Unix-like systems
            if platform.system() != 'Windows':
                self.piper_binary.chmod(0o755)

            logger.info("Piper installed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to install Piper: {e}")
            raise RuntimeError(f"Piper installation failed: {e}")

    def generate_speech(
        self,
        text: str,
        voice_model: Path,
        output_path: Path,
        sample_rate: int = 16000
    ) -> bool:
        """
        Generate speech from text using Piper

        Args:
            text: Text to convert to speech
            voice_model: Path to voice model (.onnx file)
            output_path: Output WAV file path
            sample_rate: Output sample rate (default: 16000)

        Returns:
            True if generation successful

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_installed():
            raise RuntimeError("Piper is not installed")

        if not voice_model.exists():
            raise RuntimeError(f"Voice model not found: {voice_model}")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Run Piper
            # Piper command: echo "text" | piper --model voice.onnx --output_file output.wav
            cmd = [
                str(self.piper_binary),
                '--model', str(voice_model),
                '--output_file', str(output_path),
                '--sample_rate', str(sample_rate)
            ]

            # Check if model config exists
            model_config = voice_model.with_suffix('.onnx.json')
            if model_config.exists():
                cmd.extend(['--config', str(model_config)])

            logger.debug(f"Running Piper: {' '.join(cmd)}")

            # Run command with text as input
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=30
            )

            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Piper failed: {stderr}")

            if not output_path.exists():
                raise RuntimeError("Output file was not created")

            logger.debug(f"Generated speech: {output_path}")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Piper command timed out")
            raise RuntimeError("Speech generation timed out")
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
