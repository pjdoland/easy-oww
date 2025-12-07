"""
Voice model downloader for Piper TTS
"""
import requests
import json
from pathlib import Path
from typing import List, Dict, Optional
from rich.progress import Progress, DownloadColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from easy_oww.utils.logger import get_logger

logger = get_logger()
console = Console()


class VoiceDownloader:
    """Downloads and manages Piper voice models"""

    # Piper voices repository
    VOICES_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

    # Curated list of recommended voices for wake word training
    RECOMMENDED_VOICES = {
        'en_US': [
            {
                'name': 'en_US-lessac-medium',
                'path': 'en/en_US/lessac/medium',
                'language': 'English (US)',
                'quality': 'medium',
                'description': 'Clear American English, good for training',
                'size_mb': 63
            },
            {
                'name': 'en_US-amy-medium',
                'path': 'en/en_US/amy/medium',
                'language': 'English (US)',
                'quality': 'medium',
                'description': 'Female American English voice',
                'size_mb': 63
            },
            {
                'name': 'en_US-joe-medium',
                'path': 'en/en_US/joe/medium',
                'language': 'English (US)',
                'quality': 'medium',
                'description': 'Male American English voice',
                'size_mb': 63
            },
            {
                'name': 'en_US-lessac-high',
                'path': 'en/en_US/lessac/high',
                'language': 'English (US)',
                'quality': 'high',
                'description': 'High quality Lessac voice',
                'size_mb': 116
            },
        ],
        'en_GB': [
            {
                'name': 'en_GB-alan-medium',
                'path': 'en/en_GB/alan/medium',
                'language': 'English (GB)',
                'quality': 'medium',
                'description': 'British English male voice',
                'size_mb': 63
            },
        ],
        'es_ES': [
            {
                'name': 'es_ES-mls_9972-low',
                'path': 'es/es_ES/mls_9972/low',
                'language': 'Spanish (Spain)',
                'quality': 'low',
                'description': 'Spanish voice',
                'size_mb': 18
            },
        ],
        'fr_FR': [
            {
                'name': 'fr_FR-mls_1840-low',
                'path': 'fr/fr_FR/mls_1840/low',
                'language': 'French (France)',
                'quality': 'low',
                'description': 'French voice',
                'size_mb': 18
            },
        ],
        'de_DE': [
            {
                'name': 'de_DE-thorsten-medium',
                'path': 'de/de_DE/thorsten/medium',
                'language': 'German',
                'quality': 'medium',
                'description': 'German male voice',
                'size_mb': 63
            },
        ],
    }

    def __init__(self, voices_dir: Path):
        """
        Initialize voice downloader

        Args:
            voices_dir: Directory to store voice models
        """
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def get_voice_url(self, voice_name: str, voice_path: Optional[str] = None) -> tuple[str, str]:
        """
        Get URLs for voice model and config

        Args:
            voice_name: Name of voice (e.g., 'en_US-lessac-medium')
            voice_path: Optional path in repository (e.g., 'en/en_US/lessac/medium')

        Returns:
            Tuple of (model_url, config_url)
        """
        # Voice files are organized as: language/code/voice/quality/model.onnx
        if voice_path:
            model_url = f"{self.VOICES_BASE_URL}/{voice_path}/{voice_name}.onnx"
            config_url = f"{self.VOICES_BASE_URL}/{voice_path}/{voice_name}.onnx.json"
        else:
            # Fallback to flat structure (legacy)
            model_url = f"{self.VOICES_BASE_URL}/{voice_name}.onnx"
            config_url = f"{self.VOICES_BASE_URL}/{voice_name}.onnx.json"

        return model_url, config_url

    def download_voice(
        self,
        voice_name: str,
        voice_path: Optional[str] = None,
        show_progress: bool = True
    ) -> tuple[Path, Path]:
        """
        Download voice model and config

        Args:
            voice_name: Name of voice to download
            voice_path: Optional path in repository
            show_progress: Show download progress bar

        Returns:
            Tuple of (model_path, config_path)

        Raises:
            RuntimeError: If download fails
        """
        model_path = self.voices_dir / f"{voice_name}.onnx"
        config_path = self.voices_dir / f"{voice_name}.onnx.json"

        # Check if already downloaded
        if model_path.exists() and config_path.exists():
            logger.info(f"Voice already downloaded: {voice_name}")
            return model_path, config_path

        model_url, config_url = self.get_voice_url(voice_name, voice_path)

        try:
            # Download model
            logger.info(f"Downloading voice model: {voice_name}")

            if show_progress:
                self._download_with_progress(model_url, model_path, f"Model: {voice_name}")
            else:
                self._download_file(model_url, model_path)

            # Download config
            logger.debug(f"Downloading config for: {voice_name}")
            self._download_file(config_url, config_path)

            logger.info(f"Successfully downloaded: {voice_name}")
            return model_path, config_path

        except Exception as e:
            # Clean up partial downloads
            if model_path.exists():
                model_path.unlink()
            if config_path.exists():
                config_path.unlink()

            logger.error(f"Failed to download voice {voice_name}: {e}")
            raise RuntimeError(f"Failed to download voice: {e}")

    def _download_file(self, url: str, output_path: Path):
        """
        Download file without progress display

        Args:
            url: URL to download
            output_path: Output file path

        Raises:
            RuntimeError: If download fails
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def _download_with_progress(self, url: str, output_path: Path, description: str):
        """
        Download file with progress bar

        Args:
            url: URL to download
            output_path: Output file path
            description: Progress bar description

        Raises:
            RuntimeError: If download fails
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Progress(
            *Progress.get_default_columns(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(description, total=total_size)

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    def download_recommended_voices(
        self,
        language: str = 'en_US',
        max_voices: int = 2
    ) -> List[Path]:
        """
        Download recommended voices for a language

        Args:
            language: Language code (e.g., 'en_US')
            max_voices: Maximum number of voices to download

        Returns:
            List of downloaded model paths
        """
        if language not in self.RECOMMENDED_VOICES:
            raise ValueError(f"No recommended voices for language: {language}")

        voices = self.RECOMMENDED_VOICES[language][:max_voices]
        downloaded = []

        console.print(f"\n[bold]Downloading {len(voices)} voice(s) for {language}...[/bold]")

        for voice_info in voices:
            voice_name = voice_info['name']
            voice_path = voice_info.get('path')
            console.print(f"\n[cyan]Voice:[/cyan] {voice_info['description']} ({voice_info['size_mb']} MB)")

            try:
                model_path, _ = self.download_voice(voice_name, voice_path=voice_path, show_progress=True)
                downloaded.append(model_path)
            except Exception as e:
                console.print(f"[red]Failed to download {voice_name}: {e}[/red]")
                continue

        return downloaded

    def list_available_voices(self, language: Optional[str] = None) -> List[Dict]:
        """
        List available recommended voices

        Args:
            language: Optional language filter

        Returns:
            List of voice information dictionaries
        """
        if language:
            if language in self.RECOMMENDED_VOICES:
                return self.RECOMMENDED_VOICES[language]
            else:
                return []
        else:
            # Return all voices
            all_voices = []
            for lang_voices in self.RECOMMENDED_VOICES.values():
                all_voices.extend(lang_voices)
            return all_voices

    def list_installed_voices(self) -> List[Dict]:
        """
        List locally installed voices

        Returns:
            List of installed voice information
        """
        installed = []

        for model_file in self.voices_dir.glob('*.onnx'):
            config_file = model_file.with_suffix('.onnx.json')

            voice_info = {
                'name': model_file.stem,
                'path': model_file,
                'size_mb': model_file.stat().st_size / (1024 * 1024)
            }

            # Read config if available
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    voice_info['language'] = config.get('language', {}).get('code', 'unknown')
                    voice_info['sample_rate'] = config.get('audio', {}).get('sample_rate', 22050)
                except Exception as e:
                    logger.warning(f"Failed to read config {config_file}: {e}")

            installed.append(voice_info)

        return installed

    def is_voice_installed(self, voice_name: str) -> bool:
        """
        Check if voice is installed

        Args:
            voice_name: Name of voice

        Returns:
            True if voice is installed
        """
        model_path = self.voices_dir / f"{voice_name}.onnx"
        config_path = self.voices_dir / f"{voice_name}.onnx.json"

        return model_path.exists() and config_path.exists()

    def remove_voice(self, voice_name: str) -> bool:
        """
        Remove installed voice

        Args:
            voice_name: Name of voice to remove

        Returns:
            True if removed successfully
        """
        model_path = self.voices_dir / f"{voice_name}.onnx"
        config_path = self.voices_dir / f"{voice_name}.onnx.json"

        removed = False

        if model_path.exists():
            model_path.unlink()
            removed = True

        if config_path.exists():
            config_path.unlink()
            removed = True

        if removed:
            logger.info(f"Removed voice: {voice_name}")

        return removed

    def get_total_size(self, voices: List[str]) -> int:
        """
        Calculate total download size for voices

        Args:
            voices: List of voice names

        Returns:
            Total size in bytes
        """
        total = 0

        for voice_name in voices:
            # Find voice in recommended list
            for lang_voices in self.RECOMMENDED_VOICES.values():
                for voice_info in lang_voices:
                    if voice_info['name'] == voice_name:
                        total += voice_info['size_mb'] * 1024 * 1024
                        break

        return total

    def verify_voice(self, voice_name: str) -> tuple[bool, str]:
        """
        Verify voice installation

        Args:
            voice_name: Name of voice

        Returns:
            Tuple of (success, message)
        """
        model_path = self.voices_dir / f"{voice_name}.onnx"
        config_path = self.voices_dir / f"{voice_name}.onnx.json"

        if not model_path.exists():
            return False, "Model file missing"

        if not config_path.exists():
            return False, "Config file missing"

        # Check file sizes
        if model_path.stat().st_size < 1024:
            return False, "Model file appears corrupted (too small)"

        if config_path.stat().st_size < 10:
            return False, "Config file appears corrupted (too small)"

        # Try to parse config
        try:
            with open(config_path, 'r') as f:
                json.load(f)
        except Exception as e:
            return False, f"Config file is invalid: {e}"

        return True, "Voice installation verified"
