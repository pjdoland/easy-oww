"""
Audio recording functionality for wake word samples
"""
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, Tuple
from scipy.io import wavfile
from easy_oww.utils.logger import get_logger

logger = get_logger()


class AudioRecorder:
    """Records audio samples from microphone"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = 'int16'
    ):
        """
        Initialize audio recorder

        Args:
            sample_rate: Sample rate in Hz (default: 16000 for OpenWakeWord)
            channels: Number of audio channels (default: 1 for mono)
            dtype: Data type for audio samples (default: int16)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.recording = False
        self._audio_data = []

    def list_devices(self) -> list:
        """
        List available audio input devices

        Returns:
            List of available input devices
        """
        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })

        return input_devices

    def get_default_device(self) -> Optional[dict]:
        """
        Get default input device

        Returns:
            Default input device info or None
        """
        try:
            device_idx = sd.default.device[0]
            if device_idx is None:
                return None

            device = sd.query_devices(device_idx)
            return {
                'index': device_idx,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate']
            }
        except Exception as e:
            logger.error(f"Failed to get default device: {e}")
            return None

    def record_duration(
        self,
        duration: float,
        device: Optional[int] = None
    ) -> np.ndarray:
        """
        Record audio for a specific duration

        Args:
            duration: Recording duration in seconds
            device: Device index (None for default)

        Returns:
            Recorded audio as numpy array

        Raises:
            RuntimeError: If recording fails
        """
        try:
            logger.debug(f"Recording {duration}s from device {device}")

            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=device
            )
            sd.wait()

            logger.debug(f"Recorded {len(audio)} samples")
            return audio.flatten()

        except Exception as e:
            logger.error(f"Recording failed: {e}")
            raise RuntimeError(f"Failed to record audio: {e}")

    def record_until_stopped(
        self,
        device: Optional[int] = None,
        callback=None
    ) -> np.ndarray:
        """
        Record audio until stop() is called

        Args:
            device: Device index (None for default)
            callback: Optional callback function called periodically with audio chunk

        Returns:
            Recorded audio as numpy array
        """
        self._audio_data = []
        self.recording = True

        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")

            if self.recording:
                self._audio_data.append(indata.copy())
                if callback:
                    callback(indata)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=device,
                callback=audio_callback
            ):
                while self.recording:
                    sd.sleep(100)

        except Exception as e:
            logger.error(f"Recording failed: {e}")
            raise RuntimeError(f"Failed to record audio: {e}")

        # Combine all chunks
        if self._audio_data:
            audio = np.concatenate(self._audio_data, axis=0)
            return audio.flatten()
        else:
            return np.array([], dtype=self.dtype)

    def stop(self):
        """Stop recording"""
        self.recording = False

    def save_wav(
        self,
        audio: np.ndarray,
        output_path: Path
    ):
        """
        Save audio to WAV file

        Args:
            audio: Audio data as numpy array
            output_path: Output file path

        Raises:
            RuntimeError: If saving fails
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure correct data type
            if audio.dtype != np.int16:
                # Normalize to [-1, 1] then scale to int16
                audio_float = audio.astype(np.float32)
                audio_float = audio_float / np.max(np.abs(audio_float))
                audio = (audio_float * 32767).astype(np.int16)

            wavfile.write(
                str(output_path),
                self.sample_rate,
                audio
            )

            logger.debug(f"Saved audio to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise RuntimeError(f"Failed to save audio: {e}")

    def get_audio_level(self, audio: np.ndarray) -> float:
        """
        Calculate audio level (RMS) in decibels

        Args:
            audio: Audio data as numpy array

        Returns:
            Audio level in dB
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))

        # Convert to dB (reference to full scale)
        if rms > 0:
            db = 20 * np.log10(rms / 32768.0)
        else:
            db = -96.0  # Silence threshold

        return db

    def playback_audio(self, audio: np.ndarray, device: Optional[int] = None):
        """
        Playback recorded audio

        Args:
            audio: Audio data as numpy array (mono)
            device: Output device index (None for default)

        Raises:
            RuntimeError: If playback fails
        """
        try:
            logger.debug(f"Playing back {len(audio)} samples")

            # Ensure correct data type for playback
            if audio.dtype != np.int16:
                audio_int16 = audio.astype(np.int16)
            else:
                audio_int16 = audio

            # Convert mono to stereo for playback compatibility
            # Many output devices don't support mono playback
            if len(audio_int16.shape) == 1:
                # Duplicate mono channel to create stereo
                audio_stereo = np.column_stack((audio_int16, audio_int16))
            else:
                audio_stereo = audio_int16

            # Play audio (stereo, 2 channels)
            sd.play(audio_stereo, samplerate=self.sample_rate, device=device)
            sd.wait()

            logger.debug("Playback complete")

        except Exception as e:
            logger.error(f"Playback failed: {e}")
            raise RuntimeError(f"Failed to play audio: {e}")

    def test_microphone(
        self,
        duration: float = 2.0,
        device: Optional[int] = None
    ) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        Test microphone by recording a short sample

        Args:
            duration: Test duration in seconds
            device: Device index (None for default)

        Returns:
            Tuple of (success, message, audio_data)
        """
        try:
            audio = self.record_duration(duration, device)

            # Check if we got any audio
            if len(audio) == 0:
                return False, "No audio captured", None

            # Check audio level
            level_db = self.get_audio_level(audio)

            if level_db < -60:
                return False, f"Audio level too low ({level_db:.1f} dB). Check microphone volume.", audio
            elif level_db > -10:
                return False, f"Audio level too high ({level_db:.1f} dB). Risk of clipping.", audio
            else:
                return True, f"Microphone working ({level_db:.1f} dB)", audio

        except Exception as e:
            return False, f"Microphone test failed: {e}", None
