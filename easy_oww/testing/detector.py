"""
Real-time wake word detection for model testing
"""
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, List
import time
from collections import deque
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from easy_oww.audio.recorder import AudioRecorder
from easy_oww.utils.logger import get_logger

logger = get_logger()
console = Console()


class ModelDetector:
    """Real-time wake word detector using ONNX model"""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        detection_threshold: float = 0.5,
        debounce_time: float = 2.0
    ):
        """
        Initialize model detector

        Args:
            model_path: Path to ONNX model file
            sample_rate: Audio sample rate
            chunk_duration_ms: Audio chunk duration in milliseconds
            detection_threshold: Detection confidence threshold (0-1)
            debounce_time: Minimum time (seconds) between detections to count as separate (default: 2.0)
        """
        self.model_path = Path(model_path) if model_path else None
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.detection_threshold = detection_threshold
        self.debounce_time = debounce_time

        # Audio buffer settings
        self.chunk_samples = int((chunk_duration_ms / 1000.0) * sample_rate)
        self.buffer_duration_ms = 1500  # 1.5 seconds buffer
        self.buffer_samples = int((self.buffer_duration_ms / 1000.0) * sample_rate)

        # Audio buffer
        self.audio_buffer = deque(maxlen=self.buffer_samples)

        # Model (lazy loaded)
        self.model = None
        self.model_loaded = False

        # Recorder
        self.recorder = AudioRecorder(sample_rate=sample_rate)

        # Detection state
        self.is_running = False
        self.detections = []
        self.last_detection_time = 0.0  # For debouncing

    def load_model(self):
        """
        Load ONNX model

        Raises:
            RuntimeError: If model loading fails
        """
        if self.model_loaded:
            return

        if not self.model_path or not self.model_path.exists():
            raise RuntimeError(f"Model not found: {self.model_path}")

        try:
            # Try to import OpenWakeWord
            try:
                # Suppress TFLite runtime warning (we use ONNX models only)
                import warnings
                import logging
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*tflite runtime.*')
                    # Temporarily suppress OpenWakeWord's root logger warnings
                    logging.getLogger().setLevel(logging.ERROR)

                    from openwakeword.model import Model as OWWModel
                    self.model = OWWModel(wakeword_models=[str(self.model_path)])

                    # Restore logging level
                    logging.getLogger().setLevel(logging.WARNING)

                self.model_type = 'openwakeword'
                logger.info(f"Loaded OpenWakeWord model from {self.model_path}")
            except ImportError:
                # Fallback to basic ONNX runtime
                import onnxruntime as ort
                self.model = ort.InferenceSession(str(self.model_path))
                self.model_type = 'onnx'
                logger.info(f"Loaded ONNX model from {self.model_path}")

            self.model_loaded = True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def predict(self, audio: np.ndarray) -> float:
        """
        Run model prediction on audio

        Args:
            audio: Audio data (int16)

        Returns:
            Detection confidence score (0-1)
        """
        if not self.model_loaded:
            self.load_model()

        try:
            if self.model_type == 'openwakeword':
                # OpenWakeWord model
                # Use predict_clip for discrete audio clips instead of predict
                # predict_clip processes audio in chunks and returns frame-level predictions
                predictions = self.model.predict_clip(audio, padding=1, chunk_size=1280)

                # Extract scores from all frames
                model_name = list(self.model.models.keys())[0]
                scores = [pred[model_name] for pred in predictions if model_name in pred]

                # Return maximum score across all frames
                if scores:
                    return float(max(scores))
                else:
                    return 0.0

            else:
                # Basic ONNX model
                # Prepare input
                audio_float = audio.astype(np.float32) / 32768.0
                audio_input = audio_float.reshape(1, -1)

                # Run inference
                input_name = self.model.get_inputs()[0].name
                output_name = self.model.get_outputs()[0].name

                result = self.model.run([output_name], {input_name: audio_input})
                score = float(result[0][0])

                return score

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict]:
        """
        Process audio chunk and detect wake word

        Args:
            audio_chunk: Audio chunk data

        Returns:
            Detection result dict or None
        """
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)

        # Check if buffer is full enough
        if len(self.audio_buffer) < self.buffer_samples:
            return None

        # Get audio window
        audio_window = np.array(list(self.audio_buffer))

        # Run prediction
        score = self.predict(audio_window)

        # Check for detection
        if score >= self.detection_threshold:
            current_time = time.time()

            # Debouncing: only count as new detection if enough time has passed
            if current_time - self.last_detection_time >= self.debounce_time:
                self.last_detection_time = current_time
                detection = {
                    'timestamp': current_time,
                    'score': score,
                    'audio': audio_window.copy()
                }
                return detection

        return None

    def start_detection(
        self,
        device: Optional[int] = None,
        duration: Optional[float] = None,
        on_detection: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Start real-time detection

        Args:
            device: Audio device index
            duration: Detection duration in seconds (None for continuous)
            on_detection: Callback function for detections

        Returns:
            List of detection results
        """
        if not self.model_loaded:
            self.load_model()

        self.is_running = True
        self.detections = []
        self.audio_buffer.clear()

        start_time = time.time()

        logger.info("Starting real-time detection...")

        try:
            # Start audio stream
            import sounddevice as sd

            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")

                if not self.is_running:
                    return

                # Convert to int16
                audio_chunk = (indata * 32767).astype(np.int16).flatten()

                # Process chunk
                detection = self.process_audio_chunk(audio_chunk)

                if detection:
                    self.detections.append(detection)

                    # Call callback
                    if on_detection:
                        on_detection(detection)

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                device=device,
                callback=audio_callback,
                blocksize=self.chunk_samples
            ):
                # Run until duration expires or stopped
                while self.is_running:
                    if duration and (time.time() - start_time) >= duration:
                        break

                    time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
        finally:
            self.is_running = False

        return self.detections

    def stop_detection(self):
        """Stop detection"""
        self.is_running = False

    def test_model(
        self,
        test_audio_path: Path,
        ground_truth: bool
    ) -> Dict:
        """
        Test model on audio file

        Args:
            test_audio_path: Path to test audio file
            ground_truth: True if audio contains wake word

        Returns:
            Test result dictionary
        """
        from scipy.io import wavfile

        if not self.model_loaded:
            self.load_model()

        # Load audio
        sample_rate, audio = wavfile.read(str(test_audio_path))

        # Resample if needed
        if sample_rate != self.sample_rate:
            from scipy import signal
            num_samples = int(len(audio) * self.sample_rate / sample_rate)
            audio = signal.resample(audio, num_samples).astype(np.int16)

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.int16)

        # Pad audio to 2 seconds if needed (OpenWakeWord expects 32000 samples for proper feature extraction)
        min_samples = 32000  # 2 seconds at 16kHz
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')

        # Get prediction
        score = self.predict(audio)
        predicted = score >= self.detection_threshold

        # Calculate result
        result = {
            'file': test_audio_path.name,
            'score': score,
            'predicted': predicted,
            'ground_truth': ground_truth,
            'correct': predicted == ground_truth,
            'true_positive': predicted and ground_truth,
            'true_negative': not predicted and not ground_truth,
            'false_positive': predicted and not ground_truth,
            'false_negative': not predicted and ground_truth
        }

        return result


def run_realtime_test(
    model_path: Path,
    duration: int = 60,
    device: Optional[int] = None,
    threshold: float = 0.5
):
    """
    Run real-time detection test with visualization

    Args:
        model_path: Path to model file
        duration: Test duration in seconds
        device: Audio device index
        threshold: Detection threshold
    """
    console.print(Panel.fit(
        f"[bold cyan]Real-Time Wake Word Detection Test[/bold cyan]\n\n"
        f"Model: {model_path.name}\n"
        f"Duration: {duration} seconds\n"
        f"Threshold: {threshold}\n\n"
        f"[yellow]Speak your wake word to test detection[/yellow]",
        title="Model Testing"
    ))

    # Initialize detector
    detector = ModelDetector(
        model_path=model_path,
        detection_threshold=threshold
    )

    # Load model
    console.print("\n[bold]Loading model...[/bold]")
    try:
        detector.load_model()
        console.print("[green]✓[/green] Model loaded successfully")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return

    # Detection tracking
    detection_count = 0
    detection_times = []

    def on_detection(detection):
        nonlocal detection_count, detection_times
        detection_count += 1
        detection_times.append(detection['timestamp'])

    # Create live display
    console.print("\n[bold cyan]Starting detection...[/bold cyan]")
    console.print("Press Ctrl+C to stop\n")

    start_time = time.time()

    with Live(console=console, refresh_per_second=4) as live:
        def update_display():
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)

            # Create status table
            table = Table(title="Detection Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Elapsed Time", f"{elapsed:.1f}s")
            table.add_row("Remaining Time", f"{remaining:.1f}s")
            table.add_row("Detections", f"[green]{detection_count}[/green]")

            if detection_count > 0:
                last_detection = time.time() - detection_times[-1]
                table.add_row("Last Detection", f"{last_detection:.1f}s ago")

            live.update(table)

        try:
            # Start detection in background
            import threading

            stop_event = threading.Event()

            def detection_thread():
                detector.start_detection(
                    device=device,
                    duration=duration,
                    on_detection=on_detection
                )
                stop_event.set()

            thread = threading.Thread(target=detection_thread)
            thread.start()

            # Update display
            while not stop_event.is_set():
                update_display()
                time.sleep(0.25)

            thread.join()

        except KeyboardInterrupt:
            detector.stop_detection()
            console.print("\n[yellow]Test interrupted[/yellow]")

    # Display results
    console.print("\n" + "=" * 60)
    console.print(Panel.fit(
        f"[bold green]Test Complete![/bold green]\n\n"
        f"Total Detections: {detection_count}\n"
        f"Test Duration: {time.time() - start_time:.1f}s\n"
        f"Detection Rate: {detection_count / duration:.2f} detections/second",
        title="Results"
    ))

    if detection_count > 0:
        console.print("\n[bold]Detection Times:[/bold]")
        for i, dt in enumerate(detection_times, 1):
            relative_time = dt - start_time
            console.print(f"  {i}. {relative_time:.2f}s")
    else:
        console.print("\n[yellow]No detections recorded[/yellow]")
        console.print("Try:")
        console.print("  • Speaking louder")
        console.print("  • Speaking closer to microphone")
        console.print("  • Lowering detection threshold")
