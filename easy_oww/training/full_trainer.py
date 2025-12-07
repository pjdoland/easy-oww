"""
Full model training using OpenWakeWord's training pipeline
"""

# IMPORTANT: Apply monkey-patch FIRST before any other imports
# This fixes compatibility between speechbrain and modern torchaudio
try:
    import torchaudio

    # Patch 1: Add list_audio_backends for speechbrain
    if not hasattr(torchaudio, 'list_audio_backends'):
        def list_audio_backends():
            # Return empty list since backends are auto-detected in torchaudio 2.x
            return []
        torchaudio.list_audio_backends = list_audio_backends

    # Patch 2: Ensure torchaudio.info is available for torch_audiomentations
    if not hasattr(torchaudio, 'info'):
        # Try to import from backend
        try:
            from torchaudio.backend.soundfile_backend import info
            torchaudio.info = info
        except ImportError:
            # Fallback: create info wrapper using soundfile directly
            import soundfile as sf
            from collections import namedtuple

            AudioMetaData = namedtuple('AudioMetaData', ['sample_rate', 'num_frames', 'num_channels', 'bits_per_sample', 'encoding'])

            def info(filepath):
                """Get audio file metadata using soundfile"""
                info_obj = sf.info(str(filepath))

                # Extract bits per sample from subtype string if available
                bits_per_sample = 16  # default
                if hasattr(info_obj, 'subtype'):
                    subtype = str(info_obj.subtype)
                    # Extract number from subtype like 'PCM_16' or 'PCM_24'
                    import re
                    match = re.search(r'(\d+)$', subtype)
                    if match:
                        bits_per_sample = int(match.group(1))

                return AudioMetaData(
                    sample_rate=info_obj.samplerate,
                    num_frames=info_obj.frames,
                    num_channels=info_obj.channels,
                    bits_per_sample=bits_per_sample,
                    encoding=str(info_obj.subtype) if hasattr(info_obj, 'subtype') else 'UNKNOWN'
                )

            torchaudio.info = info
except ImportError:
    pass

import os
import numpy as np
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from easy_oww.utils.logger import get_logger

logger = get_logger()
console = Console()


# Lazy imports to avoid dependency issues on module load
# These will be imported only when actually training
def _get_oww_imports():
    """Lazy import of OpenWakeWord training modules"""
    try:
        import torch
        from openwakeword.train import Model as OWWModel
        from openwakeword.data import augment_clips, mmap_batch_generator
        from openwakeword.utils import compute_features_from_generator, AudioFeatures
        return torch, OWWModel, augment_clips, mmap_batch_generator, compute_features_from_generator, AudioFeatures
    except Exception as e:
        raise ImportError(
            f"Failed to import OpenWakeWord training modules: {e}\n\n"
            "This might be due to incompatible dependency versions. "
            "The full model training requires many dependencies including:\n"
            "  - torch, torchaudio\n"
            "  - speechbrain\n"
            "  - audiomentations, torch-audiomentations\n"
            "  - librosa, resampy\n\n"
            "Try reinstalling with compatible versions or use a different training approach."
        )


class FullModelTrainer:
    """Trains a complete wake word model from scratch using OpenWakeWord's pipeline"""

    def __init__(
        self,
        project_name: str,
        wake_word: str,
        clips_dir: Path,
        output_dir: Path,
        workspace_dir: Path
    ):
        """
        Initialize full model trainer

        Args:
            project_name: Name of the project
            wake_word: Wake word phrase
            clips_dir: Directory containing audio clips
            output_dir: Directory to save trained model
            workspace_dir: Workspace directory (for datasets)
        """
        self.project_name = project_name
        self.wake_word = wake_word
        self.clips_dir = Path(clips_dir)
        self.output_dir = Path(output_dir)
        self.workspace_dir = Path(workspace_dir)

        # Paths
        self.positive_dir = self.clips_dir / 'positive'
        self.positive_aug_dir = self.clips_dir / 'positive_augmented'
        self.negative_dir = self.clips_dir / 'negative'
        self.features_dir = self.clips_dir / 'features'
        self.features_dir.mkdir(exist_ok=True)

        # Dataset paths
        self.rir_dir = self.workspace_dir / 'datasets' / 'rir'
        self.noise_dir = self.workspace_dir / 'datasets' / 'fsd50k'

    def train(
        self,
        model_type: str = "dnn",
        layer_size: int = 128,
        steps: int = 5000,
        target_fp_per_hour: float = 0.5,
        augmentation_rounds: int = 3,
        batch_size: int = 128,
        max_negative_weight: int = 25,
        force: bool = False
    ) -> Path:
        """
        Train a full wake word model

        Args:
            model_type: Model architecture ("dnn" or "rnn")
            layer_size: Hidden layer dimension
            steps: Training steps
            target_fp_per_hour: Target false positives per hour
            augmentation_rounds: Number of augmentation rounds per clip
            batch_size: Batch size for augmentation

        Returns:
            Path to saved ONNX model
        """
        console.print("\n[bold cyan]Full Model Training Pipeline[/bold cyan]")

        # Step 1: Prepare data
        console.print("\n[bold]Step 1: Preparing data[/bold]")
        positive_clips, negative_clips = self._prepare_data()

        # Step 2: Generate adversarial negatives
        console.print("\n[bold]Step 2: Generating adversarial negatives[/bold]")
        adversarial_clips = self._generate_adversarial_negatives()
        all_negative_clips = negative_clips + adversarial_clips

        # Step 3: Extract features
        console.print("\n[bold]Step 3: Extracting features[/bold]")
        total_length = self._determine_clip_length(positive_clips)
        self._extract_features(
            positive_clips,
            all_negative_clips,
            total_length,
            augmentation_rounds,
            batch_size,
            force
        )

        # Step 4: Train model
        console.print("\n[bold]Step 4: Training model[/bold]")
        model_path = self._train_model(
            total_length,
            model_type,
            layer_size,
            steps,
            target_fp_per_hour,
            batch_size,
            max_negative_weight
        )

        console.print(f"\n[green]✓[/green] Model training complete: {model_path}")
        return model_path

    def _prepare_data(self):
        """Prepare positive and negative clips"""
        # Collect positive clips (real + augmented)
        positive_clips = []
        if self.positive_dir.exists():
            positive_clips.extend(list(self.positive_dir.glob('*.wav')))
        if self.positive_aug_dir.exists():
            positive_clips.extend(list(self.positive_aug_dir.glob('*.wav')))

        # Collect negative clips (regular + adversarial)
        negative_clips = []
        if self.negative_dir.exists():
            negative_clips.extend(list(self.negative_dir.glob('*.wav')))

        # Add adversarial negatives
        adversarial_dir = self.clips_dir / 'adversarial_negative'
        if adversarial_dir.exists():
            adversarial_clips = list(adversarial_dir.glob('*.wav'))
            negative_clips.extend(adversarial_clips)
            console.print(f"  Added {len(adversarial_clips)} adversarial negatives")

        console.print(f"  Positive clips: {len(positive_clips)}")
        console.print(f"  Negative clips: {len(negative_clips)}")

        if len(positive_clips) == 0:
            raise RuntimeError("No positive clips found!")

        if len(negative_clips) == 0:
            logger.warning("No negative clips found - model will have high false positive rate")

        return [str(p) for p in positive_clips], [str(n) for n in negative_clips]

    def _generate_adversarial_negatives(self):
        """Generate adversarial negative samples using TTS"""
        # Import only what we need for this specific function
        try:
            from openwakeword.data import generate_adversarial_texts
        except ImportError as e:
            logger.error(f"Failed to import generate_adversarial_texts: {e}")
            console.print("[yellow]⚠[/yellow] Skipping adversarial generation due to import error")
            return []

        from easy_oww.tts import PiperTTS

        adversarial_dir = self.clips_dir / 'adversarial_negative'
        adversarial_dir.mkdir(exist_ok=True)

        # Check if we already have adversarial samples
        existing = list(adversarial_dir.glob('*.wav'))
        if len(existing) >= 500:
            console.print(f"  Using {len(existing)} existing adversarial samples")
            return [str(p) for p in existing]

        # Generate adversarial texts (phrases similar to wake word)
        console.print(f"  Generating adversarial phrases for '{self.wake_word}'...")
        adversarial_texts = generate_adversarial_texts(
            input_text=self.wake_word,
            N=1000,
            include_partial_phrase=1.0,
            include_input_words=0.5
        )

        # Generate audio using Piper TTS
        console.print(f"  Synthesizing {len(adversarial_texts)} adversarial samples...")
        piper = PiperTTS(self.workspace_dir)

        # Get available voices
        voices = []
        for voice_name in ['en_US-lessac-medium', 'en_US-amy-medium',
                          'en_US-joe-medium', 'en_GB-alan-medium']:
            voice_path = piper.get_voice(voice_name)
            if voice_path:
                voices.append(voice_path)

        if not voices:
            logger.warning("No voices available for adversarial generation")
            return []

        # Generate clips
        adversarial_clips = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task(f"Generating adversarial clips", total=len(adversarial_texts[:1000]))

            for i, text in enumerate(adversarial_texts[:1000]):  # Limit to 1000
                try:
                    voice = voices[i % len(voices)]
                    output_path = adversarial_dir / f"adversarial_{i:04d}.wav"

                    # generate_speech writes directly to file and returns boolean
                    success = piper.generate_speech(text, voice, output_path, sample_rate=16000)
                    if success:
                        adversarial_clips.append(str(output_path))

                    progress.update(task, advance=1)
                except Exception as e:
                    logger.warning(f"Failed to generate adversarial clip {i}: {e}")

        console.print(f"  Generated {len(adversarial_clips)} adversarial samples")
        return adversarial_clips

    def _determine_clip_length(self, positive_clips):
        """Determine appropriate clip length based on samples"""
        import scipy.io.wavfile as wavfile

        durations = []
        for clip in positive_clips[:50]:  # Sample 50 clips
            try:
                sr, data = wavfile.read(clip)
                durations.append(len(data))
            except:
                pass

        if not durations:
            total_length = 32000  # Default 2 seconds
        else:
            median_duration = int(np.median(durations))
            total_length = int(round(median_duration / 1000) * 1000) + 12000  # Add 750ms buffer

            # Ensure minimum of 2 seconds
            if total_length < 32000:
                total_length = 32000

        console.print(f"  Using clip length: {total_length} samples ({total_length/16000:.2f}s)")
        return total_length

    def _extract_features(
        self,
        positive_clips,
        negative_clips,
        total_length,
        augmentation_rounds,
        batch_size,
        force=False
    ):
        """Extract OpenWakeWord features from clips"""
        # Check if features already exist
        required_features = [
            self.features_dir / 'positive_train.npy',
            self.features_dir / 'positive_test.npy',
            self.features_dir / 'negative_train.npy',
            self.features_dir / 'negative_test.npy'
        ]

        if not force and all(f.exists() for f in required_features):
            console.print("[green]✓[/green] Features already extracted, skipping...")
            for feat in required_features:
                console.print(f"  Found: {feat.name}")
            return

        if force:
            console.print("[yellow]⚠[/yellow] Force flag set - regenerating features...")

        # Get lazy imports
        torch, _, augment_clips, _, compute_features_from_generator, _ = _get_oww_imports()

        # Get augmentation data
        rir_paths = []
        if self.rir_dir.exists():
            rir_paths = [str(p) for p in self.rir_dir.glob('*.wav')]

        noise_paths = []
        if self.noise_dir.exists():
            noise_paths = [str(p) for p in self.noise_dir.glob('**/*.wav')]

        console.print(f"  RIR files: {len(rir_paths)}")
        console.print(f"  Noise files: {len(noise_paths)}")

        # Split into train and test
        split_idx_pos = int(len(positive_clips) * 0.8)
        split_idx_neg = int(len(negative_clips) * 0.8)

        pos_train = positive_clips[:split_idx_pos] * augmentation_rounds
        pos_test = positive_clips[split_idx_pos:] * augmentation_rounds
        neg_train = negative_clips[:split_idx_neg] * augmentation_rounds
        neg_test = negative_clips[split_idx_neg:] * augmentation_rounds

        console.print(f"  Training: {len(pos_train)} positive, {len(neg_train)} negative")
        console.print(f"  Testing: {len(pos_test)} positive, {len(neg_test)} negative")

        # Create augmentation generators
        # Note: augment_clips doesn't accept None for paths, so we skip those params if empty
        console.print("\n  Creating augmentation generators...")

        augment_kwargs = {
            'total_length': total_length,
            'batch_size': batch_size
        }

        if noise_paths:
            augment_kwargs['background_clip_paths'] = noise_paths
        if rir_paths:
            augment_kwargs['RIR_paths'] = rir_paths

        pos_train_gen = augment_clips(pos_train, **augment_kwargs)
        pos_test_gen = augment_clips(pos_test, **augment_kwargs)
        neg_train_gen = augment_clips(neg_train, **augment_kwargs)
        neg_test_gen = augment_clips(neg_test, **augment_kwargs)

        # Extract features
        console.print("\n  Extracting features (this may take a while)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n_cpus = os.cpu_count() // 2 if os.cpu_count() else 1

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn()
        ) as progress:
            task = progress.add_task("Extracting features", total=4)

            compute_features_from_generator(
                pos_train_gen,
                n_total=len(pos_train),
                clip_duration=total_length,
                output_file=str(self.features_dir / "positive_train.npy"),
                device=device,
                ncpu=n_cpus if device == "cpu" else 1
            )
            progress.update(task, advance=1, description="Positive train features extracted")

            compute_features_from_generator(
                neg_train_gen,
                n_total=len(neg_train),
                clip_duration=total_length,
                output_file=str(self.features_dir / "negative_train.npy"),
                device=device,
                ncpu=n_cpus if device == "cpu" else 1
            )
            progress.update(task, advance=1, description="Negative train features extracted")

            compute_features_from_generator(
                pos_test_gen,
                n_total=len(pos_test),
                clip_duration=total_length,
                output_file=str(self.features_dir / "positive_test.npy"),
                device=device,
                ncpu=n_cpus if device == "cpu" else 1
            )
            progress.update(task, advance=1, description="Positive test features extracted")

            compute_features_from_generator(
                neg_test_gen,
                n_total=len(neg_test),
                clip_duration=total_length,
                output_file=str(self.features_dir / "negative_test.npy"),
                device=device,
                ncpu=n_cpus if device == "cpu" else 1
            )
            progress.update(task, advance=1, description="All features extracted")

        console.print("  [green]✓[/green] Feature extraction complete")

    def _train_model(
        self,
        total_length,
        model_type,
        layer_size,
        steps,
        target_fp_per_hour,
        batch_size,
        max_negative_weight
    ):
        """Train the wake word model"""
        # Get lazy imports
        torch, OWWModel, _, mmap_batch_generator, _, AudioFeatures = _get_oww_imports()

        # Initialize AudioFeatures to get input shape
        F = AudioFeatures(device='cpu')
        input_shape = F.get_embedding_shape(total_length // 16000)

        console.print(f"  Model type: {model_type}")
        console.print(f"  Input shape: {input_shape}")
        console.print(f"  Layer size: {layer_size}")

        # Create model
        oww = OWWModel(
            n_classes=1,
            input_shape=input_shape,
            model_type=model_type,
            layer_dim=layer_size,
            seconds_per_example=1280 * input_shape[0] / 16000
        )

        # Prepare data loaders
        console.print("\n  Preparing data loaders...")

        # IMPORTANT: OpenWakeWord's auto_train expects all data in memory or as DataLoaders
        # We can't use a generator directly. We need to either:
        # 1. Load all training data into memory (if it fits)
        # 2. Create a proper IterableDataset
        # Let's try loading training data into memory first

        console.print("\n  Loading training data into memory...")
        console.print("  [dim](This may take a moment for large datasets)[/dim]")

        # Load all training features
        train_pos_features = np.load(str(self.features_dir / "positive_train.npy"))
        train_neg_features = np.load(str(self.features_dir / "negative_train.npy"))

        console.print(f"  Loaded {len(train_pos_features)} positive and {len(train_neg_features)} negative training samples")

        # Combine and create labels
        X_train_data = np.vstack((train_pos_features, train_neg_features))
        y_train_data = np.hstack((
            np.ones(len(train_pos_features)),
            np.zeros(len(train_neg_features))
        )).astype(np.float32)

        # Create dataset
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train_data),
            torch.from_numpy(y_train_data)
        )

        # Create repeating DataLoader that loops indefinitely
        # OpenWakeWord's auto_train expects the DataLoader to provide data for the full number of steps
        def infinite_dataloader(dataset, batch_size):
            """Create a DataLoader that repeats indefinitely"""
            while True:
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True
                )
                for batch in dataloader:
                    yield batch

        X_train = infinite_dataloader(train_dataset, batch_size)

        # Validation data
        X_val_pos = np.load(str(self.features_dir / "positive_test.npy"))
        X_val_neg = np.load(str(self.features_dir / "negative_test.npy"))
        labels = np.hstack((np.ones(X_val_pos.shape[0]), np.zeros(X_val_neg.shape[0]))).astype(np.float32)

        X_val = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(np.vstack((X_val_pos, X_val_neg))),
                torch.from_numpy(labels)
            ),
            batch_size=len(labels)
        )

        # Use negative validation data for false positive estimation
        X_val_fp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_val_neg),
                torch.from_numpy(np.zeros(X_val_neg.shape[0]).astype(np.float32))
            ),
            batch_size=len(X_val_neg)
        )

        # Train model
        console.print(f"\n  Training for {steps} steps...")
        console.print(f"  Target: {target_fp_per_hour} false positives per hour")
        console.print(f"  Max negative weight: {max_negative_weight}")
        console.print(f"  Batch size: {batch_size}")

        best_model = oww.auto_train(
            X_train=X_train,
            X_val=X_val,
            false_positive_val_data=X_val_fp,
            steps=steps,
            max_negative_weight=max_negative_weight,
            target_fp_per_hour=target_fp_per_hour
        )

        # Validate model was actually trained
        console.print("\n  Validating trained model...")
        if best_model is None:
            raise RuntimeError("Training failed: auto_train returned None")

        # Check if model has actual weights (not all zeros)
        try:
            import torch
            has_nonzero_weights = False
            for name, param in best_model.named_parameters():
                if torch.any(param != 0):
                    has_nonzero_weights = True
                    break

            if not has_nonzero_weights:
                console.print("  [yellow]⚠ Warning: Model appears to have all-zero weights![/yellow]")
            else:
                console.print("  [green]✓ Model has non-zero weights[/green]")

        except Exception as e:
            console.print(f"  [yellow]⚠ Could not validate weights: {e}[/yellow]")

        # Export model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{self.project_name}.onnx"

        console.print(f"\n  Exporting model to: {output_path}")
        try:
            # Export manually with opset_version=17 to avoid LayerNormalization compatibility issues
            # OpenWakeWord's export_model uses opset 13 which doesn't support LayerNormalization
            import torch
            import copy

            model_to_save = copy.deepcopy(best_model).to("cpu")
            dummy_input = torch.rand(oww.input_shape)[None, ]

            console.print(f"  [dim]Using legacy ONNX exporter with opset 18[/dim]")

            # Put model in eval mode before export
            model_to_save.eval()

            # Use legacy ONNX exporter (more stable than dynamo-based exporter)
            with torch.no_grad():
                torch.onnx.export(
                    model_to_save,
                    dummy_input,
                    str(output_path),
                    opset_version=18,
                    input_names=['input'],
                    output_names=['output'],
                    export_params=True,
                    do_constant_folding=True,
                    dynamo=False  # Use legacy exporter, not the new dynamo-based one
                )

            # Verify export
            if output_path.exists():
                size_mb = output_path.stat().st_size / 1024 / 1024
                console.print(f"  [green]✓ Model exported: {size_mb:.2f} MB[/green]")

                if size_mb < 0.1:
                    console.print(f"  [red]✗ Warning: Model file is suspiciously small ({size_mb:.2f} MB)[/red]")
                    console.print(f"  [red]  This suggests the export failed[/red]")
                    raise RuntimeError("Model export produced an invalid file")

                # Try to load the model to verify it works
                try:
                    import onnxruntime as ort
                    sess = ort.InferenceSession(str(output_path))
                    console.print(f"  [green]✓ Model verified - successfully loaded with ONNX Runtime[/green]")
                except Exception as e:
                    console.print(f"  [yellow]⚠ Warning: Could not verify model with ONNX Runtime: {e}[/yellow]")
            else:
                raise RuntimeError(f"Model file not created at {output_path}")

        except Exception as e:
            console.print(f"  [red]✗ Export failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            raise

        return output_path


def train_full_model(
    project_name: str,
    wake_word: str,
    clips_dir: Path,
    output_dir: Path,
    workspace_dir: Path,
    **kwargs
) -> Path:
    """
    Train a full wake word model from scratch

    Args:
        project_name: Name of the project
        wake_word: Wake word phrase
        clips_dir: Directory containing audio clips
        output_dir: Directory to save trained model
        workspace_dir: Workspace directory
        **kwargs: Additional training parameters

    Returns:
        Path to trained ONNX model
    """
    trainer = FullModelTrainer(
        project_name=project_name,
        wake_word=wake_word,
        clips_dir=clips_dir,
        output_dir=output_dir,
        workspace_dir=workspace_dir
    )

    return trainer.train(**kwargs)
