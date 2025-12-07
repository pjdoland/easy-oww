#!/usr/bin/env python3
"""
Debug script to investigate model predictions
"""
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import sys

# Add easy_oww to path
sys.path.insert(0, str(Path(__file__).parent))

from easy_oww.utils.paths import PathManager

def debug_model(project_name='hey_johnny'):
    """Debug model predictions with detailed output"""

    paths = PathManager()
    project_path = paths.get_project_path(project_name)

    # Find model
    models_dir = project_path / 'models'
    model_files = list(models_dir.glob('*.onnx'))

    if not model_files:
        print("No model found!")
        return

    model_path = model_files[0]
    print(f"Model: {model_path}")
    print(f"Model size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    # Load model with OpenWakeWord
    print("Loading model with OpenWakeWord...")
    try:
        from openwakeword.model import Model as OWWModel
        model = OWWModel(wakeword_models=[str(model_path)])
        print("✓ Model loaded successfully")
        print(f"Model name: {list(model.models.keys())}")
        print()
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get test clips
    clips_dir = project_path / 'clips'
    positive_clips = list((clips_dir / 'positive').glob('*.wav'))[:5]
    negative_clips = list((clips_dir / 'negative').glob('*.wav'))[:5]

    print(f"Testing with {len(positive_clips)} positive and {len(negative_clips)} negative clips")
    print()

    # Test positive clips
    print("=" * 60)
    print("POSITIVE CLIPS (should have high scores)")
    print("=" * 60)

    for i, clip_path in enumerate(positive_clips):
        # Load audio
        sample_rate, audio = wavfile.read(str(clip_path))

        print(f"\nClip {i+1}: {clip_path.name}")
        print(f"  Sample rate: {sample_rate}")
        print(f"  Shape: {audio.shape}")
        print(f"  Duration: {len(audio) / sample_rate:.3f}s")
        print(f"  Dtype: {audio.dtype}")
        print(f"  Range: [{audio.min()}, {audio.max()}]")

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.int16)
            print(f"  Converted to mono")

        # Resample if needed (to 16kHz)
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples).astype(np.int16)
            sample_rate = 16000
            print(f"  Resampled to 16kHz")

        # Convert to float32 and normalize
        audio_float = audio.astype(np.float32) / 32768.0
        print(f"  Normalized range: [{audio_float.min():.3f}, {audio_float.max():.3f}]")

        # Predict
        try:
            prediction = model.predict(audio_float)
            print(f"  Prediction: {prediction}")

            # Get score
            model_name = list(prediction.keys())[0]
            score = prediction[model_name]
            print(f"  Score: {score}")

        except Exception as e:
            print(f"  ✗ Prediction failed: {e}")
            import traceback
            traceback.print_exc()

    # Test negative clips
    print("\n" + "=" * 60)
    print("NEGATIVE CLIPS (should have low scores)")
    print("=" * 60)

    for i, clip_path in enumerate(negative_clips):
        # Load audio
        sample_rate, audio = wavfile.read(str(clip_path))

        print(f"\nClip {i+1}: {clip_path.name}")
        print(f"  Sample rate: {sample_rate}")
        print(f"  Shape: {audio.shape}")
        print(f"  Duration: {len(audio) / sample_rate:.3f}s")

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.int16)

        # Resample if needed
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples).astype(np.int16)
            sample_rate = 16000

        # Convert to float32 and normalize
        audio_float = audio.astype(np.float32) / 32768.0

        # Predict
        try:
            prediction = model.predict(audio_float)
            model_name = list(prediction.keys())[0]
            score = prediction[model_name]
            print(f"  Score: {score}")

        except Exception as e:
            print(f"  ✗ Prediction failed: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If all scores are 0.0 or very close to 0.0, the model didn't learn.")
    print("If positive and negative scores are similar, the model can't discriminate.")
    print("Expected: positive scores > 0.5, negative scores < 0.1")
    print()

if __name__ == '__main__':
    debug_model()
