# Training Orchestration System

This module provides a complete training pipeline for custom wake word models, orchestrating clip generation, augmentation, and model training.

## Overview

The training system automates the entire process of creating a wake word model:

1. **Clip Generation**: Process real recordings and generate synthetic samples using TTS
2. **Audio Augmentation**: Apply room acoustics and noise for robustness
3. **Model Training**: Train OpenWakeWord-compatible models

## Components

### TrainingConfig (`config.py`)

Manages all training parameters and settings.

**Key Parameters:**
- `wake_word`: The target wake word or phrase
- `target_samples`: Total samples to generate (default: 1000)
- `real_samples`: Expected real recordings (default: 20)
- `synthetic_samples`: TTS-generated samples (default: 980)
- `max_steps`: Training iterations (default: 10000)
- `sample_rate`: Audio sample rate (16000 Hz)
- `clip_duration_ms`: Target clip length (1000 ms)

**Example:**
```python
from easy_oww.training import TrainingConfig, ConfigManager

# Create default config
config = TrainingConfig.create_default(
    project_name="my_wake_word",
    wake_word="hey assistant",
    project_path=Path("./projects/my_wake_word")
)

# Customize settings
config.max_steps = 15000
config.synthetic_samples = 2000
config.voices = ['en_US-lessac-medium', 'en_US-amy-medium']

# Save config
config.save(Path("config.yaml"))
```

### ClipGenerator (`clips.py`)

Generates and prepares audio clips for training.

**Features:**
- Process real recordings (normalize, resample, trim/pad)
- Generate synthetic clips using TTS
- Generate negative samples from audio datasets
- Clip verification and validation

**Example:**
```python
from easy_oww.training import ClipGenerator
from easy_oww.tts import PiperTTS

generator = ClipGenerator(
    recordings_dir=Path("recordings"),
    clips_dir=Path("clips"),
    sample_rate=16000,
    target_duration_ms=1000
)

# Process real recordings
real_clips = generator.process_real_recordings()

# Generate synthetic clips
piper = PiperTTS(Path("piper"))
voice_models = [Path("voices/en_US-lessac-medium.onnx")]

synthetic_clips = generator.generate_synthetic_clips(
    wake_word="hey assistant",
    voice_models=voice_models,
    piper=piper,
    count=500
)

# Generate negative samples
negative_clips = generator.generate_negative_clips(
    negative_audio_dir=Path("datasets/fsd50k"),
    count=1000
)
```

### AudioAugmenter (`augmentation.py`)

Applies audio augmentations for robust training.

**Augmentation Types:**
- **Room Impulse Response (RIR)**: Simulate different acoustic environments
- **Background Noise**: Add realistic ambient sounds
- **Pitch Shifting**: Vary voice pitch (-2 to +2 semitones)
- **Time Stretching**: Speed variations (0.9x to 1.1x)
- **Volume Changes**: Level adjustments (-6 to +6 dB)

**Example:**
```python
from easy_oww.training import AudioAugmenter

augmenter = AudioAugmenter(
    rir_dir=Path("datasets/rir"),
    noise_dir=Path("datasets/fsd50k"),
    sample_rate=16000
)

# Augment single audio
augmented = augmenter.augment(
    audio,
    rir_prob=0.5,
    noise_prob=0.5,
    pitch_prob=0.3
)

# Augment batch of clips
augmented_clips = augmenter.augment_clips(
    input_clips=clip_paths,
    output_dir=Path("clips/augmented"),
    augmentations_per_clip=3
)
```

### TrainingOrchestrator (`orchestrator.py`)

Coordinates the complete training pipeline.

**Pipeline Phases:**
1. Load and validate configuration
2. Generate clips (real + synthetic + negative)
3. Apply audio augmentation
4. Train model (OpenWakeWord integration)

**Example:**
```python
from easy_oww.training import run_training

# Run complete pipeline
run_training(
    project_path=Path("projects/my_wake_word"),
    workspace_path=Path("~/.easy-oww"),
    resume=False,
    verbose=True
)
```

## CLI Usage

### Create Project

```bash
# Create project with interactive prompts
easy-oww create my_wake_word

# Create with parameters
easy-oww create my_wake_word \
  --wake-word "hey assistant" \
  --samples 2000 \
  --steps 15000
```

This creates:
- Project directory structure
- Training configuration file
- Empty directories for recordings, clips, features, models

### Record Samples

```bash
# Record 20 samples (default)
easy-oww record my_wake_word

# Record custom amount
easy-oww record my_wake_word --count 50
```

### Train Model

```bash
# Run complete training pipeline
easy-oww train my_wake_word

# Resume from checkpoint
easy-oww train my_wake_word --resume

# Verbose output
easy-oww train my_wake_word --verbose
```

## Training Pipeline Details

### Phase 1: Clip Generation

**Real Recordings:**
- Loads recordings from `recordings/` directory
- Resamples to 16 kHz if needed
- Converts to mono if stereo
- Normalizes audio levels
- Trims or pads to target duration (1000 ms)

**Synthetic Generation:**
- Uses Piper TTS with multiple voices
- Generates text variations of wake word
- Processes to standard format
- Distributes across available voices

**Negative Samples:**
- Extracts random segments from FSD50K dataset
- Ensures diverse non-wake-word sounds
- Equal or greater count than positive samples

### Phase 2: Augmentation

**Room Acoustics (RIR):**
- Applies room impulse responses via convolution
- Simulates various acoustic environments
- Uses MIT RIR dataset

**Background Noise:**
- Mixes ambient sounds at various SNR levels (5-20 dB)
- Uses FSD50K dataset for diverse noise types
- Maintains intelligibility

**Voice Variations:**
- Pitch shifting for voice diversity
- Time stretching for speed variations
- Volume adjustments for level robustness

### Phase 3: Model Training

**OpenWakeWord Integration:**
The system prepares clips in the format expected by OpenWakeWord:
- 16 kHz sample rate
- 1 second duration
- Organized as positive/negative classes
- Ready for embedding extraction

**Note:** Direct OpenWakeWord model training integration is in progress. Currently, the pipeline:
1. Prepares all clips in correct format
2. Organizes clips for training
3. Provides instructions for using OpenWakeWord's training scripts

## Configuration File Format

The training configuration is stored as `config.yaml`:

```yaml
wake_word: hey assistant
project_name: my_wake_word
target_samples: 1000
real_samples: 20
synthetic_samples: 980
max_steps: 10000
batch_size: 512
learning_rate: 0.001
sample_rate: 16000
clip_duration_ms: 1000
use_augmentation: true
augmentation_probability: 0.8
noise_probability: 0.5
rir_probability: 0.5
voices:
  - en_US-lessac-medium
  - en_US-amy-medium
min_voices: 2
recordings_dir: ./recordings
clips_dir: ./clips
features_dir: ./features
models_dir: ./models
```

## Best Practices

### Sample Counts

**Real Recordings:**
- Minimum: 20 samples
- Recommended: 50+ samples
- More is better for capturing natural variations

**Synthetic Samples:**
- Minimum: 500 samples
- Recommended: 1000-2000 samples
- Use 2-3 different voices minimum

**Negative Samples:**
- Should equal or exceed positive samples
- Use diverse sound types
- Include similar-sounding phrases

### Voice Selection

- Use at least 2 different voices
- Mix male and female voices
- Include different accents if available
- Higher quality voices produce better results

### Augmentation

**When to Use:**
- Always recommended for robustness
- Essential for real-world deployment
- Helps model generalize

**Probabilities:**
- RIR: 0.5 (50% of samples)
- Noise: 0.5 (50% of samples)
- Combined: ~25% have both

### Training Duration

**Max Steps:**
- Quick test: 1,000 steps (~3 minutes)
- Normal: 10,000 steps (~30 minutes)
- Thorough: 20,000+ steps (~1+ hours)

Monitor for overfitting with validation set.

## Troubleshooting

### Not Enough Recordings

```
Error: Only found 10 real recordings (expected 20)
```

**Solution:** Record more samples or adjust config:
```python
config.real_samples = 10  # Match actual count
```

### No Voices Available

```
Error: No voice models available
```

**Solution:** Download TTS voices:
```bash
easy-oww download-voices --language en_US --count 2
```

### Datasets Not Found

```
Warning: FSD50K dataset not found
```

**Solution:** Download required datasets:
```bash
easy-oww download --required-only
```

### Augmentation Too Aggressive

If augmented samples sound unnatural:
```python
config.augmentation_probability = 0.5  # Reduce from 0.8
config.noise_probability = 0.3  # Reduce noise
```

### Out of Memory

If training fails with OOM:
```python
config.batch_size = 256  # Reduce from 512
config.target_samples = 500  # Reduce sample count
```

## File Organization

After training, project directory contains:

```
my_wake_word/
├── config.yaml                    # Training configuration
├── recordings/                    # Real user recordings
│   ├── sample_0000.wav
│   ├── sample_0001.wav
│   └── ...
├── clips/                         # Processed training clips
│   ├── positive/                  # Wake word clips
│   │   ├── real_0000.wav
│   │   ├── synth_0000.wav
│   │   └── ...
│   ├── positive_augmented/        # Augmented positive clips
│   │   ├── real_0000_aug0.wav
│   │   └── ...
│   └── negative/                  # Non-wake-word clips
│       ├── negative_0000.wav
│       └── ...
├── features/                      # Extracted features (future)
├── models/                        # Trained models
│   └── my_wake_word.onnx
└── checkpoints/                   # Training checkpoints
```

## Performance Metrics

### Generation Speed

- Real recording processing: ~10 samples/second
- Synthetic generation: ~2 samples/second
- Augmentation: ~20 samples/second

### Recommended Hardware

**Minimum:**
- CPU: Dual-core 2+ GHz
- RAM: 4 GB
- Storage: 10 GB free

**Recommended:**
- CPU: Quad-core 3+ GHz
- RAM: 8+ GB
- Storage: 50+ GB (for datasets)
- GPU: Optional, speeds up training

## Testing

Run training tests:

```bash
pytest tests/test_training.py -v
```

Test individual components:

```python
# Test config
from easy_oww.training import TrainingConfig
config = TrainingConfig(project_name="test", wake_word="test")
is_valid, issues = config.validate()
print(f"Valid: {is_valid}, Issues: {issues}")

# Test clip generator
from easy_oww.training import ClipGenerator
generator = ClipGenerator(
    recordings_dir=Path("recordings"),
    clips_dir=Path("clips")
)
counts = generator.get_clip_counts()
print(f"Clips: {counts}")
```

## Future Enhancements

Planned improvements:
- Direct OpenWakeWord training integration
- Real-time training progress visualization
- Hyperparameter optimization
- Multi-language support
- Cloud training support
- Pre-trained model fine-tuning
