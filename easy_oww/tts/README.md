# Text-to-Speech (TTS) Integration

This module provides TTS integration using Piper for generating synthetic wake word samples.

## Overview

The TTS system enables automatic generation of hundreds of diverse wake word samples using text-to-speech voices. This is crucial for training robust wake word models without requiring users to manually record hundreds of samples.

## Components

### PiperTTS (`piper.py`)

Manages Piper TTS installation and speech synthesis.

**Features:**
- Automatic Piper binary download and installation
- Platform detection (macOS, Linux, Windows)
- Text-to-speech generation at 16kHz for OpenWakeWord
- Voice model management
- Batch generation support

**Example:**
```python
from easy_oww.tts import PiperTTS
from pathlib import Path

# Initialize Piper
piper = PiperTTS(install_dir=Path("~/.easy-oww/piper"))

# Install if needed
if not piper.is_installed():
    piper.install()

# Generate speech
voice_model = Path("voices/en_US-lessac-medium.onnx")
piper.generate_speech(
    text="hey assistant",
    voice_model=voice_model,
    output_path=Path("output.wav"),
    sample_rate=16000
)
```

### VoiceDownloader (`voices.py`)

Downloads and manages Piper voice models from HuggingFace.

**Recommended Voices:**
- **en_US-lessac-medium**: Clear American English (63 MB)
- **en_US-amy-medium**: Female American English (63 MB)
- **en_US-ryan-high**: High quality male voice (116 MB)
- **en_GB-alan-medium**: British English (63 MB)
- Plus voices for Spanish, French, and German

**Example:**
```python
from easy_oww.tts import VoiceDownloader
from pathlib import Path

# Initialize downloader
downloader = VoiceDownloader(voices_dir=Path("voices"))

# Download recommended voices
downloaded = downloader.download_recommended_voices(
    language='en_US',
    max_voices=2
)

# List installed voices
installed = downloader.list_installed_voices()
for voice in installed:
    print(f"{voice['name']}: {voice['language']}")
```

### SampleGenerator (`generator.py`)

Generates synthetic wake word samples with variations.

**Features:**
- Automatic text variation generation
- Multi-voice sample generation
- Phrase variations (plain, with punctuation, with prefixes)
- Progress tracking
- Quality validation

**Example:**
```python
from easy_oww.tts import PiperTTS, SampleGenerator
from pathlib import Path

piper = PiperTTS(install_dir=Path("~/.easy-oww/piper"))
generator = SampleGenerator(
    piper=piper,
    output_dir=Path("samples"),
    sample_rate=16000
)

# Generate 100 samples with one voice
voice_model = Path("voices/en_US-lessac-medium.onnx")
samples = generator.generate_samples(
    wake_word="hey assistant",
    voice_model=voice_model,
    count=100
)

# Generate samples with multiple voices
voice_models = [
    Path("voices/en_US-lessac-medium.onnx"),
    Path("voices/en_US-amy-medium.onnx")
]

all_samples = generator.generate_mixed_samples(
    wake_word="hey assistant",
    voice_models=voice_models,
    total_count=500
)
```

## CLI Integration

### Setup TTS

Piper TTS is automatically set up during workspace initialization:

```bash
easy-oww init
# Will prompt to install Piper TTS
```

### Download Voices

Download voice models for your language:

```bash
# Download 2 English (US) voices (default)
easy-oww download-voices

# Download specific language
easy-oww download-voices --language en_GB --count 1

# Download more voices
easy-oww download-voices --language en_US --count 3
```

### List Voices

List installed voice models:

```bash
easy-oww list-voices
```

## Voice Model Details

### English (US) Voices

| Voice | Quality | Size | Description |
|-------|---------|------|-------------|
| en_US-lessac-medium | Medium | 63 MB | Clear, neutral voice |
| en_US-amy-medium | Medium | 63 MB | Female voice |
| en_US-ryan-high | High | 116 MB | High quality male |

### Other Languages

- **English (GB)**: en_GB-alan-medium
- **Spanish**: es_ES-mls_9972-low
- **French**: fr_FR-mls_1840-low
- **German**: de_DE-thorsten-medium

## Synthetic Sample Generation

### Best Practices

**Voice Diversity:**
- Use 2-3 different voices minimum
- Mix male and female voices
- Include voices with different accents if available

**Sample Quantity:**
- Generate 500-1000 synthetic samples
- Combine with 20-50 real recordings
- More samples = better model robustness

**Text Variations:**
The generator automatically creates variations:
- Plain: "hey assistant"
- With punctuation: "hey assistant."
- With emphasis: "hey assistant!"
- With context: "hey hey assistant"
- With pauses: "hey assistant,"

### Generation Time

Approximate generation times:
- 100 samples: ~3-5 minutes
- 500 samples: ~15-25 minutes
- 1000 samples: ~30-50 minutes

Times vary based on:
- CPU speed
- Text length
- Number of voices
- Disk I/O speed

## Technical Details

### Audio Format

All generated samples use:
- **Sample Rate**: 16000 Hz (required for OpenWakeWord)
- **Channels**: 1 (mono)
- **Bit Depth**: 16-bit PCM
- **Format**: WAV

### Piper Installation

Binaries are automatically downloaded from:
- Repository: `rhasspy/piper` on GitHub
- Version: 2023.11.14-2
- Platform-specific builds for macOS, Linux, Windows

### Voice Model Format

Voice models consist of two files:
- `.onnx` - ONNX neural network model
- `.onnx.json` - Configuration file (sample rate, language, etc.)

## Troubleshooting

### Piper Installation Fails

```bash
# Check system compatibility
# Piper requires glibc 2.31+ on Linux

# Manual installation:
# Download from: https://github.com/rhasspy/piper/releases
# Extract and place in ~/.easy-oww/piper/
```

### Voice Download Fails

```bash
# Check internet connection
# Voice files are large (60-120 MB each)

# Manual download from:
# https://huggingface.co/rhasspy/piper-voices
```

### Generated Audio Quality Issues

- Ensure voice models are not corrupted
- Try different voices
- Check that Piper binary is properly installed
- Verify disk space for output files

## Testing

Run TTS tests:

```bash
pytest tests/test_tts.py -v
```

Test Piper installation:

```python
from easy_oww.tts import PiperTTS
from pathlib import Path

piper = PiperTTS(Path("~/.easy-oww/piper"))
print(f"Installed: {piper.is_installed()}")

# Test voice
voice = Path("voices/en_US-lessac-medium.onnx")
success, message = piper.test_voice(voice)
print(message)
```

## Performance Tips

1. **Parallel Generation**: Generate samples in batches for faster processing
2. **Disk I/O**: Use SSD for faster file writing
3. **Voice Selection**: Medium-quality voices are faster than high-quality
4. **Text Length**: Shorter texts generate faster

## Future Enhancements

Potential improvements:
- Custom voice training
- Speed/pitch variations
- Background noise mixing during generation
- Real-time generation during training
- GPU acceleration for generation
