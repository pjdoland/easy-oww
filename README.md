# easy-oww

**Simplified OpenWakeWord ONNX Model Creation Tool**

`easy-oww` is a complete CLI tool that makes creating custom wake word models for [OpenWakeWord](https://github.com/dscripka/openWakeWord) as easy as possible. It guides you through the entire process from recording audio samples to training and testing a production-ready ONNX model.

## Features

- 🎤 **Interactive Audio Recording** - Guided recording with real-time quality validation
- 🤖 **Automatic TTS Generation** - Integrates Piper TTS for 500+ synthetic samples
- 📦 **Auto-Downloads Datasets** - Handles 50GB+ downloads with resume capability
- 🎚️ **Audio Augmentation** - Applies room acoustics and noise for robustness
- 🎯 **One-Command Training** - Simplified 3-phase training pipeline
- ✅ **Real-Time Testing** - Live microphone detection with accuracy metrics
- 🔄 **Resume Capability** - Continue from last checkpoint if interrupted
- 📊 **Rich Terminal UI** - Beautiful progress bars and status displays

## Documentation

- **[Installation Guide](INSTALLATION.md)** - Detailed setup including external drive support
- **[FAQ](FAQ.md)** - Frequently asked questions and troubleshooting
- **[Audio Recording](easy_oww/audio/README.md)** - Recording system documentation
- **[TTS Integration](easy_oww/tts/README.md)** - Text-to-speech setup and usage
- **[Training Pipeline](easy_oww/training/README.md)** - Training workflow details
- **[Model Testing](easy_oww/testing/README.md)** - Testing and validation guide

## Quick Start

### Installation

**Automatic Setup (Recommended):**
```bash
# Clone the repository
git clone https://github.com/yourusername/easy-oww.git
cd easy-oww

# Run the setup script
./setup.sh
```

The setup script will:
- Check Python version (3.7+ required)
- Create a virtual environment
- Install all dependencies
- Verify installation

**Manual Installation:**
```bash
# Clone the repository
git clone https://github.com/yourusername/easy-oww.git
cd easy-oww

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

**External Drive Setup (for limited space):**

After installation, initialize your workspace on an external drive:
```bash
# Activate virtual environment
source venv/bin/activate

# Initialize workspace on external drive
easy-oww init --workspace /Volumes/MyUSB/easy-oww

# Set environment variable (optional, avoids typing --workspace)
echo 'export EASY_OWW_WORKSPACE="/Volumes/MyUSB/easy-oww"' >> ~/.bashrc
source ~/.bashrc
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions, including USB-C drive configuration.

### Create Your First Wake Word Model

```bash
# Activate the virtual environment (if not already active)
source venv/bin/activate

# 1. Initialize workspace (one-time setup)
easy-oww init

# 2. Download required datasets (~20GB)
easy-oww download --required-only

# 3. Download TTS voices
easy-oww download-voices --language en_US --count 2

# 4. Create a new project
easy-oww create hey_assistant --wake-word "hey assistant"

# 5. Record your wake word samples
easy-oww record hey_assistant --count 20

# 6. (Optional) Record negative/adversarial samples to reduce false positives
easy-oww record-negative hey_assistant --count 10

# 7. Train the model (generates 1000+ samples automatically)
easy-oww train hey_assistant

# 8. Test the trained model
easy-oww test hey_assistant
```

That's it! You'll have a custom wake word model ready to deploy.

**Note:** Always activate the virtual environment before using easy-oww:
```bash
source venv/bin/activate
```

## Requirements

### System Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **Python** | 3.7+ | 3.9-3.10 | Best compatibility |
| **Storage** | 20GB free | 40GB+ free | Can use external USB-C drive |
| **RAM** | 4GB | 8GB+ | More = faster training |
| **Microphone** | Any | USB or built-in | For recording samples |
| **GPU** | None | CUDA-compatible | 50-70% faster training |

### Storage Breakdown

**Required (~20GB):**
- ACAV100M features: 17.5GB (audio embeddings)
- MIT RIR dataset: 2GB (room acoustics)
- TTS voices: 200MB (2-3 voices)
- Workspace: 1-2GB (projects, models)

**Optional (+30GB):**
- FSD50K dataset: 30GB (background sounds)

**Limited Space? Use a USB-C Drive!**

You can run easy-oww entirely from an external drive (USB-C SSD recommended). See [INSTALLATION.md](INSTALLATION.md#external-storage-setup) for setup instructions.

**Recommended Drives:**
- Budget: SanDisk Extreme PRO USB 3.1 (128GB+)
- Better: Samsung T7 Portable SSD (500GB+)
- Best: Samsung T7 Shield SSD (1TB)

## How It Works

`easy-oww` automates the complete OpenWakeWord training workflow:

### 1. **Initialize Workspace**
- Checks system requirements (Python, disk, RAM, GPU)
- Creates workspace directory structure
- Installs Piper TTS for synthetic sample generation

### 2. **Download Datasets**
- ACAV100M features (17.5GB) - Pre-computed audio embeddings
- MIT Room Impulse Responses (2GB) - Acoustic simulations
- FSD50K (30GB, optional) - Background noise and negative samples
- Resume capability for interrupted downloads

### 3. **Create Project**
- Sets up project directory structure
- Creates training configuration
- Detects installed TTS voices
- Configures sample counts and training parameters

### 4. **Record Samples**
- Interactive microphone selection
- Real-time audio level monitoring
- Quality validation (duration, volume, SNR)
- Automatic retry for failed samples
- Session summary with statistics

### 5. **Train Model**

**Phase 1: Clip Generation**
- Processes real recordings (normalize, resample, trim/pad)
- Generates synthetic samples using multiple TTS voices
- Creates text variations (punctuation, prefixes)
- Extracts negative samples from FSD50K
- Validates all generated clips

**Phase 2: Audio Augmentation**
- Applies room impulse responses (RIR convolution)
- Adds background noise at various SNR levels
- Applies pitch shifting and time stretching
- Adjusts volume levels
- Creates 2-3x augmented variations per clip

**Phase 3: Model Training**
- Prepares clips in OpenWakeWord format
- Generates embeddings from melspectrogram features
- Trains classification model
- Exports to ONNX format

### 6. **Test Model**
- Real-time microphone detection
- Live detection visualization
- Accuracy metrics on test clips
- Confusion matrix and F1 scores
- Results saved to JSON

## CLI Commands

### Workspace Management

```bash
# Initialize workspace
easy-oww init

# Custom workspace location
easy-oww init --workspace /path/to/workspace
```

### Dataset Management

```bash
# Download all datasets
easy-oww download

# Download only required datasets
easy-oww download --required-only

# Download to custom location
easy-oww download --workspace /path/to/workspace
```

### TTS Voice Management

```bash
# Download voices for English (US)
easy-oww download-voices

# Download voices for other languages
easy-oww download-voices --language en_GB --count 1

# List installed voices
easy-oww list-voices
```

### Project Management

```bash
# Create project with interactive prompts
easy-oww create my_wake_word

# Create with all parameters
easy-oww create my_wake_word \
  --wake-word "hey assistant" \
  --samples 2000 \
  --steps 15000

# List all projects
easy-oww list
```

### Recording

```bash
# Record default 20 positive samples
easy-oww record my_wake_word

# Record custom number
easy-oww record my_wake_word --count 50

# Record negative/adversarial samples (reduces false positives)
easy-oww record-negative my_wake_word --count 20
```

**What are negative samples?**

Negative samples are phrases that should NOT trigger your wake word. Recording these helps your model distinguish between your wake word and similar-sounding phrases, significantly reducing false positives.

**Examples of good negative samples:**
- Similar-sounding phrases: "hey system" when wake word is "hey assistant"
- Partial wake words: just "hey" or just "assistant"
- Rhyming words: "resistance" for "assistant"
- Common phrases in your environment

### Training

```bash
# Train model
easy-oww train my_wake_word

# Resume interrupted training
easy-oww train my_wake_word --resume

# Verbose output
easy-oww train my_wake_word --verbose
```

### Testing

```bash
# Test model (interactive selection)
easy-oww test my_wake_word

# Test for specific duration
easy-oww test my_wake_word --duration 120
```

## External Storage Support

**Yes! You can run easy-oww from a USB-C thumb drive or external SSD.**

This is perfect for users with limited internal drive space. The tool works great on external storage, especially USB 3.0+ drives.

### Quick External Drive Setup

```bash
# 1. Connect your USB-C drive

# 2. Initialize workspace on the drive (macOS example)
easy-oww init --workspace /Volumes/MyDrive/easy-oww

# 3. Set as default (optional)
export EASY_OWW_WORKSPACE="/Volumes/MyDrive/easy-oww"

# 4. Use normally
easy-oww download --required-only
easy-oww create my_project
easy-oww train my_project
```

### Performance Notes

| Drive Type | Read/Write | Training Time | Recommended? |
|------------|------------|---------------|--------------|
| USB-C SSD | 500+ MB/s | Same as internal | ✅ Yes |
| USB 3.0 SSD | 400+ MB/s | +10-20% slower | ✅ Yes |
| USB 3.0 Flash | 100-150 MB/s | +50-100% slower | ⚠️ OK |
| USB 2.0 Flash | 10-30 MB/s | 3-4x slower | ❌ No |

**Key Takeaway:** USB-C SSDs perform nearly identically to internal drives. USB 3.0 drives work well too.

See [INSTALLATION.md](INSTALLATION.md#external-storage-setup) for detailed setup, formatting tips, and optimization guide.

## Project Structure

After running through the workflow, your project will look like:

```
~/.easy-oww/
├── datasets/                    # Downloaded datasets
│   ├── acav100m_features/      # 40GB pre-computed features
│   ├── rir/                    # Room impulse responses
│   └── fsd50k/                 # Sound events (optional)
├── voices/                      # TTS voice models
│   ├── en_US-lessac-medium.onnx
│   └── en_US-amy-medium.onnx
├── piper-sample-generator/      # Piper TTS binary
└── projects/
    └── my_wake_word/            # Your project
        ├── config.yaml          # Training configuration
        ├── recordings/          # Your recorded wake word samples
        │   ├── sample_0000.wav
        │   └── ...
        ├── recordings_negative/ # Your recorded negative samples (optional)
        │   ├── sample_0000.wav
        │   └── ...
        ├── clips/               # Processed training clips
        │   ├── positive/        # Wake word clips
        │   ├── positive_augmented/  # Augmented clips
        │   └── negative/        # Non-wake-word clips
        ├── features/            # Extracted features
        ├── models/              # Trained models
        │   └── my_wake_word.onnx
        └── test_results.json    # Testing metrics
```

## Training Best Practices

### Recording Tips

**Recording Workflow:**
- Each recording is **3 seconds** by default
- After recording, you'll hear an **automatic playback**
- Choose to: **Accept**, **Re-record**, or **Skip**
- This ensures every sample meets your quality standards

**Positive Samples (Wake Word):**
- **Environment**: Record in a quiet room
- **Distance**: Keep consistent distance from mic (6-12 inches)
- **Variations**: Vary tone, speed, and emphasis
- **Consistency**: Pronounce the wake word the same way
- **Quantity**: 20 minimum, 50+ recommended

**Negative Samples (Adversarial):**
- **Diversity**: Record various similar-sounding phrases
- **Partial phrases**: Say individual words from your wake word
- **Common mistakes**: Think about what people might say by accident
- **Environment sounds**: Include common phrases you say often
- **Quantity**: 10-20 samples recommended for better accuracy
- **Impact**: Can reduce false positives by 30-50%

### Model Configuration

**For Quick Testing:**
```yaml
target_samples: 500
synthetic_samples: 480
max_steps: 5000
voices: 2
clip_duration_ms: 3000  # 3 seconds
```

**For Production:**
```yaml
target_samples: 2000
synthetic_samples: 1950
max_steps: 15000
voices: 3+
clip_duration_ms: 3000  # 3 seconds
```

### Voice Selection

- Use at least 2 different voices
- Mix male and female voices
- Include different accents if available
- Higher quality voices = better results

### Augmentation

Augmentation makes models robust to real-world conditions:

- **RIR Probability**: 0.5 (simulate different rooms)
- **Noise Probability**: 0.5 (handle background noise)
- **Pitch/Time Variations**: 0.3 (voice diversity)

## Performance Metrics

### Training Time Estimates

On typical hardware (CPU training):

| Samples | TTS Generation | Augmentation | Training | Total |
|---------|---------------|--------------|----------|-------|
| 500 | ~10 min | ~5 min | ~15 min | ~30 min |
| 1000 | ~20 min | ~10 min | ~30 min | ~60 min |
| 2000 | ~40 min | ~20 min | ~60 min | ~2 hours |

With GPU: Training time reduced by 50-70%

### Accuracy Expectations

With proper training:
- **Accuracy**: 90-95%
- **Precision**: 85-92%
- **Recall**: 92-98%
- **F1 Score**: 88-95%

Higher quality recordings and more samples improve results.

## Advanced Configuration

### Custom Training Config

Edit `config.yaml` in your project:

```yaml
# Sample counts
target_samples: 2000
real_samples: 50
synthetic_samples: 1950

# Training parameters
max_steps: 15000
batch_size: 512
learning_rate: 0.001

# Audio settings
sample_rate: 16000
clip_duration_ms: 1000

# Augmentation
use_augmentation: true
rir_probability: 0.5
noise_probability: 0.5

# TTS voices
voices:
  - en_US-lessac-medium
  - en_US-amy-medium
  - en_US-ryan-high
```

### Detection Threshold Tuning

Find optimal threshold for your use case:

- **Low Threshold (0.3-0.4)**: Fewer misses, more false alarms
- **Medium Threshold (0.5)**: Balanced (default)
- **High Threshold (0.6-0.7)**: Fewer false alarms, more misses

Test multiple thresholds:

```bash
easy-oww test my_wake_word
> Customize threshold? Yes
> Enter threshold: 0.4
```

## Troubleshooting

### Installation Issues

```bash
# Missing dependencies
pip install -r requirements.txt

# Permission errors
pip install --user -e .
```

### Download Failures

- Check internet connection
- Verify disk space
- Use `--required-only` flag to skip optional datasets
- Downloads auto-resume on restart

### Low Recording Quality

- Check microphone volume settings
- Reduce background noise
- Speak clearly and consistently
- Move closer to microphone

### Training Failures

- Ensure datasets are downloaded
- Check TTS voices are installed
- Verify sufficient disk space
- Review error messages with `--verbose`

### No Detections When Testing

- Lower detection threshold
- Check if model file exists
- Verify microphone is working
- Test with the wake word you trained on

## Complete Documentation

### Getting Started
- **[Installation Guide](INSTALLATION.md)** - Complete setup instructions
  - Standard installation
  - External drive setup (USB-C, USB 3.0)
  - Virtual environment configuration
  - Platform-specific notes (macOS, Linux, Windows)
  - Troubleshooting common issues

- **[FAQ](FAQ.md)** - Comprehensive Q&A
  - Storage and performance questions
  - Recording tips and troubleshooting
  - Training configuration advice
  - Testing and accuracy improvement
  - Technical questions answered

### Module Documentation
- **[Audio Recording](easy_oww/audio/README.md)** - Recording system
  - Microphone selection and testing
  - Quality validation rules
  - Session management
  - Audio processing details

- **[TTS Integration](easy_oww/tts/README.md)** - Text-to-speech
  - Piper TTS installation
  - Voice model management
  - Multi-language support
  - Sample generation strategies

- **[Training Pipeline](easy_oww/training/README.md)** - Training workflow
  - Configuration options
  - Clip generation process
  - Augmentation techniques
  - Best practices and optimization

- **[Model Testing](easy_oww/testing/README.md)** - Testing and metrics
  - Real-time detection
  - Accuracy evaluation
  - Metrics explanation
  - Threshold tuning guide

## Testing

Run the test suite:

```bash
# Activate virtual environment
source venv/bin/activate

# Install dev dependencies (if not already installed)
pip install pytest pytest-mock

# Run all tests
pytest

# Run with coverage
pytest --cov=easy_oww

# Run specific test module
pytest tests/test_audio.py -v
```

## Common Use Cases

### Home Automation
```bash
# Create wake words for different rooms
easy-oww create kitchen_lights --wake-word "kitchen lights"
easy-oww create bedroom_fan --wake-word "bedroom fan"

# Deploy models to Raspberry Pi with OpenWakeWord
```

### Voice Assistants
```bash
# Custom wake word for your assistant
easy-oww create hey_jarvis --wake-word "hey jarvis"
easy-oww create ok_friday --wake-word "ok friday"

# Use in place of "Alexa" or "Hey Google"
```

### Multi-Language Support
```bash
# Download voices for your language
easy-oww download-voices --language de_DE  # German
easy-oww download-voices --language fr_FR  # French
easy-oww download-voices --language es_ES  # Spanish

# Create project in your language
easy-oww create mein_projekt --wake-word "hey assistent"
```

### Embedded Devices
```bash
# Train on PC, deploy to embedded device
easy-oww train my_wake_word

# Copy model to device
scp ~/.easy-oww/projects/my_wake_word/models/*.onnx pi@raspberrypi:/home/pi/models/

# Use with OpenWakeWord on the device
```

### Multiple Wake Words
```bash
# Create different wake words for different actions
easy-oww create wake_word_1 --wake-word "hey assistant"
easy-oww create wake_word_2 --wake-word "turn on lights"
easy-oww create wake_word_3 --wake-word "play music"

# Load all models in your application
```

## Tips and Tricks

### Faster Training
```bash
# Use fewer samples for quick testing
easy-oww create test_project --samples 500 --steps 5000

# Full training for production
easy-oww create prod_project --samples 2000 --steps 15000
```

### Better Accuracy
```yaml
# Edit project config.yaml for better results
target_samples: 2000      # More samples
max_steps: 15000          # Train longer
voices:                   # Use more voices
  - en_US-lessac-medium
  - en_US-amy-medium
  - en_US-ryan-high
  - en_GB-alan-medium
```

### Space Management
```bash
# After training, delete intermediate files
cd ~/.easy-oww/projects/my_project
rm -rf clips/  # Can regenerate if needed
rm -rf features/  # Can regenerate if needed

# Keep: recordings/, models/, config.yaml
```

### Backup Important Files
```bash
# Backup your recordings (irreplaceable)
cp -r ~/.easy-oww/projects/my_project/recordings ~/Backups/

# Backup trained models
cp ~/.easy-oww/projects/*/models/*.onnx ~/Backups/models/
```

### Environment Variable for Convenience
```bash
# Add to ~/.bashrc or ~/.zshrc
export EASY_OWW_WORKSPACE="/Volumes/MyDrive/easy-oww"

# Now you can omit --workspace flag
easy-oww create my_project
easy-oww train my_project
```

## Roadmap

**Current Version: 0.1.0** - Full feature set complete!

### Completed Features
- ✅ Complete CLI workflow (init → download → create → record → train → test)
- ✅ External storage support
- ✅ Multi-language TTS
- ✅ Audio augmentation
- ✅ Real-time testing with metrics
- ✅ Comprehensive documentation

### Future Enhancements
- [ ] Direct OpenWakeWord training integration (currently prepares clips)
- [ ] GUI application for non-technical users
- [ ] Cloud training option (Google Colab integration)
- [ ] Web dashboard for monitoring training
- [ ] Pre-trained model fine-tuning
- [ ] Docker containerization
- [ ] Model quantization for edge devices
- [ ] Automatic threshold optimization
- [ ] Model versioning and comparison tools

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) - Wake word detection framework
- [Piper TTS](https://github.com/rhasspy/piper) - Text-to-speech synthesis
- [ACAV100M Dataset](https://huggingface.co/datasets/davidscripka/openwakeword_features) - Pre-computed features
- [MIT RIR Dataset](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) - Room impulse responses
- [FSD50K](https://zenodo.org/record/4060432) - Sound event dataset

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/easy-oww/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/easy-oww/discussions)

---

Made for the OpenWakeWord community
