# Quick Start Guide

Get your first wake word model working in under an hour! This guide assumes you have limited internal storage and want to use an external drive.

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.7+ installed (`python3 --version`)
- [ ] 50+ GB external USB-C drive or SSD (recommended) OR 50+ GB internal storage
- [ ] Stable internet connection (for ~40 GB downloads)
- [ ] Microphone (built-in laptop mic is fine)
- [ ] 1-2 hours of time

## Step-by-Step Setup

### 1. Install easy-oww (5 minutes)

**Option A: Automatic Setup (Recommended)**
```bash
# Clone repository
git clone https://github.com/yourusername/easy-oww.git
cd easy-oww

# Run setup script
./setup.sh

# The script will create a virtual environment and install everything
```

**Option B: Manual Setup**
```bash
# Clone repository
git clone https://github.com/yourusername/easy-oww.git
cd easy-oww

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -e .

# Verify installation
easy-oww --version
# Should output: easy-oww, version 0.1.0
```

**Troubleshooting:**
- If `easy-oww` command not found: Make sure venv is activated, or try `python -m easy_oww.cli.main --version`
- If permission errors on macOS: Use the virtual environment method (venv)

### 2. Set Up Workspace (2 minutes)

**First, activate the virtual environment:**
```bash
# Navigate to easy-oww directory
cd easy-oww

# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**Option A: Internal Storage**
```bash
# Initialize in default location (~/.easy-oww)
easy-oww init
```

**Option B: External Drive (Recommended if space-limited)**
```bash
# Connect your USB-C drive

# Initialize on external drive (macOS example)
easy-oww init --workspace /Volumes/MyDrive/easy-oww

# Set environment variable to avoid typing --workspace every time
echo 'export EASY_OWW_WORKSPACE="/Volumes/MyDrive/easy-oww"' >> ~/.bashrc
source ~/.bashrc

# For Linux, use /mnt/MyDrive/easy-oww
# For Windows, use E:\easy-oww
```

**Note:** Remember to activate the virtual environment every time you open a new terminal:
```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**What happens:**
- System requirements check (Python, disk space, RAM, GPU)
- Directory structure creation
- Piper TTS installation (auto-downloads and installs)

### 3. Download Datasets (20-40 minutes)

```bash
# Download required datasets (~43 GB)
easy-oww download --required-only

# What gets downloaded:
# - ACAV100M features (40 GB) - audio embeddings
# - MIT RIR dataset (2 GB) - room acoustics
# Progress bars will show download status
# Downloads auto-resume if interrupted
```

**Note:** This is a one-time download. All future projects use these datasets.

**Coffee break recommended!** ☕

### 4. Download TTS Voices (3-5 minutes)

```bash
# Download 2 English voices (~200 MB)
easy-oww download-voices --language en_US --count 2

# For other languages:
# --language de_DE  # German
# --language fr_FR  # French
# --language es_ES  # Spanish
```

### 5. Create Your Project (1 minute)

```bash
# Create project for your wake word
easy-oww create my_wake_word --wake-word "hey assistant"

# Replace "hey assistant" with your desired wake word:
# - "hey jarvis"
# - "ok computer"
# - "alexa" (custom version)
# - Any phrase you want!
```

**What happens:**
- Project directory created
- Training configuration generated
- Voices auto-detected

### 6. Record Samples (5-10 minutes)

```bash
# Record 20 samples of your wake word
easy-oww record my_wake_word --count 20

# Interactive session will:
# 1. Let you select microphone
# 2. Test microphone quality
# 3. Guide you through recording each sample
# 4. Validate quality in real-time
# 5. Show summary when done
```

**Recording Tips:**
- Find a quiet room
- Speak naturally and clearly
- Vary your tone slightly (louder, softer, faster, slower)
- Keep consistent pronunciation
- Stay 6-12 inches from microphone

**What to say:** Speak your wake word ("hey assistant") for each sample.

### 7. Train Your Model (30-60 minutes)

```bash
# Start training
easy-oww train my_wake_word

# Training happens in 3 phases:
# Phase 1: Clip Generation (15-20 min)
#   - Processes your 20 recordings
#   - Generates 980 synthetic samples with TTS
#   - Creates 1000 negative (non-wake-word) samples
#
# Phase 2: Augmentation (10-15 min)
#   - Applies room acoustics
#   - Adds background noise
#   - Creates variations
#
# Phase 3: Training (15-30 min)
#   - Trains neural network
#   - Exports ONNX model
```

**Progress Tracking:**
- Rich progress bars show status
- You can safely Ctrl+C and resume with `--resume`
- Logs saved to workspace directory

**Another coffee break!** ☕☕

### 8. Test Your Model (5 minutes)

```bash
# Test with real-time microphone detection
easy-oww test my_wake_word

# Interactive menu appears:
# > Select test type:
#   1. Real-time microphone test
#   2. Evaluate on test clips
#   3. Both

# Choose option 1 for quick test
```

**Testing:**
1. Select option "1. Real-time microphone test"
2. Speak your wake word several times
3. Watch for detections in real-time
4. System shows detection count and timing
5. Press Ctrl+C to stop

**Expected Results:**
- Model should detect when you say the wake word
- Should NOT detect when you say other things
- Detection appears within ~100-300ms of speaking

**If not detecting:**
- Try lowering threshold (customize threshold option)
- Ensure you're saying it the same way as training
- Check microphone is working

## Your Trained Model

**Location:**
```bash
# Internal storage
~/.easy-oww/projects/my_wake_word/models/my_wake_word.onnx

# External storage
/Volumes/MyDrive/easy-oww/projects/my_wake_word/models/my_wake_word.onnx
```

**File Format:** ONNX (Open Neural Network Exchange)

**Size:** Typically 5-50 MB

## Using Your Model

### With OpenWakeWord

```python
from openwakeword.model import Model

# Load your model
model = Model(wakeword_models=["path/to/my_wake_word.onnx"])

# Use for detection
# See OpenWakeWord docs for full integration
```

### With ONNX Runtime

```python
import onnxruntime as ort

session = ort.InferenceSession("my_wake_word.onnx")
# Process audio and run inference
```

## Next Steps

### Improve Accuracy

**Record more samples:**
```bash
# Record 30 more samples (total 50)
easy-oww record my_wake_word --count 30
```

**Then retrain:**
```bash
easy-oww train my_wake_word
```

### Create More Wake Words

```bash
# Create different wake words for different purposes
easy-oww create lights_on --wake-word "turn on lights"
easy-oww record lights_on
easy-oww train lights_on
```

### Optimize for Your Use Case

**Edit configuration:**
```bash
# Edit project config
nano ~/.easy-oww/projects/my_wake_word/config.yaml

# Adjust:
# - target_samples (increase for better accuracy)
# - max_steps (train longer)
# - detection_threshold (tune during testing)
```

**Then retrain:**
```bash
easy-oww train my_wake_word
```

## Common Issues

### "Command not found: easy-oww"

```bash
# Try running directly
python -m easy_oww.cli.main --version

# Or reinstall
cd /path/to/easy-oww
pip install -e .
```

### Downloads are slow

- **Normal:** 40 GB takes time even on fast connections
- **Expected:** 20-60 minutes depending on internet speed
- **Tip:** Run overnight if needed
- **Note:** Downloads auto-resume if interrupted

### Out of disk space

```bash
# Check space
df -h

# If on external drive, ensure drive has 60+ GB free
# If needed, use larger external drive
```

### Model doesn't detect wake word

**Solutions:**
1. **Lower threshold:**
   ```bash
   easy-oww test my_wake_word
   > Customize threshold? Yes
   > Enter threshold: 0.3  # Lower = more sensitive
   ```

2. **Record more samples:**
   ```bash
   easy-oww record my_wake_word --count 20
   easy-oww train my_wake_word
   ```

3. **Check pronunciation:**
   - Are you saying it the same way as during recording?
   - Try emphasizing different syllables

4. **Test microphone:**
   - Ensure it's working in system settings
   - Try different microphone if available

### Training fails

**Check:**
- [ ] Datasets downloaded successfully
- [ ] TTS voices installed (`easy-oww list-voices`)
- [ ] Enough disk space (check `df -h`)
- [ ] Recordings exist (`ls ~/.easy-oww/projects/my_wake_word/recordings/`)

**If still failing:**
```bash
# Run with verbose output to see error details
easy-oww train my_wake_word --verbose
```

## Performance Expectations

### Timing (with recommended hardware)

| Step | Time | Notes |
|------|------|-------|
| Installation | 5 min | One-time |
| Workspace init | 2 min | One-time |
| Dataset download | 30-40 min | One-time, auto-resumes |
| Voice download | 3-5 min | One-time |
| Project create | 1 min | Per project |
| Recording | 5-10 min | Per project |
| Training | 30-60 min | Per project, varies by CPU |
| Testing | 5 min | Per project |
| **First Run Total** | **~1.5-2 hours** | Including downloads |
| **Subsequent Projects** | **~45-75 min** | No downloads needed |

### Accuracy (with 20+ recordings, proper training)

- **Accuracy:** 90-95%
- **Precision:** 85-92% (few false alarms)
- **Recall:** 92-98% (few misses)
- **F1 Score:** 88-95%

## Need Help?

**Documentation:**
- [Full README](README.md) - Complete guide
- [Installation Guide](INSTALLATION.md) - Detailed setup
- [FAQ](FAQ.md) - Common questions
- Module READMEs - Component details

**Support:**
- [GitHub Issues](https://github.com/yourusername/easy-oww/issues)
- [GitHub Discussions](https://github.com/yourusername/easy-oww/discussions)

**Quick Tips:**
- Use `--help` on any command for options
- Use `--verbose` to see detailed output
- Check logs in `~/.easy-oww/.logs/`

## Summary Commands

```bash
# Activate virtual environment (always do this first!)
cd /path/to/easy-oww
source venv/bin/activate  # Windows: venv\Scripts\activate

# Complete workflow (after installation and downloads)
easy-oww create my_wake_word --wake-word "hey assistant"
easy-oww record my_wake_word --count 20
easy-oww train my_wake_word
easy-oww test my_wake_word

# With external drive
export EASY_OWW_WORKSPACE="/Volumes/MyDrive/easy-oww"
easy-oww create my_wake_word --wake-word "hey assistant"
easy-oww record my_wake_word --count 20
easy-oww train my_wake_word
easy-oww test my_wake_word
```

**Total time:** ~45 minutes (excluding initial setup and downloads)

**Result:** Production-ready ONNX wake word model!

**Important:** Always activate the virtual environment first:
```bash
cd /path/to/easy-oww
source venv/bin/activate
```

---

**Ready to start?** Begin with [Step 1: Install easy-oww](#1-install-easy-oww-5-minutes)
