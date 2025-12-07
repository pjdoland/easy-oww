# Installation Guide

Complete installation instructions for easy-oww, including external storage setup.

## Table of Contents

- [System Requirements](#system-requirements)
- [Standard Installation](#standard-installation)
- [External Storage Setup](#external-storage-setup)
- [Dependency Installation](#dependency-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **Python** | 3.7 or higher | Python 3.9+ recommended |
| **Storage** | 20GB free | For required datasets only |
| **RAM** | 4GB | 8GB+ recommended for training |
| **Microphone** | Any USB or built-in | For recording samples |
| **Internet** | Stable connection | For downloading datasets (~40GB) |

### Recommended Requirements

| Component | Requirement | Benefit |
|-----------|-------------|---------|
| **Python** | 3.9 - 3.11 | Better compatibility |
| **Storage** | 40GB+ free | For all datasets + workspace |
| **RAM** | 8GB+ | Faster training, larger batches |
| **GPU** | CUDA-compatible | 50-70% faster training |
| **SSD** | Any SSD | Faster I/O operations |

### Storage Breakdown

Understanding where the space goes:

| Component | Size | Required | Description |
|-----------|------|----------|-------------|
| **ACAV100M Features** | 17.5GB | Yes | Pre-computed audio embeddings for training |
| **MIT RIR Dataset** | 2GB | Yes | Room impulse responses for augmentation |
| **FSD50K Dataset** | 30GB | No | Background sounds and negative samples |
| **TTS Voices** | ~200MB | Yes | 2-3 voice models for synthesis |
| **Piper Binary** | ~50MB | Yes | TTS engine binary |
| **Workspace** | ~1-5GB | Yes | Projects, models, recordings |
| **Total (Required)** | **~20GB** | - | Minimum for basic functionality |
| **Total (Recommended)** | **~52GB** | - | For full feature set |

## Standard Installation

### 1. Install Python

**Check Current Version:**
```bash
python3 --version
```

**If Python 3.7+ is not installed:**

**On macOS:**
```bash
# Using Homebrew
brew install python@3.11

# Or download from python.org
# https://www.python.org/downloads/macos/
```

**On Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**On Windows:**
- Download from https://www.python.org/downloads/windows/
- Ensure "Add Python to PATH" is checked during installation

### 2. Clone Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/easy-oww.git

# Navigate to directory
cd easy-oww

# Verify files
ls -la
```

### 3. Set Up Virtual Environment and Install

**Method 1: Automatic Setup with Script (Recommended)**
```bash
# Run the setup script
./setup.sh

# This will:
# - Check Python version
# - Create virtual environment
# - Install all dependencies
# - Verify installation
```

**Method 2: Manual Setup**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install easy-oww in development mode
pip install -e .
```

**Method 3: Install from requirements.txt**
```bash
# Create and activate virtual environment first
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Then run directly with:
python -m easy_oww.cli.main
```

### 4. Verify Installation

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Windows: venv\Scripts\activate

# Check that command is available
easy-oww --version

# Should output: easy-oww, version 0.1.0

# Check help
easy-oww --help
```

**Important:** Always activate the virtual environment before using easy-oww:
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

## External Storage Setup

**Yes! You can absolutely run easy-oww from a USB-C thumb drive or external SSD.** This is ideal for users with limited internal storage.

### Why Use External Storage?

- **Save Internal Space**: Keep your main drive free
- **Portability**: Move your workspace between computers
- **Organization**: Keep all datasets and projects in one place
- **Flexibility**: Easy to upgrade storage by getting a larger drive

### Recommended External Drives

**Minimum Specifications:**
- **Capacity**: 64GB or larger
- **Interface**: USB 3.0 or USB-C (USB 2.0 will be very slow)
- **Type**: SSD strongly recommended over flash drive

**Performance Comparison:**

| Drive Type | Read/Write Speed | Training Time | Dataset Download |
|------------|------------------|---------------|------------------|
| USB 2.0 Flash | 10-30 MB/s | 3-4x slower | 2-3 hours |
| USB 3.0 Flash | 60-150 MB/s | 1.5-2x slower | 30-60 min |
| USB-C SSD | 400-1000 MB/s | Similar to internal | 10-20 min |

**Recommended Options:**
- **Budget**: SanDisk Extreme PRO USB 3.1 (128GB+)
- **Better**: Samsung T7 Portable SSD (500GB+)
- **Best**: Samsung T7 Shield or SanDisk Extreme PRO SSD (1TB)

### Setup Process

#### Step 1: Format the Drive (Optional but Recommended)

**On macOS:**
```bash
# 1. Open Disk Utility
# 2. Select your external drive
# 3. Click "Erase"
# 4. Choose format:
#    - macOS only: APFS
#    - Cross-platform: exFAT
# 5. Name it "easy-oww" or similar
```

**On Linux:**
```bash
# Find the drive
lsblk

# Format as ext4 (Linux) or exfat (cross-platform)
sudo mkfs.ext4 /dev/sdX1  # Replace sdX1 with your drive
# OR
sudo mkfs.exfat /dev/sdX1

# Mount the drive
sudo mount /dev/sdX1 /mnt/easy-oww
```

**On Windows:**
```bash
# 1. Open "Disk Management"
# 2. Right-click your drive
# 3. Select "Format"
# 4. Choose:
#    - File System: exFAT (for compatibility)
#    - Allocation size: Default
# 5. Quick Format: Checked
```

#### Step 2: Set Up Workspace on External Drive

```bash
# Create workspace directory on external drive
# Replace /Volumes/easy-oww with your drive path

# macOS
mkdir -p /Volumes/easy-oww/workspace

# Linux
mkdir -p /mnt/easy-oww/workspace

# Windows (in PowerShell)
mkdir E:\workspace
```

#### Step 3: Initialize with Custom Path

```bash
# Initialize workspace on external drive
# macOS
easy-oww init --workspace /Volumes/easy-oww/workspace

# Linux
easy-oww init --workspace /mnt/easy-oww/workspace

# Windows
easy-oww init --workspace E:\workspace
```

#### Step 4: Download Datasets to External Drive

```bash
# All subsequent commands use the --workspace flag
# macOS
easy-oww download --required-only --workspace /Volumes/easy-oww/workspace
easy-oww download-voices --workspace /Volumes/easy-oww/workspace

# Linux
easy-oww download --required-only --workspace /mnt/easy-oww/workspace

# Windows
easy-oww download --required-only --workspace E:\workspace
```

### Setting Default Workspace Path

To avoid typing `--workspace` every time, set an environment variable:

**On macOS/Linux (add to ~/.bashrc or ~/.zshrc):**
```bash
export EASY_OWW_WORKSPACE="/Volumes/easy-oww/workspace"

# Then you can run commands without --workspace flag:
easy-oww create my_project
easy-oww record my_project
easy-oww train my_project
```

**On Windows (PowerShell, add to profile):**
```powershell
$env:EASY_OWW_WORKSPACE = "E:\workspace"

# Or set permanently:
[System.Environment]::SetEnvironmentVariable('EASY_OWW_WORKSPACE', 'E:\workspace', 'User')
```

### External Drive Best Practices

#### 1. **Keep Drive Connected**
- Don't unplug during downloads or training
- Use a USB hub with individual power switches if needed

#### 2. **Safely Eject**
```bash
# macOS
diskutil eject /Volumes/easy-oww

# Linux
sudo umount /mnt/easy-oww

# Windows: Use "Safely Remove Hardware"
```

#### 3. **Backup Important Files**
- Copy trained models to internal storage or cloud
- External drives can fail; back up your recordings

#### 4. **Speed Optimization**
```bash
# Check drive speed
# macOS
diskutil info /dev/diskX

# Linux
sudo hdparm -Tt /dev/sdX

# If slow, ensure:
# - Using USB 3.0+ port (usually blue or marked with SS)
# - Drive is formatted correctly (APFS/ext4/NTFS, not FAT32)
# - USB cable is high quality
```

#### 5. **Auto-Mount on Startup (Optional)**

**macOS:**
- Drive should auto-mount when connected
- Check System Preferences > Users & Groups > Login Items

**Linux (add to /etc/fstab):**
```bash
UUID=your-drive-uuid /mnt/easy-oww ext4 defaults,nofail 0 2

# Find UUID with:
sudo blkid
```

**Windows:**
- Assign a permanent drive letter in Disk Management

### Hybrid Setup (Internal + External)

For optimal performance, you can use a hybrid approach:

```bash
# Keep Python packages and code on internal drive
cd ~/Projects/easy-oww
pip install -e .

# Store datasets and workspaces on external drive
easy-oww init --workspace /Volumes/external/easy-oww
easy-oww download --workspace /Volumes/external/easy-oww

# Projects on external drive
easy-oww create project1 --workspace /Volumes/external/easy-oww

# But keep trained models on fast internal drive
cp /Volumes/external/easy-oww/projects/project1/models/*.onnx ~/models/
```

## Dependency Installation

### Core Dependencies

All installed automatically with `pip install -e .`:

```
openwakeword>=0.5.0    # Wake word detection framework
sounddevice>=0.4.6     # Audio recording
numpy>=1.21.0          # Numerical computing
scipy>=1.7.0           # Signal processing
pyyaml>=6.0            # Configuration files
requests>=2.28.0       # HTTP downloads
tqdm>=4.64.0           # Progress bars
click>=8.1.0           # CLI framework
questionary>=1.10.0    # Interactive prompts
rich>=13.0.0           # Rich terminal output
psutil>=5.9.0          # System monitoring
```

### Optional Dependencies

**For GPU Training:**
```bash
pip install torch>=2.0.0
pip install onnxruntime-gpu>=1.15.0

# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available())"
```

**For Development:**
```bash
pip install -e ".[dev]"

# Installs:
# - pytest (testing)
# - pytest-cov (coverage)
# - black (code formatting)
# - flake8 (linting)
```

**For Audio Format Support:**
```bash
# MP3 support (optional, for custom datasets)
pip install pydub

# FLAC support
pip install soundfile
```

### Platform-Specific Requirements

**macOS:**
```bash
# Install PortAudio (for sounddevice)
brew install portaudio

# If you get "xcrun: error" during installation
sudo xcode-select --install
```

**Linux (Ubuntu/Debian):**
```bash
# Install PortAudio
sudo apt-get install portaudio19-dev python3-pyaudio

# Install ALSA libraries (if needed)
sudo apt-get install libasound2-dev
```

**Windows:**
- Most dependencies work out of the box
- If sounddevice fails, download wheel from:
  https://www.lfd.uci.edu/~gohlke/pythonlibs/#sounddevice

### Troubleshooting Dependencies

**Issue: sounddevice import error**
```bash
# Solution 1: Reinstall with pip
pip uninstall sounddevice
pip install sounddevice --no-cache-dir

# Solution 2: Install system libraries (Linux)
sudo apt-get install libportaudio2

# Solution 3: Use conda
conda install -c conda-forge sounddevice
```

**Issue: numpy/scipy import error**
```bash
# Update pip first
pip install --upgrade pip

# Then reinstall
pip install numpy scipy --upgrade
```

**Issue: Permission denied**
```bash
# Use --user flag
pip install --user -e .

# Or use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Verification

### 1. Test Command Line Tool

```bash
# Check version
easy-oww --version

# Check help
easy-oww --help

# Should see list of commands:
# - init
# - download
# - download-voices
# - create
# - record
# - train
# - test
# - list
```

### 2. Test Python Import

```bash
python3 << EOF
# Test imports
from easy_oww.audio import AudioRecorder
from easy_oww.tts import PiperTTS
from easy_oww.training import TrainingConfig
from easy_oww.testing import ModelDetector
print("✓ All imports successful")
EOF
```

### 3. Test System Requirements

```bash
# Run init to check system
easy-oww init --workspace /tmp/test_workspace

# Should show:
# - Python version (✓ if 3.7+)
# - Disk space
# - RAM
# - GPU (if available)

# Clean up test
rm -rf /tmp/test_workspace
```

### 4. Test Microphone

```bash
python3 << EOF
from easy_oww.audio import AudioRecorder
recorder = AudioRecorder()
devices = recorder.list_devices()
print(f"✓ Found {len(devices)} audio input device(s)")
for d in devices:
    print(f"  - {d['name']}")
EOF
```

### 5. Quick Functionality Test

```bash
# Create temporary workspace
mkdir -p /tmp/easy-oww-test
cd /tmp/easy-oww-test

# Test project creation
easy-oww init --workspace .
easy-oww create test_project --wake-word "test" --workspace .
easy-oww list --workspace .

# Should show test_project in list

# Clean up
cd -
rm -rf /tmp/easy-oww-test
```

## Virtual Environment Setup

**Virtual environments are now the default installation method.** The setup script creates one automatically, or you can create it manually.

### Why Virtual Environments?

- **Isolation**: Dependencies don't conflict with other Python projects
- **Cleanliness**: Doesn't modify system Python installation
- **Reproducibility**: Easy to recreate exact environment
- **Safety**: Required on many systems (macOS with Homebrew Python)

### Automatic Setup

The easiest way is to use the setup script:

```bash
./setup.sh
```

### Manual Setup

If you prefer manual control:

```bash
# Navigate to project directory
cd easy-oww

# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Your prompt should change to show (venv)

# Install easy-oww
pip install --upgrade pip
pip install -e .

# Verify
easy-oww --version
```

### Daily Usage

**Always activate before using:**
```bash
# Navigate to project directory
cd /path/to/easy-oww

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Now you can use easy-oww
easy-oww init
easy-oww download

# Deactivate when done (optional)
deactivate
```

### Auto-Activation (Optional)

**Add to ~/.bashrc or ~/.zshrc:**
```bash
# Create an alias for easy activation
alias easy-oww-env='cd ~/path/to/easy-oww && source venv/bin/activate'

# Then just run:
easy-oww-env
```

**Or use direnv for automatic activation:**
```bash
# Install direnv: https://direnv.net/

# In easy-oww directory, create .envrc:
echo "source venv/bin/activate" > .envrc

# Allow it:
direnv allow

# Now venv activates automatically when you cd into the directory
```

## Troubleshooting

### Common Issues

#### "Command not found: easy-oww"

**Solution 1: Ensure installation completed**
```bash
cd /path/to/easy-oww
pip install -e .
```

**Solution 2: Check PATH**
```bash
# Find where pip installed it
pip show easy-oww

# Add to PATH if needed (add to ~/.bashrc):
export PATH="$HOME/.local/bin:$PATH"
```

**Solution 3: Run directly**
```bash
python -m easy_oww.cli.main --help
```

#### "ModuleNotFoundError"

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Or reinstall package
pip uninstall easy-oww
pip install -e .
```

#### "Permission denied" errors

```bash
# Option 1: Use --user
pip install --user -e .

# Option 2: Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

#### External drive not recognized

```bash
# macOS: Check mount
df -h | grep Volumes

# Linux: Check mount
mount | grep /mnt

# Windows: Check drive letter
dir E:\
```

#### Slow performance on external drive

```bash
# Check connection type
# USB 3.0 ports are usually blue or have "SS" marking

# Test drive speed
# Should be >100 MB/s for USB 3.0
# macOS:
diskutil activity

# Linux:
sudo hdparm -Tt /dev/sdX
```

### Getting Help

If you encounter issues:

1. **Check logs**:
   ```bash
   # Logs are in workspace directory
   cat ~/.easy-oww/.logs/easy-oww.log
   ```

2. **Run with verbose flag**:
   ```bash
   easy-oww init --verbose
   easy-oww download --verbose
   ```

3. **Check GitHub Issues**:
   https://github.com/yourusername/easy-oww/issues

4. **Create new issue** with:
   - Operating system and version
   - Python version
   - Error message (full traceback)
   - Steps to reproduce

## Next Steps

After successful installation:

1. **Initialize workspace**: `easy-oww init`
2. **Read the main README**: For workflow guide
3. **Download datasets**: `easy-oww download --required-only`
4. **Create your first project**: `easy-oww create my_wake_word`

See [README.md](README.md) for the complete usage guide.
