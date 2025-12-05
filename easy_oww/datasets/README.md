# Dataset Management

This module provides dataset downloading, caching, and management for training wake word models. It handles large-scale datasets including ACAV100M audio embeddings, MIT Room Impulse Responses, and FSD50K background sounds.

## Overview

The dataset management system enables:

1. **Automated Downloads**: Smart downloading with progress tracking
2. **Cache Management**: Integrity verification and resumable downloads
3. **Multiple Datasets**: ACAV100M, MIT RIR, FSD50K support
4. **Storage Optimization**: Only download what you need
5. **External Storage**: Full support for USB drives and external SSDs

## Components

### DatasetManager (`manager.py`)

Central orchestrator for all dataset operations.

**Features:**
- Unified interface for all datasets
- Download status tracking
- Dataset information and statistics
- Required vs. optional dataset management

**Example:**
```python
from easy_oww.datasets import DatasetManager

# Initialize manager
manager = DatasetManager(
    datasets_dir="~/.easy-oww/datasets",
    cache_dir="~/.easy-oww/.cache"
)

# Show dataset status
manager.show_status()

# Download required datasets
manager.download_required()

# Get dataset info
info = manager.get_dataset_info()
for dataset in info:
    print(f"{dataset['name']}: {dataset['size_gb']}GB - {dataset['description']}")
```

### CacheManager (`cache.py`)

Handles dataset caching and integrity verification.

**Features:**
- Manifest-based cache tracking
- File integrity verification (size, checksum)
- Cache invalidation and cleanup
- Metadata storage

**Example:**
```python
from easy_oww.datasets import CacheManager

# Initialize cache
cache = CacheManager(cache_dir="~/.easy-oww/.cache")

# Check if dataset is cached
if cache.is_cached('acav100m_train'):
    print("Dataset already downloaded")
else:
    print("Need to download dataset")

# Add cache entry after download
cache.add_entry(
    dataset_name='acav100m_train',
    filepath='/path/to/dataset.tar',
    checksum='abc123...',
    metadata={'version': '1.0', 'downloaded_at': '2024-01-01'}
)

# Verify integrity
if cache.verify_integrity('acav100m_train'):
    print("Cache valid")
```

### ACAV100MDownloader (`acav100m_features.py`)

Downloads pre-computed audio embeddings from ACAV100M dataset.

**Dataset Info:**
- **Training Set**: 40GB of negative audio embeddings
- **Validation Set**: 1GB for model validation
- **Source**: ACAV100M (A Context-Aware Audio-Visual dataset)
- **Format**: Pre-computed embeddings (not raw audio)
- **Use**: Negative samples for wake word training

**Features:**
- Split downloads (train/validation)
- Resumable downloads with HTTP range requests
- Integrity verification
- Progress tracking

**Example:**
```python
from easy_oww.datasets import ACAV100MDownloader

# Initialize downloader
downloader = ACAV100MDownloader(datasets_dir="~/.easy-oww/datasets")

# Check if already downloaded
if downloader.is_training_cached():
    print("Training data already available")
else:
    # Download training data (40GB)
    downloader.download_training()

# Download validation data (1GB)
if not downloader.is_validation_cached():
    downloader.download_validation()

# Get paths
train_path = downloader.get_training_path()
val_path = downloader.get_validation_path()
```

**Storage Requirements:**
- Training: ~40GB
- Validation: ~1GB
- Total: ~41GB

### RIRDownloader (`rir.py`)

Downloads MIT Room Impulse Response dataset for acoustic augmentation.

**Dataset Info:**
- **Size**: 2GB
- **Source**: MIT Acoustical Reverberation Scene Statistics Survey
- **Content**: Room impulse responses from real environments
- **Use**: Add realistic room acoustics to training samples

**Features:**
- Single archive download
- Automatic extraction
- Multiple room environments
- Various acoustic properties

**Example:**
```python
from easy_oww.datasets import RIRDownloader

# Initialize downloader
downloader = RIRDownloader(datasets_dir="~/.easy-oww/datasets")

# Download and extract
if not downloader.is_cached():
    downloader.download()

# Get path to RIR files
rir_path = downloader.get_rir_path()
rir_files = list(rir_path.glob("*.wav"))
print(f"Found {len(rir_files)} room impulse responses")
```

**Why Room Impulse Responses?**
- Make synthetic samples sound more natural
- Add reverberation and room acoustics
- Improve model robustness to different environments
- Reduce domain gap between synthetic and real audio

### FSD50kDownloader (`fsd50k.py`)

Downloads FSD50K (Freesound Dataset 50K) for background noise augmentation.

**Dataset Info:**
- **Size**: 30GB
- **Source**: FSD50K from Freesound
- **Content**: 50,000 audio clips across 200 sound categories
- **Use**: Background noise and negative samples

**Features:**
- Large-scale sound dataset
- Diverse audio categories
- Optional (can use ACAV100M instead)
- Automatic download and extraction

**Example:**
```python
from easy_oww.datasets import FSD50kDownloader

# Initialize downloader
downloader = FSD50kDownloader(datasets_dir="~/.easy-oww/datasets")

# Download (optional, large file)
if not downloader.is_cached():
    print("Downloading 30GB dataset...")
    downloader.download()

# Get sound clips
sounds_path = downloader.get_sounds_path()
sound_files = list(sounds_path.glob("*.wav"))
```

**When to Download:**
- You want maximum negative sample diversity
- You have 30GB+ free space
- You're training many different wake words
- You want high-quality background sounds

**When to Skip:**
- Limited storage (<60GB available)
- Only training 1-2 wake words
- ACAV100M provides sufficient negative samples

## Dataset Overview

### Required Datasets

| Dataset | Size | Purpose | Priority |
|---------|------|---------|----------|
| **ACAV100M Train** | 40GB | Negative training samples | Critical |
| **ACAV100M Val** | 1GB | Model validation | Critical |
| **MIT RIR** | 2GB | Acoustic augmentation | Recommended |
| **Total (Required)** | **~43GB** | - | - |

### Optional Datasets

| Dataset | Size | Purpose | Priority |
|---------|------|---------|----------|
| **FSD50K** | 30GB | Additional background sounds | Optional |
| **Total (All)** | **~73GB** | - | - |

## CLI Usage

### Download Required Datasets

```bash
# Download only critical datasets (~43GB)
easy-oww download --required-only

# Shows progress:
# [━━━━━━━━━━━━━━━━━━━━] Downloading ACAV100M Training (40GB)
# [━━━━━━━━━━━━━━━━━━━━] Downloading ACAV100M Validation (1GB)
# [━━━━━━━━━━━━━━━━━━━━] Downloading MIT RIR (2GB)
```

### Download All Datasets

```bash
# Download all datasets including optional (~73GB)
easy-oww download --all

# Includes FSD50K (30GB) for additional diversity
```

### Check Download Status

```bash
# View dataset status
easy-oww download --status

# Output shows:
# ┌─────────────────────────┬────────┬──────────┬────────┐
# │ Dataset                 │ Size   │ Priority │ Status │
# ├─────────────────────────┼────────┼──────────┼────────┤
# │ ACAV100M Training       │ 40GB   │ Critical │ ✓      │
# │ ACAV100M Validation     │ 1GB    │ Critical │ ✓      │
# │ MIT RIR                 │ 2GB    │ Required │ ✓      │
# │ FSD50K                  │ 30GB   │ Optional │ -      │
# └─────────────────────────┴────────┴──────────┴────────┘
```

### External Storage

```bash
# Download to external drive
easy-oww download --required-only --workspace /Volumes/MyDrive/easy-oww

# Or set environment variable
export EASY_OWW_WORKSPACE="/Volumes/MyDrive/easy-oww"
easy-oww download --required-only
```

## Download Features

### Resumable Downloads

All downloads support resuming if interrupted:

```python
# If download is interrupted (Ctrl+C, connection loss, etc.)
# Simply run the command again
easy-oww download --required-only

# Downloads will resume from where they left off
# No need to re-download already completed files
```

### Progress Tracking

Rich progress bars show:
- Download speed (MB/s)
- Estimated time remaining
- Percentage complete
- Current file being downloaded

### Integrity Verification

Automatic verification of:
- File size matches expected size
- Checksum validation (when available)
- Archive extraction validation
- Cache consistency

## Storage Considerations

### Minimum Requirements

For basic functionality:
- **43GB** free space for required datasets
- Internal storage OR external USB 3.0+ drive
- Stable internet connection (downloads take 20-60 minutes)

### Recommended Setup

For optimal experience:
- **60GB+** free space (includes workspace and projects)
- USB-C SSD or fast internal storage
- Multiple projects and wake words

### Storage Optimization Tips

**1. Use External Storage**
```bash
# Keep large datasets on external drive
easy-oww init --workspace /Volumes/MyDrive/easy-oww
easy-oww download --required-only
```

**2. Skip Optional Datasets**
```bash
# Only download critical datasets (saves 30GB)
easy-oww download --required-only
```

**3. Clean Up After Training**
```bash
# Remove intermediate clips after successful training
rm -rf ~/.easy-oww/projects/my_wake_word/clips/
rm -rf ~/.easy-oww/projects/my_wake_word/features/

# Keep only the final model (5-50MB)
# Saves 1-5GB per project
```

**4. Share Datasets Across Projects**
All projects share the same datasets - you only download once!

## Troubleshooting

### Download Fails or Times Out

```bash
# Simply re-run the command - downloads will resume
easy-oww download --required-only

# For verbose output to see what's happening
easy-oww download --required-only --verbose
```

### Out of Disk Space

```bash
# Check available space
df -h ~/.easy-oww

# If internal drive is full, use external drive
easy-oww init --workspace /Volumes/MyDrive/easy-oww
easy-oww download --required-only --workspace /Volumes/MyDrive/easy-oww
```

### Slow Download Speeds

**Expected speeds:**
- Fast connection (100+ Mbps): 10-20 MB/s
- Average connection (50 Mbps): 5-10 MB/s
- Slow connection (10 Mbps): 1-2 MB/s

**40GB download time:**
- Fast: 30-60 minutes
- Average: 1-2 hours
- Slow: 3-6 hours

**Tips:**
- Download during off-peak hours
- Use wired connection instead of Wi-Fi
- Close other bandwidth-heavy applications

### Cache Corruption

```bash
# Clear cache and re-download
rm -rf ~/.easy-oww/.cache/
easy-oww download --required-only

# This will re-download datasets
# Make sure you have enough space
```

### Extraction Fails

```bash
# If dataset downloaded but extraction failed:
# 1. Check disk space
df -h

# 2. Manually extract
cd ~/.easy-oww/datasets/
tar -xzf acav100m_train.tar.gz

# 3. Verify extraction
ls -lh acav100m/
```

## Advanced Usage

### Custom Dataset Paths

```python
from easy_oww.datasets import ACAV100MDownloader

# Use custom dataset directory
downloader = ACAV100MDownloader(
    datasets_dir="/Volumes/External/datasets"
)

downloader.download_training()
```

### Parallel Downloads

```python
from easy_oww.datasets import DatasetManager
import concurrent.futures

manager = DatasetManager(
    datasets_dir="~/.easy-oww/datasets",
    cache_dir="~/.easy-oww/.cache"
)

# Download multiple datasets in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(manager.acav100m.download_training),
        executor.submit(manager.rir.download),
    ]

    for future in concurrent.futures.as_completed(futures):
        future.result()
```

### Custom Cache Location

```python
from easy_oww.datasets import CacheManager

# Use custom cache directory
cache = CacheManager(cache_dir="/tmp/easy-oww-cache")

# Useful for testing or temporary storage
```

## Dataset Details

### ACAV100M Features

**What are audio embeddings?**
- Pre-computed feature vectors from audio
- Extracted using deep learning models
- Much smaller than raw audio
- Ready for model training

**Why embeddings instead of raw audio?**
- Faster training (no feature extraction needed)
- Smaller storage (40GB vs. several TB)
- Pre-processed and normalized
- Compatible with OpenWakeWord architecture

### MIT Room Impulse Responses

**What is a room impulse response?**
- Recording of how sound behaves in a room
- Captures reverberation and echo
- Measured in real environments

**How are RIRs used?**
- Convolve with clean audio
- Adds realistic room acoustics
- Makes synthetic speech sound natural
- Improves model robustness

### FSD50K Sounds

**What's in FSD50K?**
- 50,000 audio clips
- 200 sound categories
- Real-world recordings
- Creative Commons licensed

**Example categories:**
- Speech, music, animals
- Vehicles, tools, nature
- Indoor/outdoor sounds
- Various noise types

## Testing

Run dataset module tests:

```bash
pytest tests/test_datasets.py -v
```

## Future Enhancements

Planned features:
- Mirror site support for faster downloads
- Torrent-based distribution
- Dataset versioning and updates
- Automatic dataset pruning (remove unused)
- Custom dataset integration
- Dataset statistics and analysis tools
