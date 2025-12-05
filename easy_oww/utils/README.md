# Utilities Module

This module provides core utilities for easy-oww including path management, system checks, logging, and progress tracking. These utilities are used throughout the application to maintain consistency and provide a polished user experience.

## Overview

The utilities module enables:

1. **Path Management**: Centralized workspace and project path handling
2. **System Checks**: Verify requirements before operations
3. **Logging**: Consistent logging across all modules
4. **Progress Tracking**: Rich terminal progress bars and status updates

## Components

### PathManager (`paths.py`)

Centralized path management for workspace, projects, datasets, and voices.

**Features:**
- Workspace directory structure management
- Project path resolution
- Dataset and voice path handling
- External storage support
- Environment variable integration

**Example:**
```python
from easy_oww.utils import PathManager

# Initialize with default workspace (~/.easy-oww)
paths = PathManager()

# Or use custom workspace (e.g., external drive)
paths = PathManager(workspace="/Volumes/MyDrive/easy-oww")

# Create workspace structure
paths.ensure_structure()

# Get project path
project_path = paths.get_project_path("my_wake_word")
print(project_path)  # ~/.easy-oww/projects/my_wake_word

# Check if project exists
if paths.project_exists("my_wake_word"):
    print("Project found!")

# Create project structure
paths.create_project_structure("new_project")
# Creates: recordings/, clips/, features/, models/

# Get dataset paths
acav_path = paths.get_dataset_path("acav100m")
rir_path = paths.get_dataset_path("rir")

# Get voice model path
voice_path = paths.get_voice_path("en_US-lessac-medium")
```

**Workspace Structure:**
```
~/.easy-oww/                    # Main workspace
├── datasets/                   # Downloaded datasets
│   ├── acav100m/              # ACAV100M embeddings
│   ├── rir/                   # Room impulse responses
│   └── fsd50k/                # Background sounds
├── projects/                   # User projects
│   └── my_wake_word/          # Example project
│       ├── recordings/        # User recordings
│       ├── clips/             # Generated clips
│       ├── features/          # Extracted features
│       ├── models/            # Trained models
│       └── config.yaml        # Project config
├── voices/                     # TTS voice models
│   ├── en_US-lessac-medium.onnx
│   └── en_US-amy-medium.onnx
├── piper-sample-generator/     # Piper TTS binary
└── .cache/                     # Download cache
    └── manifest.json
```

**Environment Variable Support:**
```bash
# Set default workspace via environment variable
export EASY_OWW_WORKSPACE="/Volumes/MyDrive/easy-oww"

# PathManager will automatically use this path
# No need to specify --workspace flag
```

**Properties:**
- `workspace`: Main workspace directory (Path)
- `datasets`: Datasets directory (Path)
- `projects`: Projects directory (Path)
- `voices`: Voice models directory (Path)
- `piper`: Piper binary directory (Path)
- `cache`: Cache directory (Path)

### SystemChecker (`system.py`)

Verify system requirements before operations.

**Features:**
- Python version verification
- Disk space checking
- RAM availability checking
- GPU detection (CUDA)
- Platform detection
- Requirements validation

**Example:**
```python
from easy_oww.utils import SystemChecker

checker = SystemChecker()

# Check Python version
valid, version = checker.check_python_version()
if valid:
    print(f"Python {version} - OK")
else:
    print(f"Python {version} - Need 3.7+")

# Check disk space
has_space, available = checker.check_disk_space(
    path="/Users/username",
    required_gb=60
)
print(f"Available: {available:.1f}GB")

# Check RAM
has_memory, available = checker.check_memory(required_gb=8)
print(f"Available RAM: {available:.1f}GB")

# Check GPU
gpu_info = checker.check_gpu()
if gpu_info['available']:
    print(f"GPU: {gpu_info['device_name']}")
    print(f"Count: {gpu_info['device_count']}")
else:
    print("No GPU available (CPU training will be slower)")

# Get platform info
platform = checker.get_platform_info()
print(f"Platform: {platform['system']}")
print(f"Version: {platform['version']}")
print(f"Architecture: {platform['machine']}")

# Run all checks
results = checker.run_all_checks()
if results['all_passed']:
    print("All system requirements met!")
else:
    print("Issues found:")
    for issue in results['issues']:
        print(f"  - {issue}")
```

**System Requirements:**
- **Python**: 3.7 or higher (3.9+ recommended)
- **Disk Space**: 60GB+ free (43GB for datasets, rest for workspace)
- **RAM**: 8GB+ recommended (4GB minimum)
- **GPU**: Optional but recommended (50-70% faster training)

**Platform Support:**
- macOS (Intel and Apple Silicon)
- Linux (Ubuntu, Debian, Fedora, etc.)
- Windows 10/11

### Logger Setup (`logger.py`)

Consistent logging configuration with rich terminal output.

**Features:**
- Rich terminal formatting
- Console and file logging
- Configurable log levels
- Colored output
- Traceback formatting
- Timestamp support

**Example:**
```python
from easy_oww.utils import setup_logger, get_logger
from pathlib import Path

# Setup logger with console and file output
logger = setup_logger(
    name='easy-oww',
    level=logging.INFO,
    log_file=Path('~/.easy-oww/.logs/easy-oww.log')
)

# Log messages
logger.info("Starting training pipeline...")
logger.warning("No GPU detected, using CPU")
logger.error("Failed to load dataset")
logger.debug("Augmentation parameters: snr=20db, rir_prob=0.5")

# Get logger from anywhere in the application
logger = get_logger('easy-oww')
logger.info("This is a log message")

# Use rich markup
logger.info("[bold green]✓[/bold green] Training completed successfully")
logger.warning("[yellow]⚠[/yellow] Low disk space detected")
logger.error("[bold red]✗[/bold red] Failed to save model")
```

**Log Levels:**
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages (default)
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for failures
- `CRITICAL`: Critical errors requiring immediate attention

**Log Output:**
```
[INFO] Starting training pipeline...
[WARNING] No GPU detected, using CPU
[ERROR] Failed to load dataset
```

**File Logging:**
Logs are saved to `~/.easy-oww/.logs/easy-oww.log` with timestamps:
```
2024-01-15 10:30:45 - easy-oww - INFO - Starting training pipeline...
2024-01-15 10:30:46 - easy-oww - WARNING - No GPU detected, using CPU
2024-01-15 10:30:47 - easy-oww - ERROR - Failed to load dataset
```

### ProgressTracker (`progress.py`)

Rich terminal progress bars and status updates.

**Features:**
- Multiple progress bars
- Spinner animations
- Percentage indicators
- Time remaining estimates
- Transfer speed (for downloads)
- Console logging during progress
- Context manager support

**Example:**
```python
from easy_oww.utils import ProgressTracker
import time

# Create progress tracker
tracker = ProgressTracker()

# Use as context manager
with tracker:
    # Add tasks
    task1 = tracker.add_task("Downloading dataset...", total=100)
    task2 = tracker.add_task("Processing audio...", total=50)

    # Update progress
    for i in range(100):
        tracker.update(task1, advance=1)
        time.sleep(0.1)

    for i in range(50):
        tracker.update(task2, advance=1)
        time.sleep(0.05)

    # Log messages during progress
    tracker.log("✓ Download complete", style="green")

# Progress bars automatically clean up on exit
```

**Download Progress:**
```python
from easy_oww.utils import ProgressTracker

tracker = ProgressTracker()

with tracker:
    task = tracker.add_task(
        "Downloading ACAV100M...",
        total=40 * 1024 * 1024 * 1024  # 40GB in bytes
    )

    # Update with downloaded bytes
    while downloading:
        bytes_downloaded = get_download_progress()
        tracker.update(task, completed=bytes_downloaded)
```

**Output Example:**
```
⠋ Downloading dataset...    [━━━━━━━━━━━━━━━━━━━━]  45% 0:02:30
⠹ Processing audio...       [━━━━━━━━━━          ]  20% 0:01:15
```

**Indeterminate Progress:**
```python
# For tasks without known total
task = tracker.add_task("Searching for files...", total=None)

# Shows spinner without percentage
⠋ Searching for files...
```

## CLI Integration

The utilities are used throughout the CLI:

### Path Management

```bash
# All commands respect --workspace flag
easy-oww init --workspace /Volumes/MyDrive/easy-oww
easy-oww create my_project --workspace /Volumes/MyDrive/easy-oww

# Or use environment variable
export EASY_OWW_WORKSPACE="/Volumes/MyDrive/easy-oww"
easy-oww create my_project
```

### System Checks

```bash
# Automatic system check on init
easy-oww init

# Output:
# Checking system requirements...
# ✓ Python 3.11.5
# ✓ Disk space: 120GB available
# ✓ RAM: 16GB available
# ✓ GPU: NVIDIA GeForce RTX 3080
```

### Logging

```bash
# Use --verbose for debug logging
easy-oww train my_project --verbose

# Logs saved to ~/.easy-oww/.logs/easy-oww.log
```

### Progress Tracking

All long-running operations show progress:
```bash
easy-oww download --required-only
# Shows download progress with speed and time remaining

easy-oww train my_project
# Shows training progress through phases
```

## Advanced Usage

### Custom Workspace Management

```python
from easy_oww.utils import PathManager
from pathlib import Path

class CustomPathManager(PathManager):
    """Extended path manager with custom locations"""

    def __init__(self, workspace: str):
        super().__init__(workspace)
        # Add custom paths
        self.models_backup = Path("/backup/models")
        self.temp_clips = Path("/tmp/clips")

    def backup_model(self, project_name: str):
        """Backup trained model"""
        model_path = self.get_project_path(project_name) / "models"
        backup_path = self.models_backup / project_name
        shutil.copytree(model_path, backup_path)
```

### System Requirements Validation

```python
from easy_oww.utils import SystemChecker

def validate_before_training():
    """Check requirements before starting training"""
    checker = SystemChecker()

    # Check Python
    valid_python, version = checker.check_python_version()
    if not valid_python:
        raise RuntimeError(f"Python 3.7+ required, got {version}")

    # Check disk space (need 10GB for training)
    has_space, available = checker.check_disk_space(
        path="~/.easy-oww",
        required_gb=10
    )
    if not has_space:
        raise RuntimeError(f"Need 10GB free, only {available:.1f}GB available")

    # Check RAM (need 4GB)
    has_memory, available = checker.check_memory(required_gb=4)
    if not has_memory:
        raise RuntimeError(f"Need 4GB RAM, only {available:.1f}GB available")

    # Check GPU (optional but log warning)
    gpu_info = checker.check_gpu()
    if not gpu_info['available']:
        print("Warning: No GPU detected, training will be slower")

    return True

# Use before training
if validate_before_training():
    start_training()
```

### Advanced Progress Tracking

```python
from easy_oww.utils import ProgressTracker
from pathlib import Path

def process_audio_files(audio_files: list):
    """Process multiple audio files with progress tracking"""
    tracker = ProgressTracker()

    with tracker:
        # Main progress
        main_task = tracker.add_task(
            "Processing audio files",
            total=len(audio_files)
        )

        for audio_file in audio_files:
            # Sub-task for each file
            file_size = Path(audio_file).stat().st_size
            file_task = tracker.add_task(
                f"Processing {audio_file.name}",
                total=file_size
            )

            # Process with progress updates
            for chunk in process_file_chunks(audio_file):
                tracker.update(file_task, advance=len(chunk))

            # Complete file
            tracker.update(main_task, advance=1)
            tracker.log(f"✓ Processed {audio_file.name}", style="green")
```

### Multi-Stage Progress

```python
from easy_oww.utils import ProgressTracker

def train_with_progress():
    """Training with multi-stage progress tracking"""
    tracker = ProgressTracker()

    with tracker:
        # Stage 1: Clip generation
        tracker.log("[bold]Stage 1: Generating clips[/bold]")
        clip_task = tracker.add_task("Generating clips", total=1000)
        for i in range(1000):
            generate_clip()
            tracker.update(clip_task, advance=1)
        tracker.log("✓ Clip generation complete", style="green")

        # Stage 2: Augmentation
        tracker.log("[bold]Stage 2: Augmenting audio[/bold]")
        aug_task = tracker.add_task("Augmenting clips", total=1000)
        for i in range(1000):
            augment_clip()
            tracker.update(aug_task, advance=1)
        tracker.log("✓ Augmentation complete", style="green")

        # Stage 3: Training
        tracker.log("[bold]Stage 3: Training model[/bold]")
        train_task = tracker.add_task("Training", total=10000)
        for step in range(10000):
            train_step()
            tracker.update(train_task, advance=1)
        tracker.log("✓ Training complete", style="green")
```

## Best Practices

### Path Management

1. **Always use PathManager** instead of hardcoded paths
2. **Check existence** before operations
3. **Create structure** at initialization
4. **Support external storage** by accepting custom workspace

```python
# Good
paths = PathManager(workspace)
project_path = paths.get_project_path(name)
if not project_path.exists():
    paths.create_project_structure(name)

# Bad
project_path = f"~/.easy-oww/projects/{name}"
os.makedirs(project_path)  # May fail on external drives
```

### System Checks

1. **Check early** before long operations
2. **Provide helpful errors** with actual values
3. **Make GPU optional** - don't require it

```python
# Good
checker = SystemChecker()
has_space, available = checker.check_disk_space(path, required_gb=10)
if not has_space:
    raise RuntimeError(
        f"Need 10GB free space, only {available:.1f}GB available. "
        f"Consider using external drive: --workspace /Volumes/MyDrive"
    )

# Bad
if not has_disk_space():
    raise RuntimeError("Not enough space")  # Not helpful!
```

### Logging

1. **Use appropriate levels** - don't log everything as INFO
2. **Include context** in messages
3. **Use rich markup** sparingly

```python
# Good
logger.info(f"Training {project_name} with {sample_count} samples")
logger.warning(f"Low SNR ({snr}db) in {audio_file}")
logger.error(f"Failed to load model from {model_path}: {error}")

# Bad
logger.info("Training")  # Not enough context
logger.error("Error")     # No details
```

### Progress Tracking

1. **Show progress for >5 second operations**
2. **Use descriptive task names**
3. **Update frequently** but not too frequently (every 0.1s minimum)
4. **Clean up** with context manager

```python
# Good
with ProgressTracker() as tracker:
    task = tracker.add_task("Downloading ACAV100M (40GB)", total=bytes)
    # Updates every chunk

# Bad
tracker = ProgressTracker()
task = tracker.add_task("Processing", total=None)  # Vague description
# Never updates, never cleans up
```

## Testing

Run utils module tests:

```bash
pytest tests/test_utils.py -v
```

## Future Enhancements

Planned features:
- Configuration file management
- Backup and restore utilities
- Workspace migration tools
- Cross-platform path normalization
- Resource monitoring dashboard
- Automatic cleanup utilities
- Project archiving
- Cloud storage integration
