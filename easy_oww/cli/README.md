# Command Line Interface (CLI)

This module provides the command-line interface for easy-oww, built with Click and Rich for a beautiful and intuitive user experience.

## Overview

The CLI provides a complete workflow for creating wake word models:

1. **init** - Initialize workspace and check system requirements
2. **download** - Download required datasets
3. **download-voices** - Download TTS voice models
4. **create** - Create new wake word project
5. **record** - Record wake word samples
6. **train** - Train wake word model
7. **test** - Test trained model
8. **list** - List projects and voices

## Architecture

### Main Entry Point (`main.py`)

Defines the main CLI group and global options.

**Features:**
- Version information (`--version`)
- Verbose logging (`--verbose`)
- Context passing between commands
- Help system

**Structure:**
```python
@click.group()
@click.version_option()
@click.option('--verbose', '-v')
def cli(ctx, verbose):
    """Main CLI group"""
    pass

@cli.command()
def init(...):
    """Initialize workspace"""
    pass

# ... more commands
```

### Command Implementations (`commands.py`)

Implements the actual command logic with rich formatting.

**Features:**
- Rich console output with colors and formatting
- Progress bars and status indicators
- Interactive prompts
- Error handling and validation
- System checks before operations

## Commands

### init - Initialize Workspace

Initialize easy-oww workspace and verify system requirements.

**Usage:**
```bash
# Initialize in default location (~/.easy-oww)
easy-oww init

# Initialize on external drive
easy-oww init --workspace /Volumes/MyDrive/easy-oww

# Initialize with verbose output
easy-oww init --verbose
```

**What It Does:**
1. Displays welcome message and overview
2. Checks system requirements:
   - Python version (3.7+ required)
   - Disk space (60GB+ recommended)
   - RAM (8GB+ recommended)
   - GPU availability (optional)
3. Creates workspace directory structure
4. Installs Piper TTS (with user confirmation)
5. Shows next steps

**Output:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Welcome to easy-oww!                     │
│                                                             │
│  This tool will guide you through creating custom wake     │
│  word models.                                               │
└─────────────────────────────────────────────────────────────┘

Checking system requirements...

┌───────────────┬────────┬───────────────────────────┐
│ Check         │ Status │ Details                   │
├───────────────┼────────┼───────────────────────────┤
│ Python Version│ ✓      │ 3.11.5 (OK)               │
│ Disk Space    │ ✓      │ 120GB available (OK)      │
│ RAM           │ ✓      │ 16GB available (OK)       │
│ GPU           │ ✓      │ NVIDIA RTX 3080 detected  │
└───────────────┴────────┴───────────────────────────┘

Creating workspace at: ~/.easy-oww
✓ Workspace initialized successfully!

Next steps:
  1. Download datasets: easy-oww download
  2. Download TTS voices: easy-oww download-voices
  3. Create a project: easy-oww create <project_name>
```

**Options:**
- `--workspace, -w`: Custom workspace path
- `--verbose, -v`: Enable verbose logging

### download - Download Datasets

Download required datasets for training.

**Usage:**
```bash
# Download only required datasets (~43GB)
easy-oww download --required-only

# Download all datasets including optional (~73GB)
easy-oww download --all

# Show download status
easy-oww download --status

# Download to external drive
easy-oww download --required-only --workspace /Volumes/MyDrive/easy-oww
```

**What It Does:**
1. Shows dataset information and sizes
2. Downloads required datasets:
   - ACAV100M Training (40GB)
   - ACAV100M Validation (1GB)
   - MIT RIR (2GB)
3. Optionally downloads FSD50K (30GB)
4. Shows progress with speed and time remaining
5. Verifies download integrity
6. Extracts archives if needed

**Output:**
```
Dataset Status:
┌─────────────────────┬──────┬──────────┬────────────────┐
│ Dataset             │ Size │ Priority │ Status         │
├─────────────────────┼──────┼──────────┼────────────────┤
│ ACAV100M Training   │ 40GB │ Critical │ Not downloaded │
│ ACAV100M Validation │ 1GB  │ Critical │ Not downloaded │
│ MIT RIR             │ 2GB  │ Required │ Not downloaded │
│ FSD50K              │ 30GB │ Optional │ Not downloaded │
└─────────────────────┴──────┴──────────┴────────────────┘

Downloading datasets...

⠋ ACAV100M Training [━━━━━━━━━━        ] 45% 15.2MB/s 0:08:30
```

**Options:**
- `--required-only, -r`: Download only required datasets (default)
- `--all, -a`: Download all datasets including optional
- `--status, -s`: Show download status without downloading
- `--workspace, -w`: Custom workspace path
- `--verbose, -v`: Enable verbose logging

### download-voices - Download TTS Voices

Download Piper TTS voice models for speech synthesis.

**Usage:**
```bash
# Download 2 English voices (default)
easy-oww download-voices

# Download specific number of voices
easy-oww download-voices --count 3

# Download voices for specific language
easy-oww download-voices --language de_DE --count 2

# List available languages
easy-oww download-voices --list-languages
```

**What It Does:**
1. Shows available languages
2. Downloads recommended voice models
3. Verifies voice model integrity
4. Shows download progress

**Output:**
```
Downloading TTS voices...

Available languages:
  • en_US (English - US)
  • en_GB (English - UK)
  • de_DE (German)
  • fr_FR (French)
  • es_ES (Spanish)

Downloading 2 English voices...

⠋ en_US-lessac-medium [━━━━━━━━━━━━━━] 100% Complete
⠋ en_US-amy-medium    [━━━━━━━━━━━━━━] 100% Complete

✓ Downloaded 2 voices successfully
```

**Options:**
- `--language, -l`: Language code (default: en_US)
- `--count, -c`: Number of voices to download (default: 2)
- `--list-languages`: Show available languages
- `--workspace, -w`: Custom workspace path
- `--verbose, -v`: Enable verbose logging

### create - Create Project

Create a new wake word project.

**Usage:**
```bash
# Create project (interactive prompts)
easy-oww create my_wake_word

# Create with wake word specified
easy-oww create my_wake_word --wake-word "hey assistant"

# Create with custom parameters
easy-oww create my_wake_word \
  --wake-word "hey assistant" \
  --samples 2000 \
  --steps 15000

# Create on external drive
easy-oww create my_wake_word --workspace /Volumes/MyDrive/easy-oww
```

**What It Does:**
1. Prompts for wake word if not provided
2. Detects available TTS voices
3. Creates project directory structure
4. Generates training configuration
5. Saves config.yaml

**Output:**
```
Creating project: my_wake_word

Wake word: "hey assistant"
Target samples: 1000
Training steps: 10000
Voices detected: 2 (en_US-lessac-medium, en_US-amy-medium)

✓ Project created successfully!

Project location: ~/.easy-oww/projects/my_wake_word

Next steps:
  1. Record samples: easy-oww record my_wake_word --count 20
  2. Train model: easy-oww train my_wake_word
```

**Options:**
- `--wake-word`: Wake word phrase
- `--samples`: Target number of training samples (default: 1000)
- `--steps`: Training steps (default: 10000)
- `--workspace, -w`: Custom workspace path
- `--verbose, -v`: Enable verbose logging

### record - Record Samples

Record wake word samples with your microphone.

**Usage:**
```bash
# Record 20 samples (default)
easy-oww record my_wake_word

# Record specific number
easy-oww record my_wake_word --count 50

# Record additional samples (adds to existing)
easy-oww record my_wake_word --count 10
```

**What It Does:**
1. Lists available microphones
2. Lets you select microphone
3. Tests microphone quality
4. Guides you through recording each sample
5. Validates audio quality in real-time
6. Allows re-recording of poor samples
7. Shows summary when complete

**Output:**
```
Recording Session

Available microphones:
  1. Built-in Microphone
  2. USB Microphone

Select microphone [1]: 1

Testing microphone...
✓ Microphone working properly

Recording 20 samples of "hey assistant"

Sample 1/20
Press ENTER when ready...
[Recording...] ●
✓ Sample 1 recorded (1.8s, good quality)

Sample 2/20
Press ENTER when ready...
[Recording...] ●
⚠ Volume too low - please speak louder
Press ENTER to retry...
[Recording...] ●
✓ Sample 2 recorded (1.9s, good quality)

...

✓ Recording complete!

Summary:
  • Total samples: 20
  • Average duration: 1.9s
  • Average SNR: 28dB
  • Quality: Good

Recordings saved to: ~/.easy-oww/projects/my_wake_word/recordings/

Next step:
  Train your model: easy-oww train my_wake_word
```

**Options:**
- `--count, -c`: Number of samples to record (default: 20)
- `--workspace, -w`: Custom workspace path
- `--verbose, -v`: Enable verbose logging

### train - Train Model

Train wake word model using recorded and synthetic samples.

**Usage:**
```bash
# Start training
easy-oww train my_wake_word

# Resume training from checkpoint
easy-oww train my_wake_word --resume

# Train with verbose output
easy-oww train my_wake_word --verbose
```

**What It Does:**
1. **Phase 1: Clip Generation** (15-20 min)
   - Processes user recordings
   - Generates synthetic samples with TTS
   - Creates negative (non-wake-word) samples
   - Validates all clips
2. **Phase 2: Augmentation** (10-15 min)
   - Applies room impulse responses
   - Adds background noise
   - Applies pitch/time variations
   - Creates training variations
3. **Phase 3: Training** (15-30 min)
   - Trains neural network
   - Validates on test set
   - Exports ONNX model
   - Saves training metrics

**Output:**
```
Training: my_wake_word
Wake word: "hey assistant"

Phase 1: Clip Generation
⠋ Processing recordings      [━━━━━━━━━━] 20/20 Complete
⠋ Generating synthetic clips [━━━━━━━━━━] 980/980 Complete
⠋ Generating negative clips  [━━━━━━━━━━] 1000/1000 Complete
✓ Clip generation complete (18 min)

Phase 2: Augmentation
⠋ Augmenting clips [━━━━━━━━━━━━━━] 2000/2000 Complete
✓ Augmentation complete (12 min)

Phase 3: Training
⠋ Training model [━━━━━━━━━━━━━━━] Step 10000/10000
  Loss: 0.0023 | Accuracy: 99.2% | Validation: 98.1%
✓ Training complete (25 min)

✓ Model saved: ~/.easy-oww/projects/my_wake_word/models/my_wake_word.onnx

Next step:
  Test your model: easy-oww test my_wake_word
```

**Options:**
- `--resume`: Resume from last checkpoint
- `--workspace, -w`: Custom workspace path
- `--verbose, -v`: Enable verbose logging

### test - Test Model

Test trained model with real-time detection or clip evaluation.

**Usage:**
```bash
# Interactive test menu
easy-oww test my_wake_word

# Test with custom duration
easy-oww test my_wake_word --duration 120

# Test on external drive project
easy-oww test my_wake_word --workspace /Volumes/MyDrive/easy-oww
```

**What It Does:**
Shows interactive menu:
1. **Real-time microphone test**
   - Opens microphone stream
   - Detects wake word in real-time
   - Shows detection count and timing
2. **Evaluate on test clips**
   - Runs model on test dataset
   - Calculates accuracy metrics
   - Shows confusion matrix
3. **Both**
   - Runs both tests sequentially

**Output (Real-time test):**
```
Real-Time Wake Word Detection Test

Model: my_wake_word.onnx
Duration: 60 seconds
Threshold: 0.5

Starting detection...
Press Ctrl+C to stop

┌── Detection Status ──────────────┐
│ Elapsed Time    │ 15.3s          │
│ Remaining Time  │ 44.7s          │
│ Detections      │ 3              │
│ Last Detection  │ 2.1s ago       │
└──────────────────────────────────┘

✓ Detection test complete
  • Total detections: 8
  • Average confidence: 0.87
  • False positives: 0
```

**Output (Clip evaluation):**
```
Evaluating on test clips...

⠋ Processing clips [━━━━━━━━━━━━] 200/200 Complete

┌── Detection Metrics ────────────┐
│ Total Samples  │ 200            │
│ Accuracy       │ 92.50%         │
│ Precision      │ 90.00%         │
│ Recall         │ 94.74%         │
│ F1 Score       │ 92.31%         │
│                                 │
│ True Positives  │ 90            │
│ True Negatives  │ 95            │
│ False Positives │ 10            │
│ False Negatives │ 5             │
└─────────────────────────────────┘

✓ Results saved to test_results.json
```

**Options:**
- `--duration, -d`: Test duration in seconds (default: 60)
- `--workspace, -w`: Custom workspace path
- `--verbose, -v`: Enable verbose logging

### list - List Projects and Voices

List all projects and installed voices.

**Usage:**
```bash
# List all projects and voices
easy-oww list

# List on external drive
easy-oww list --workspace /Volumes/MyDrive/easy-oww
```

**Output:**
```
Projects:
┌──────────────────┬─────────────┬────────────┬─────────┐
│ Project          │ Wake Word   │ Samples    │ Trained │
├──────────────────┼─────────────┼────────────┼─────────┤
│ my_wake_word     │ hey assist  │ 20         │ ✓       │
│ lights_on        │ lights on   │ 30         │ ✓       │
│ alexa_custom     │ alexa       │ 15         │ -       │
└──────────────────┴─────────────┴────────────┴─────────┘

Installed Voices:
  • en_US-lessac-medium (63 MB)
  • en_US-amy-medium (63 MB)
  • de_DE-thorsten-medium (63 MB)

Total: 3 voices, 189 MB
```

**Options:**
- `--workspace, -w`: Custom workspace path
- `--verbose, -v`: Enable verbose logging

## Global Options

Available for all commands:

- `--verbose, -v`: Enable verbose logging (shows DEBUG level messages)
- `--version`: Show version and exit
- `--help`: Show help message and exit

## External Storage Support

All commands support the `--workspace` flag or `EASY_OWW_WORKSPACE` environment variable:

**Using --workspace flag:**
```bash
easy-oww init --workspace /Volumes/MyDrive/easy-oww
easy-oww download --required-only --workspace /Volumes/MyDrive/easy-oww
easy-oww create my_project --workspace /Volumes/MyDrive/easy-oww
```

**Using environment variable:**
```bash
# Set once in your shell profile
export EASY_OWW_WORKSPACE="/Volumes/MyDrive/easy-oww"

# Then use commands without --workspace flag
easy-oww init
easy-oww download --required-only
easy-oww create my_project
```

## Complete Workflow Example

```bash
# 1. Initialize workspace (one-time setup)
easy-oww init --workspace /Volumes/MyDrive/easy-oww

# Set environment variable (add to ~/.bashrc or ~/.zshrc)
export EASY_OWW_WORKSPACE="/Volumes/MyDrive/easy-oww"

# 2. Download datasets (one-time, ~43GB)
easy-oww download --required-only

# 3. Download TTS voices (one-time, ~200MB)
easy-oww download-voices --language en_US --count 2

# 4. Create project
easy-oww create my_wake_word --wake-word "hey assistant"

# 5. Record samples
easy-oww record my_wake_word --count 20

# 6. Train model
easy-oww train my_wake_word

# 7. Test model
easy-oww test my_wake_word

# 8. Create more projects (repeat steps 4-7)
easy-oww create lights_on --wake-word "turn on lights"
easy-oww record lights_on --count 30
easy-oww train lights_on
easy-oww test lights_on
```

## Error Handling

The CLI provides helpful error messages:

```bash
# Project doesn't exist
$ easy-oww record nonexistent
Error: Project 'nonexistent' does not exist
Create it with: easy-oww create nonexistent

# Not enough recordings
$ easy-oww train my_project
Error: Need at least 10 recordings, found 3
Record more with: easy-oww record my_project --count 10

# Out of disk space
$ easy-oww download --required-only
Error: Need 43GB free space, only 15GB available
Consider using external drive: easy-oww init --workspace /Volumes/MyDrive
```

## Advanced Usage

### Scripting and Automation

The CLI can be used in scripts:

```bash
#!/bin/bash
# Create multiple wake words

WAKE_WORDS=("hey assistant" "ok computer" "alexa custom")

for word in "${WAKE_WORDS[@]}"; do
    # Create project
    project_name=$(echo "$word" | tr ' ' '_')
    easy-oww create "$project_name" --wake-word "$word"

    # Record samples (interactive)
    echo "Record samples for: $word"
    easy-oww record "$project_name" --count 20

    # Train
    easy-oww train "$project_name"

    echo "✓ Completed: $word"
done
```

### Batch Processing

Process multiple projects:

```bash
# Train all projects
for project in ~/.easy-oww/projects/*/; do
    project_name=$(basename "$project")
    easy-oww train "$project_name"
done

# Test all trained models
for project in ~/.easy-oww/projects/*/; do
    project_name=$(basename "$project")
    if [ -f "$project/models/$project_name.onnx" ]; then
        easy-oww test "$project_name"
    fi
done
```

### Custom Configuration

Override configuration for specific runs:

```bash
# Create project with custom training config
easy-oww create high_accuracy \
  --wake-word "hey assistant" \
  --samples 5000 \
  --steps 50000

# This will train a more accurate but slower model
```

## Troubleshooting

### Command Not Found

```bash
# If easy-oww command not found:
# Option 1: Reinstall
pip install -e /path/to/easy-oww

# Option 2: Run directly
python -m easy_oww.cli.main --help

# Option 3: Check PATH
which easy-oww
```

### Verbose Output

Use `--verbose` to see detailed output:

```bash
easy-oww train my_project --verbose

# Shows:
# - Debug log messages
# - File operations
# - Detailed progress
# - Stack traces on errors
```

### Reset Workspace

To start fresh:

```bash
# Remove workspace (WARNING: deletes all projects and models!)
rm -rf ~/.easy-oww

# Reinitialize
easy-oww init
easy-oww download --required-only
```

## Testing

Test the CLI module:

```bash
pytest tests/test_cli.py -v
```

## Future Enhancements

Planned CLI features:
- `easy-oww status` - Show overall status dashboard
- `easy-oww clean` - Clean up intermediate files
- `easy-oww export` - Export model with metadata
- `easy-oww import` - Import pre-trained models
- `easy-oww benchmark` - Benchmark model performance
- `easy-oww compare` - Compare multiple models
- Shell completion (bash, zsh, fish)
- Configuration file support (.easy-oww.yaml)
- Cloud integration (upload/download models)
