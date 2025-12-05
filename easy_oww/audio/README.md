# Audio Recording System

This module provides a complete audio recording system for capturing wake word samples with quality validation.

## Components

### AudioRecorder (`recorder.py`)

Handles audio recording from microphone input.

**Features:**
- List available audio input devices
- Record fixed duration samples
- Record until manually stopped
- Save recordings as WAV files (16kHz, mono, int16)
- Calculate audio levels (RMS in dB)
- Test microphone functionality

**Example:**
```python
from easy_oww.audio import AudioRecorder

recorder = AudioRecorder(sample_rate=16000)

# List devices
devices = recorder.list_devices()
print(f"Found {len(devices)} input devices")

# Record 2 seconds
audio = recorder.record_duration(duration=2.0)

# Save to file
recorder.save_wav(audio, Path("sample.wav"))
```

### AudioValidator (`validator.py`)

Validates audio quality for training suitability.

**Validation Checks:**
- Duration (0.5s - 3.0s)
- Audio level (-50 dB to -10 dB)
- Clipping detection
- Silence percentage
- DC offset detection
- Signal-to-noise ratio estimation

**Example:**
```python
from easy_oww.audio import AudioValidator

validator = AudioValidator()

# Validate audio array
result = validator.validate_audio(audio)
if result['valid']:
    print("Audio quality is good!")
else:
    print("Issues:", result['issues'])

# Validate file
result = validator.validate_file(Path("sample.wav"))

# Batch validate
results = validator.batch_validate(file_list)
print(f"Valid: {results['valid']}/{results['total']}")
```

### RecordingUI (`ui.py`)

Provides an interactive terminal UI for recording sessions.

**Features:**
- Microphone selection menu
- Microphone testing
- Guided recording with countdown
- Real-time quality feedback
- Retry/skip options for failed samples
- Session summary with quality report

**Example:**
```python
from easy_oww.audio import run_recording_session
from pathlib import Path

# Run complete recording session
recorded_files = run_recording_session(
    output_dir=Path("./recordings"),
    count=20,
    duration=2.0,
    sample_rate=16000
)

print(f"Recorded {len(recorded_files)} samples")
```

## CLI Integration

The audio system is integrated into the CLI via the `record` command:

```bash
# Record 20 samples (default)
easy-oww record my_wake_word

# Record custom number of samples
easy-oww record my_wake_word --count 50
```

## Audio Quality Guidelines

For best results when recording wake word samples:

**Duration:**
- Keep samples between 0.5 and 3.0 seconds
- Aim for 1-2 seconds for most wake words

**Audio Level:**
- Speak at normal conversation volume
- Target level: -40 to -20 dB
- Avoid very quiet (<-50 dB) or very loud (>-10 dB) recordings

**Recording Environment:**
- Use a quiet room to minimize background noise
- Keep consistent distance from microphone (6-12 inches)
- Avoid rooms with excessive echo or reverb

**Variety:**
- Vary your tone and speed across samples
- Try different distances from microphone
- Include natural pronunciation variations
- Stay consistent with pronunciation

**Technical Requirements:**
- Sample rate: 16000 Hz (required for OpenWakeWord)
- Channels: 1 (mono)
- Bit depth: 16-bit PCM

## Error Handling

The system provides clear error messages for common issues:

- **No microphone detected**: Check device connections
- **Audio too quiet**: Increase microphone volume or speak louder
- **Audio too loud/clipping**: Decrease microphone volume or move back
- **Too much silence**: Speak the wake word clearly and promptly
- **Low SNR**: Reduce background noise or improve microphone quality

## Testing

Run the audio tests:

```bash
pytest tests/test_audio.py -v
```

Test microphone functionality:

```python
from easy_oww.audio import AudioRecorder

recorder = AudioRecorder()
success, message, audio = recorder.test_microphone()
print(message)
```
