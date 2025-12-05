# Frequently Asked Questions (FAQ)

Common questions and answers about easy-oww.

## Table of Contents

- [General Questions](#general-questions)
- [Storage and Performance](#storage-and-performance)
- [Recording Questions](#recording-questions)
- [Training Questions](#training-questions)
- [Testing Questions](#testing-questions)
- [Technical Questions](#technical-questions)
- [Troubleshooting](#troubleshooting)

## General Questions

### What is easy-oww?

easy-oww is a command-line tool that simplifies creating custom wake word models for OpenWakeWord. It automates the entire process from recording audio samples to training and testing ONNX models.

### Do I need programming experience?

No! easy-oww is designed to be user-friendly. If you can use a terminal and follow instructions, you can create wake word models.

### How long does it take to create a model?

**Quick version (testing):** ~30-45 minutes
- 10 min: Downloads (if already cached)
- 5 min: Recording
- 15-20 min: Training (500 samples)

**Production version:** ~2-3 hours
- 30-40 min: Initial downloads (one-time)
- 10 min: Recording
- 1.5-2 hours: Training (2000 samples)

### What can I use the trained models for?

- **Voice assistants**: Wake word detection for custom assistants
- **Home automation**: Voice-controlled smart home devices
- **Embedded devices**: Raspberry Pi, ESP32, etc.
- **Research**: Wake word detection experiments
- **Products**: Commercial products (check OpenWakeWord license)

### Is this free?

Yes! easy-oww is open source (MIT License). You can use it for personal or commercial projects.

### Do I need an internet connection?

**During setup:** Yes
- Downloading datasets (40-70 GB)
- Downloading TTS voices (~200 MB)
- Installing Piper TTS (~50 MB)

**After setup:** No
- Training runs completely offline
- Recording is offline
- Testing is offline

## Storage and Performance

### How much disk space do I need?

**Minimum (required datasets only):**
- ACAV100M features: 40 GB
- MIT RIR dataset: 2 GB
- TTS voices: 200 MB
- Workspace: 1-2 GB
- **Total: ~43 GB**

**Recommended (all features):**
- Add FSD50K dataset: +30 GB
- Multiple projects: +5-10 GB
- **Total: ~75 GB**

### Can I use an external drive?

**Yes!** External drives work great, especially USB-C SSDs or USB 3.0+ drives.

**Recommended setup:**
```bash
# Initialize on external drive
easy-oww init --workspace /Volumes/MyDrive/easy-oww

# Or set environment variable
export EASY_OWW_WORKSPACE="/Volumes/MyDrive/easy-oww"
```

**Best external drive types:**
- ✅ **USB-C SSD** (500+ MB/s) - Best performance
- ✅ **USB 3.0 SSD** (400+ MB/s) - Great performance
- ⚠️  **USB 3.0 Flash** (100-150 MB/s) - Acceptable
- ❌ **USB 2.0 Flash** (10-30 MB/s) - Very slow, not recommended

See [INSTALLATION.md](INSTALLATION.md#external-storage-setup) for detailed setup.

### Will training be slow on an external drive?

**USB-C SSD:** Negligible difference (<10% slower)
**USB 3.0 SSD:** Slightly slower (10-20% slower)
**USB 3.0 Flash:** Moderately slower (50-100% slower)
**USB 2.0 Flash:** Very slow (3-4x slower) - not recommended

The main bottleneck is usually the CPU/GPU during training, not disk I/O, especially with SSDs.

### Can I move my workspace between computers?

Yes! Just copy the entire workspace directory to another computer:

```bash
# On Computer A (export)
cp -r ~/.easy-oww /Volumes/USB/backup/

# On Computer B (import)
cp -r /Volumes/USB/backup/.easy-oww ~/

# Or use the external drive directly on both computers
```

### Can I delete datasets after training?

**After training a project, you can safely delete:**
- ❌ **Don't delete:** ACAV100M features (needed for new projects)
- ❌ **Don't delete:** RIR dataset (needed for new projects)
- ✅ **Can delete:** FSD50K dataset (if not using negative samples)
- ✅ **Can delete:** Individual project clips (keep recordings & models)

To save space per project:
```bash
# Keep only essentials
cd ~/.easy-oww/projects/my_project
rm -rf clips/  # Delete processed clips (can regenerate)
rm -rf features/  # Delete intermediate features (can regenerate)
# Keep: recordings/, models/, config.yaml
```

### Do I need a GPU?

**No, but it helps:**
- **CPU only:** Works fine, training takes 1-2 hours for 2000 samples
- **With GPU:** Training takes 30-60 minutes for 2000 samples (50-70% faster)

GPU is recommended but not required. Most people can train on CPU.

## Recording Questions

### How many samples should I record?

**Minimum:** 10 samples (for quick testing)
**Recommended:** 20-30 samples
**Optimal:** 50+ samples

More real samples = better model, especially for your specific voice.

### What should I say when recording?

Say your wake word exactly as you want the model to detect it:
- "Hey Assistant"
- "Okay Computer"
- "Jarvis"
- Any phrase you want!

**Tips:**
- Be consistent with pronunciation
- Vary tone and speed slightly
- Include natural variations (louder, softer, faster, slower)

### What if my recording fails validation?

The system checks for:
- **Too quiet**: Speak louder or move closer to mic
- **Too loud**: Move back or reduce mic volume
- **Too short/long**: Speak at normal pace
- **Too much silence**: Start speaking promptly

You can retry any failed recording. The system will guide you.

### Do I need a professional microphone?

No! Any microphone works:
- ✅ Built-in laptop microphone
- ✅ Webcam microphone
- ✅ USB microphone
- ✅ Headset microphone
- ✅ Professional microphone

Built-in mics work surprisingly well! Just ensure it's in a quiet room.

### Can I use existing audio files?

Not directly through the CLI, but you can manually copy WAV files:

```bash
# Convert to 16kHz mono WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Copy to recordings directory
cp output.wav ~/.easy-oww/projects/my_project/recordings/
```

Make sure files are:
- 16kHz sample rate
- Mono (1 channel)
- 16-bit PCM
- WAV format

### Can I record in a different language?

Yes! easy-oww supports any language:

1. Download voices for your language:
   ```bash
   easy-oww download-voices --language de_DE  # German
   easy-oww download-voices --language fr_FR  # French
   easy-oww download-voices --language es_ES  # Spanish
   ```

2. Create project with your wake word:
   ```bash
   easy-oww create mein_projekt --wake-word "hey assistent"
   ```

3. Record in your language
4. Train normally

## Training Questions

### How does training work?

**3-Phase Process:**

1. **Clip Generation** (15-30 min)
   - Processes your recordings
   - Generates 500-2000 synthetic samples with TTS
   - Creates negative (non-wake-word) samples

2. **Augmentation** (10-20 min)
   - Applies room acoustics (reverb)
   - Adds background noise
   - Varies pitch and speed
   - Creates 2-3x more samples

3. **Model Training** (30-90 min)
   - Trains neural network
   - Exports to ONNX format
   - Optimizes for inference

### What are synthetic samples?

Synthetic samples are generated using TTS (text-to-speech) to say your wake word in different voices. This gives you hundreds of samples without manual recording.

**Why use TTS?**
- Gets you to 1000+ samples quickly
- Adds voice diversity
- Fills gaps in recording variations

### Can I train without TTS?

Technically yes, but not recommended:
- You'd need to manually record 1000+ samples
- This would take hours
- Results would be limited to your voice only

TTS dramatically speeds up the process and improves model robustness.

### How do I know if training is working?

**Good signs:**
- Progress bars moving steadily
- No error messages
- Clip counts match configuration
- Augmentation completes successfully

**After training:**
- ONNX model file exists in `models/` directory
- File size is reasonable (5-50 MB)
- Test command detects the model

### Can I stop and resume training?

**Yes!** Use the `--resume` flag:

```bash
# Start training
easy-oww train my_project

# If interrupted (Ctrl+C), resume with:
easy-oww train my_project --resume
```

Currently resumes from phase 1 (regenerates clips). Full checkpoint resuming is planned.

### How can I improve model accuracy?

**Better Recording Quality:**
- Record in quiet environment
- Use consistent microphone
- Record 50+ samples instead of 20
- Include edge cases (whispered, shouted, etc.)

**Better Training Configuration:**
- Use more TTS voices (3-5 instead of 2)
- Increase sample count (2000 instead of 1000)
- Train longer (15000 steps instead of 10000)
- Enable augmentation with higher probability

**Better Testing:**
- Test in realistic conditions
- Adjust threshold for your use case
- Test with different speakers if needed

### Can I train multiple models simultaneously?

Yes, but not recommended:
- Each training uses significant CPU/RAM
- Disk I/O can bottleneck
- Better to train sequentially

However, you can have multiple projects and train them one at a time.

## Testing Questions

### How do I test my model?

**Real-time microphone test:**
```bash
easy-oww test my_project
> Select: 1. Real-time microphone test
```

Speak your wake word, and the system shows detections.

**Accuracy evaluation:**
```bash
easy-oww test my_project
> Select: 2. Evaluate on test clips
```

Tests model on your training clips and shows metrics.

### What is a good accuracy score?

**Typical ranges:**
- **Accuracy**: 85-95% (higher is better)
- **Precision**: 80-92% (fewer false alarms)
- **Recall**: 90-98% (fewer misses)
- **F1 Score**: 85-95% (balanced metric)

**90%+ accuracy** is good for most use cases.

### Why does my model not detect my wake word?

**Common causes:**

1. **Threshold too high**
   - Solution: Lower to 0.3-0.4
   - Test with different thresholds

2. **Different pronunciation**
   - Solution: Record more samples with your pronunciation
   - Ensure consistency between training and testing

3. **Different environment**
   - Training: quiet room
   - Testing: noisy room
   - Solution: Train with more augmentation

4. **Microphone issues**
   - Different microphone than training
   - Solution: Test microphone with `easy-oww record` first

### Why does my model give false positives?

False positives = detecting wake word when you didn't say it.

**Solutions:**

1. **Raise threshold** (0.6-0.7)
2. **Add more negative samples** (edit config, retrain)
3. **Train longer** (more steps)
4. **Add problematic sounds to training**

### What threshold should I use?

Depends on your use case:

**Home automation** (0.6-0.7):
- Fewer false alarms (don't want accidental triggers)
- Ok to miss occasionally (just say it again)

**Voice assistant** (0.4-0.5):
- Responsive to user
- Some false alarms acceptable

**Security application** (0.7-0.8):
- Must not false alarm
- Ok to be less sensitive

Test multiple thresholds and choose what works for your needs.

### Can I test with someone else's voice?

Yes! Have them speak your wake word during the test. If it doesn't detect:
- Their voice is different from your training data
- Solution: Have them record 10-20 samples, retrain

For multi-speaker models:
- Record samples from multiple speakers (5-10 each)
- Use more TTS voices
- Train with higher sample count

## Technical Questions

### What format are the trained models?

**ONNX format** (.onnx files)
- Industry standard for neural networks
- Cross-platform (Windows, macOS, Linux, embedded)
- Works with OpenWakeWord runtime
- Can be used with ONNX Runtime in any language

### Can I use the models in my own application?

Yes! The models are standard ONNX files:

**Python:**
```python
import onnxruntime as ort

session = ort.InferenceSession("my_wake_word.onnx")
# Use session.run() to detect
```

**C++/Java/JavaScript:**
- Use ONNX Runtime for your language
- See OpenWakeWord documentation for integration

### What's the difference between easy-oww and OpenWakeWord?

**OpenWakeWord:**
- Framework for wake word detection
- Provides runtime for running models
- Requires manual setup of training pipeline
- Lower-level, more flexibility

**easy-oww:**
- Built on top of OpenWakeWord
- Automates the entire training workflow
- User-friendly CLI interface
- Handles all the complex setup

Think of it as: easy-oww creates models, OpenWakeWord runs them.

### Does easy-oww collect any data?

**No!** Everything runs locally:
- No analytics
- No telemetry
- No data sent to servers
- No account required

Only network usage:
- Downloading datasets (public repositories)
- Downloading TTS voices (HuggingFace)

### Can I contribute to the project?

Yes! Contributions welcome:
- Report issues on GitHub
- Submit pull requests
- Improve documentation
- Share your trained models (optional)

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### What Python version should I use?

**Recommended:** Python 3.9 or 3.10
- Best compatibility
- Well-tested
- All dependencies available

**Supported:** Python 3.7+
- 3.7: Minimum, some deps may be older versions
- 3.11+: Works but some deps may not have wheels yet

### Can I run this on Raspberry Pi?

**Yes, but with caveats:**

**Raspberry Pi 4 (4GB+ RAM):**
- ✅ Can run easy-oww
- ⚠️  Training will be very slow (4-6 hours)
- ✅ Recording works fine
- Consider training on PC, deploy model to Pi

**Raspberry Pi 3:**
- ⚠️  Possible but very slow
- May run out of RAM during training
- Recommended: Use external swap file

**Best approach:**
1. Train model on your PC/Mac
2. Copy .onnx model to Raspberry Pi
3. Use OpenWakeWord on Pi for detection

### Can I use this on Windows?

Yes! Fully supported:
- Windows 10/11
- PowerShell or Command Prompt
- All features work

Minor differences:
- Path format (use `\` or `\\`)
- Some packages may require Visual C++ Build Tools

See [INSTALLATION.md](INSTALLATION.md) for Windows-specific setup.

## Troubleshooting

### Downloads fail or are very slow

**Solutions:**

1. **Check internet connection**
   - Large downloads (40GB+)
   - Stable connection needed
   - Consider overnight download

2. **Use required-only flag**
   ```bash
   easy-oww download --required-only
   ```
   Skips optional FSD50K dataset (30GB)

3. **Resume interrupted downloads**
   - Just run the command again
   - Downloads auto-resume

4. **Try different mirror** (if available)
   - Some datasets have multiple sources

### Out of memory during training

**Solutions:**

1. **Reduce batch size**
   ```yaml
   # Edit config.yaml
   batch_size: 256  # Instead of 512
   ```

2. **Reduce sample count**
   ```yaml
   target_samples: 500  # Instead of 1000
   ```

3. **Close other applications**

4. **Use swap file** (Linux/Mac)
   ```bash
   # Create 8GB swap
   sudo fallocate -l 8G /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Model file not found after training

**Check:**

1. **Did training complete successfully?**
   - Look for "✓ Training complete!" message
   - Check for errors in output

2. **Check models directory**
   ```bash
   ls ~/.easy-oww/projects/my_project/models/
   ```

3. **Check project path**
   ```bash
   easy-oww list  # Shows all projects
   ```

4. **Check workspace path**
   - Did you use custom --workspace flag?
   - Use same flag for test command

### Permission errors

**On macOS/Linux:**
```bash
# Fix ownership
sudo chown -R $USER:$USER ~/.easy-oww

# Fix permissions
chmod -R u+rw ~/.easy-oww
```

**On Windows:**
- Right-click folder > Properties > Security
- Ensure your user has Full Control

### External drive disconnected during operation

**Prevention:**
- Don't unplug during downloads/training
- Use powered USB hub
- Disable auto-sleep for external drives

**Recovery:**
1. Reconnect drive
2. Resume operation with same command
3. Downloads will auto-resume
4. Training may need to restart (uses --resume)

### Microphone not detected

**Solutions:**

1. **Check connections**
   - Ensure mic is plugged in
   - Try different USB port
   - Check system settings

2. **Test system recognition**
   ```bash
   # macOS
   system_profiler SPAudioDataType

   # Linux
   arecord -l

   # Windows
   Get-PnpDevice -Class AudioEndpoint
   ```

3. **Grant permissions**
   - macOS: System Preferences > Security > Privacy > Microphone
   - Windows: Settings > Privacy > Microphone

4. **Try different microphone**

### Model performs poorly

**Diagnosis checklist:**

1. **Recording quality**
   - [ ] Quiet environment?
   - [ ] Consistent pronunciation?
   - [ ] 20+ samples recorded?

2. **Training configuration**
   - [ ] Used 2+ TTS voices?
   - [ ] Trained 1000+ samples?
   - [ ] Augmentation enabled?

3. **Testing conditions**
   - [ ] Same environment as training?
   - [ ] Same microphone?
   - [ ] Correct threshold?

**Improvement steps:**
1. Record 20 more samples in problem conditions
2. Increase target_samples to 2000
3. Use 3-4 TTS voices
4. Train longer (15000 steps)
5. Adjust threshold during testing

## Still Have Questions?

**Documentation:**
- [README.md](README.md) - Main guide
- [INSTALLATION.md](INSTALLATION.md) - Setup guide
- Module READMEs - Detailed component docs

**Support:**
- GitHub Issues: Report bugs
- GitHub Discussions: Ask questions
- Community: Share experiences

**Contributing:**
- Improve documentation
- Submit examples
- Report issues
- Share trained models

---

*Can't find your question? Open an issue on GitHub!*
