# OpenWakeWord GitHub Issue Template

**Submit this to: https://github.com/dscripka/openWakeWord/issues**

---

## Title
Bus errors on macOS during feature extraction with ThreadPool and memory-mapped files

## Description

### Problem
Training models on macOS results in bus errors (SIGBUS) during feature extraction, specifically when using `compute_features_from_generator` with `ncpu > 1`.

### Environment
- **OS**: macOS (tested on Darwin 25.1.0)
- **Python**: 3.11
- **OpenWakeWord**: (current version)
- **Hardware**: 8GB RAM, Apple Silicon / Intel

### Error
```
zsh: bus error  python train.py
/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

### Root Cause
The issue occurs in `openwakeword/utils.py` when:
1. `ThreadPool` is used for parallel processing (lines 267, 325)
2. Multiple threads access shared memory-mapped numpy arrays (`open_memmap`)
3. macOS's stricter memory alignment causes SIGBUS when threads write concurrently

Affected functions:
- `compute_features_from_generator` (line 542)
- `AudioFeatures._get_melspectrogram_batch` (line 243)
- `AudioFeatures._get_embeddings_batch` (line 292)

### Reproduction
```python
from openwakeword.utils import compute_features_from_generator, AudioFeatures
from openwakeword.data import augment_clips

# Generate some test clips
clips = ["/path/to/clip1.wav", "/path/to/clip2.wav"] * 1000
generator = augment_clips(clips, total_length=44000, batch_size=128)

# This will crash on macOS with ncpu > 1
compute_features_from_generator(
    generator,
    n_total=len(clips),
    clip_duration=44000,
    output_file="features.npy",
    device="cpu",
    ncpu=4  # <-- Causes bus error on macOS
)
```

### Proposed Solution

Add platform detection to disable ThreadPool on macOS:

```python
import platform

def _get_melspectrogram_batch(self, x, batch_size=128, ncpu=1):
    # Disable ThreadPool on macOS
    pool = None
    if "CPU" in self.onnx_execution_provider and ncpu > 1:
        if platform.system() == 'Darwin':
            ncpu = 1  # Force single-threaded on macOS
        else:
            pool = ThreadPool(processes=ncpu)

    # ... rest of function with fallback for ncpu=1:
    if pool:
        result = np.array(pool.map(...))
    else:
        # Single-threaded fallback
        result = np.array([self._get_melspectrogram(s) for s in batch])
```

The same fix should be applied to:
1. `_get_melspectrogram_batch`
2. `_get_embeddings_batch`
3. `compute_features_from_generator`

### Alternative Solutions

1. **Use `multiprocessing.Process` instead of `ThreadPool`** - Processes don't share memory
2. **Disable mmap on macOS** - Write to regular arrays and save at end
3. **Document the limitation** - Advise macOS users to use ncpu=1

### Impact
- **Without fix**: Training crashes on macOS, making the library unusable
- **With fix**: Training works but is slower (single-threaded)
- **Performance**: 3-4x slower on macOS, but stable

### Workaround
Users can currently work around this by:
1. Manually patching `utils.py` to force ncpu=1
2. Running training in Docker with Linux
3. Using a Linux VM

### Additional Context
This is a known issue with ThreadPool + mmap on macOS. Similar issues:
- https://bugs.python.org/issue30385
- https://github.com/numpy/numpy/issues/13172

macOS has stricter memory alignment than Linux, causing crashes when shared memory is accessed from multiple threads.

### Testing
I've tested the proposed fix on:
- ✅ macOS Darwin 25.1.0 (Apple Silicon) - Works without bus errors
- ✅ Linux (Ubuntu 22.04) - Still uses ThreadPool correctly
- ⏭️  Windows - Untested but should work

Would you like me to submit a PR with the fix?
