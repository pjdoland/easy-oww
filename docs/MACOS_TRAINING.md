# macOS Training Guide

## Known Issue: Bus Errors During Training

### Problem

When training models on macOS, you may encounter bus errors during the feature extraction phase:

```
zsh: bus error  easy-oww train <project_name>
```

This is accompanied by warnings about leaked semaphore objects:

```
resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

### Root Cause

The issue is caused by **OpenWakeWord's use of ThreadPool for parallel processing** combined with **memory-mapped numpy arrays**. macOS has stricter memory alignment requirements than Linux, and when multiple threads access shared memory-mapped files concurrently, it triggers memory alignment violations that result in bus errors.

Specifically:
- OpenWakeWord uses `multiprocessing.pool.ThreadPool` to parallelize audio feature extraction
- Features are written to disk using `numpy.memmap` (memory-mapped arrays)
- Multiple threads writing to the same mmap file causes memory corruption on macOS
- This manifests as a `SIGBUS` (bus error) signal

### Solution

Easy-OWW automatically applies compatibility patches that:

1. **Force single-threaded processing (ncpu=1) on macOS** - Disables ThreadPool entirely
2. **Add explicit memory cleanup** - Uses `del` and `gc.collect()` to free memory between batches
3. **Reduce batch size** - Caps batch size at 64 on macOS (vs 128 on other platforms)

These patches are applied automatically via `easy_oww/training/oww_patches.py` when you run training.

### Performance Impact

Single-threaded processing is **slower but stable**:

- **With multiprocessing (Linux)**: 1-2 hours for typical dataset
- **Without multiprocessing (macOS)**: 3-5 hours for typical dataset

The slower speed is unavoidable on macOS CPU, but you can speed up training by:

1. **Using GPU acceleration** - Install CUDA-enabled PyTorch (if you have an NVIDIA eGPU)
2. **Reducing dataset size** - Generate fewer adversarial samples
3. **Training on Linux** - Use Docker, VM, or cloud instance

### Verification

When training starts, you should see:

```
Note: Using single CPU on macOS to prevent memory errors
Reduced batch size from 128 to 64 for macOS stability
```

This confirms the patches are active.

### Upstream Status

This is a known limitation of OpenWakeWord on macOS. Consider:

- **Contributing a PR to OpenWakeWord** to add official macOS support
- **Using Linux for training** if you need faster training times

### Technical Details

The patches are implemented in `easy_oww/training/oww_patches.py` and modify:

1. **`compute_features_from_generator`** - Forces ncpu=1, adds memory cleanup
2. **`AudioFeatures._get_melspectrogram_batch`** - Disables ThreadPool on macOS, adds single-threaded fallback
3. **`AudioFeatures._get_embeddings_batch`** - Disables ThreadPool on macOS, adds single-threaded fallback

These patches are non-invasive monkey patches that only affect the OpenWakeWord library at runtime, without modifying installed files.

## Alternative: Docker on macOS

If you need faster training, use Docker with Linux:

```bash
# Build Docker image with GPU support
docker build -t easy-oww .

# Run training in Docker (Linux environment)
docker run --gpus all -v $(pwd):/workspace easy-oww train <project_name>
```

This runs training in a Linux environment where multiprocessing works correctly.
