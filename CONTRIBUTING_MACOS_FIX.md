# Contributing the macOS Training Fix

This document outlines the macOS bus error fix and how to contribute it upstream.

## What Was Fixed

### The Problem
Training on macOS caused bus errors (SIGBUS) due to OpenWakeWord's use of ThreadPool with memory-mapped numpy arrays. macOS has stricter memory alignment requirements than Linux, causing crashes when multiple threads access shared memory concurrently.

### The Solution
We've implemented a comprehensive fix in `easy-oww` that:

1. **Detects macOS** at runtime
2. **Disables ThreadPool** on macOS (forces ncpu=1)
3. **Adds single-threaded fallbacks** when multiprocessing is disabled
4. **Improves memory management** with explicit cleanup and GC
5. **Reduces batch size** on macOS to lower memory pressure

## Files Added/Modified

### New Files
1. **`easy_oww/training/oww_patches.py`** - Monkey patches for OpenWakeWord
2. **`docs/MACOS_TRAINING.md`** - User documentation for macOS training
3. **`docs/OPENWAKEWORD_ISSUE.md`** - Template for upstream GitHub issue

### Modified Files
1. **`easy_oww/training/full_trainer.py`** - Calls `apply_patches()` before using OpenWakeWord
2. **`README.md`** - Added Platform Notes section

### Venv Files (Temporary)
These files in the venv were manually patched for testing but will be replaced by the monkey patch:
- `/venv/lib/python3.11/site-packages/openwakeword/utils.py`

## Testing Checklist

Before committing, test the following:

- [ ] **macOS Training Works** - Run `easy-oww train <project>` on macOS without bus errors
- [ ] **Patches Apply Correctly** - Check logs for "OpenWakeWord macOS compatibility patches applied"
- [ ] **Single-threaded Processing** - Verify output shows "Using single CPU on macOS"
- [ ] **Feature Extraction Completes** - All 4 phases complete without errors
- [ ] **Model Trains Successfully** - Full training pipeline completes
- [ ] **Linux Still Works** - Test on Linux to ensure patches don't break multiprocessing

## Commit Message Template

```
Fix: Add macOS compatibility patches for OpenWakeWord training

Fixes bus errors on macOS caused by ThreadPool + memory-mapped files

Changes:
- Add oww_patches.py with runtime monkey patches for OpenWakeWord
- Force single-threaded processing (ncpu=1) on macOS
- Add single-threaded fallbacks in batch processing methods
- Reduce batch size to 64 on macOS (from 128)
- Add explicit memory cleanup with gc.collect()
- Document macOS training limitations and workarounds

Technical details:
- OpenWakeWord uses ThreadPool for parallel feature extraction
- Memory-mapped numpy arrays are accessed from multiple threads
- macOS has strict memory alignment, causing SIGBUS crashes
- Solution: Disable ThreadPool on macOS, use sequential processing

Performance impact:
- macOS training time: 3-5 hours (vs 1-2 hours on Linux)
- But stable and reliable on macOS

See docs/MACOS_TRAINING.md for details
```

## Steps to Contribute Upstream

### 1. Submit Issue to OpenWakeWord

Use the template in `docs/OPENWAKEWORD_ISSUE.md`:

```bash
# Open browser to OpenWakeWord issues page
open https://github.com/dscripka/openWakeWord/issues/new

# Copy content from docs/OPENWAKEWORD_ISSUE.md
cat docs/OPENWAKEWORD_ISSUE.md | pbcopy  # macOS
```

### 2. Create Pull Request to OpenWakeWord (Optional)

If you want to contribute the fix directly:

```bash
# Fork OpenWakeWord on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/openWakeWord.git
cd openWakeWord

# Create branch
git checkout -b fix/macos-threadpool-bus-error

# Apply patches to openwakeword/utils.py
# (Copy the changes from easy_oww/training/oww_patches.py)

# Commit and push
git add openwakeword/utils.py
git commit -m "Fix macOS bus errors in ThreadPool feature extraction"
git push origin fix/macos-threadpool-bus-error

# Create PR on GitHub
```

### 3. Commit to easy-oww

```bash
cd /Users/pjdoland/Repos/easy-oww

# Check status
git status

# Stage new files
git add easy_oww/training/oww_patches.py
git add docs/MACOS_TRAINING.md
git add docs/OPENWAKEWORD_ISSUE.md
git add CONTRIBUTING_MACOS_FIX.md

# Stage modified files
git add easy_oww/training/full_trainer.py
git add README.md

# Commit
git commit -m "Fix: Add macOS compatibility patches for OpenWakeWord training

Fixes bus errors on macOS caused by ThreadPool + memory-mapped files.
See docs/MACOS_TRAINING.md for details."

# Push
git push origin main
```

## Long-term Strategy

### Short-term (Current)
- ✅ Use monkey patches in easy-oww
- ✅ Document the issue
- ✅ Provide workarounds

### Medium-term
- ⏳ Submit GitHub issue to OpenWakeWord
- ⏳ Get feedback from OpenWakeWord maintainers
- ⏳ Submit PR if maintainers are receptive

### Long-term
- ⏳ Remove monkey patches once OpenWakeWord fixes the issue
- ⏳ Update easy-oww to require fixed version of OpenWakeWord
- ⏳ Add version check and warning for old OpenWakeWord versions

## Notes

### Why Monkey Patching?

We chose monkey patching instead of forking OpenWakeWord because:

1. **Non-invasive** - Doesn't require maintaining a fork
2. **Easy to remove** - When upstream fixes it, just delete the patch
3. **Transparent** - Users don't need to install a forked version
4. **Fast iteration** - Can update patches without releasing new OpenWakeWord

### Future Improvements

Consider these enhancements:

1. **Auto-detect available memory** - Adjust batch size dynamically
2. **Process pool alternative** - Use multiprocessing.Process instead of ThreadPool
3. **GPU support on macOS** - Test with Metal acceleration
4. **Progress estimation** - Better time estimates for macOS users

## Questions?

If you have questions about this fix, see:
- `docs/MACOS_TRAINING.md` - User documentation
- `easy_oww/training/oww_patches.py` - Implementation details
- `docs/OPENWAKEWORD_ISSUE.md` - Technical background

Or open an issue on the easy-oww repository.
