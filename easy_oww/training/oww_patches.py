"""
Monkey patches for OpenWakeWord to fix macOS compatibility issues.

This module patches OpenWakeWord's compute_features_from_generator and AudioFeatures
to prevent bus errors on macOS caused by ThreadPool accessing memory-mapped files.

Issue: macOS has strict memory alignment requirements that cause crashes when
ThreadPool workers access shared memory-mapped numpy arrays concurrently.

Solution: Force single-threaded processing (ncpu=1) on macOS.
"""

import platform
import logging

logger = logging.getLogger(__name__)

# Track if patches have been applied
_patches_applied = False


def apply_patches():
    """
    Apply all OpenWakeWord patches. Call this before using any OpenWakeWord functions.
    Safe to call multiple times (patches are only applied once).
    """
    global _patches_applied

    if _patches_applied:
        logger.debug("OpenWakeWord patches already applied, skipping")
        return

    try:
        _patch_compute_features_from_generator()
        _patch_audio_features_batch_methods()
        _patches_applied = True
        logger.info("OpenWakeWord macOS compatibility patches applied successfully")
    except Exception as e:
        logger.warning(f"Failed to apply OpenWakeWord patches: {e}")
        logger.warning("Training may encounter bus errors on macOS")


def _patch_compute_features_from_generator():
    """Patch compute_features_from_generator to force ncpu=1 and add memory cleanup"""
    try:
        from openwakeword import utils
        import gc

        # Store original function
        original_compute_features = utils.compute_features_from_generator

        def patched_compute_features_from_generator(generator, n_total, clip_duration,
                                                     output_file, device="cpu", ncpu=1):
            """Patched version that forces ncpu=1 on macOS and adds memory cleanup"""
            from openwakeword.data import trim_mmap
            from numpy.lib.format import open_memmap
            from tqdm import tqdm
            import numpy as np

            # CRITICAL FIX: Force ncpu=1 on macOS to prevent bus errors
            is_macos = platform.system() == 'Darwin'
            if is_macos and ncpu > 1:
                logger.info(f"Forcing ncpu=1 on macOS (was {ncpu}) to prevent bus errors")
                ncpu = 1

            # Create audio features object with ncpu=1 to avoid ThreadPool issues
            F = utils.AudioFeatures(device=device, ncpu=1)

            # Determine the output shape and create output file
            n_feature_cols = F.get_embedding_shape(clip_duration/16000)
            output_shape = (n_total, n_feature_cols[0], n_feature_cols[1])
            fp = open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_shape)

            # Get batch size by pulling one value from the generator and store features
            row_counter = 0
            audio_data = next(generator)
            batch_size = audio_data.shape[0]

            if batch_size > n_total:
                raise ValueError(f"The value of 'n_total' ({n_total}) is less than the batch size ({batch_size})."
                               " Please increase 'n_total' to be >= batch size.")

            features = F.embed_clips(audio_data, batch_size=batch_size, ncpu=1)
            fp[row_counter:row_counter+features.shape[0], :, :] = features
            row_counter += features.shape[0]
            fp.flush()
            del features  # Explicitly free memory
            gc.collect()  # Force garbage collection

            # Compute features and add data to output file
            for audio_data in tqdm(generator, total=n_total//batch_size, desc="Computing features"):
                if row_counter >= n_total:
                    break

                features = F.embed_clips(audio_data, batch_size=batch_size, ncpu=1)
                if row_counter + features.shape[0] > n_total:
                    features = features[0:n_total-row_counter]

                fp[row_counter:row_counter+features.shape[0], :, :] = features
                row_counter += features.shape[0]
                fp.flush()
                del features  # Explicitly free memory
                if row_counter % (batch_size * 10) == 0:  # GC every 10 batches
                    gc.collect()

            # Trim empty rows from the mmapped array
            trim_mmap(output_file)

        # Replace the function
        utils.compute_features_from_generator = patched_compute_features_from_generator
        logger.debug("Patched compute_features_from_generator")

    except ImportError:
        logger.warning("Could not import openwakeword.utils for patching")
    except Exception as e:
        logger.warning(f"Failed to patch compute_features_from_generator: {e}")


def _patch_audio_features_batch_methods():
    """Patch AudioFeatures batch methods to disable ThreadPool on macOS"""
    try:
        from openwakeword.utils import AudioFeatures
        import numpy as np
        from multiprocessing.pool import ThreadPool

        # Store original methods
        original_get_melspec_batch = AudioFeatures._get_melspectrogram_batch
        original_get_embeddings_batch = AudioFeatures._get_embeddings_batch

        def patched_get_melspectrogram_batch(self, x, batch_size=128, ncpu=1):
            """Patched version that disables ThreadPool on macOS"""
            is_macos = platform.system() == 'Darwin'

            # Prepare ThreadPool object, if needed for multithreading
            pool = None
            if "CPU" in self.onnx_execution_provider and ncpu > 1:
                if is_macos:
                    # Don't use ThreadPool on macOS - causes bus errors
                    ncpu = 1
                else:
                    pool = ThreadPool(processes=ncpu)

            # Make batches
            n_frames = int(np.ceil(x.shape[1]/160-3))
            mel_bins = 32  # fixed by melspectrogram model
            melspecs = np.empty((x.shape[0], n_frames, mel_bins), dtype=np.float32)
            for i in range(0, max(batch_size, x.shape[0]), batch_size):
                batch = x[i:i+batch_size]

                if "CUDA" in self.onnx_execution_provider:
                    result = self._get_melspectrogram(batch)
                elif pool:
                    chunksize = batch.shape[0]//ncpu if batch.shape[0] >= ncpu else 1
                    result = np.array(pool.map(self._get_melspectrogram,
                                               batch, chunksize=chunksize))
                else:
                    # Single-threaded processing (used on macOS or when ncpu=1)
                    result = np.array([self._get_melspectrogram(sample) for sample in batch])

                melspecs[i:i+batch_size, :, :] = result.squeeze()

            # Cleanup ThreadPool
            if pool:
                pool.close()

            return melspecs

        def patched_get_embeddings_batch(self, x, batch_size=128, ncpu=1):
            """Patched version that disables ThreadPool on macOS"""
            is_macos = platform.system() == 'Darwin'

            # Ensure input is the correct shape
            if x.shape[1] < 76:
                raise ValueError("Embedding model requires the input melspectrograms to have at least 76 frames")

            # Prepare ThreadPool object, if needed for multithreading
            pool = None
            if "CPU" in self.onnx_execution_provider and ncpu > 1:
                if is_macos:
                    # Don't use ThreadPool on macOS - causes bus errors
                    ncpu = 1
                else:
                    pool = ThreadPool(processes=ncpu)

            # Calculate array sizes and make batches
            n_frames = (x.shape[1] - 76)//8 + 1
            embedding_dim = 96  # fixed by embedding model
            embeddings = np.empty((x.shape[0], n_frames, embedding_dim), dtype=np.float32)

            batch = []
            ndcs = []
            for ndx, melspec in enumerate(x):
                window_size = 76
                for i in range(0, melspec.shape[0], 8):
                    window = melspec[i:i+window_size]
                    if window.shape[0] == window_size:  # ignore windows that are too short
                        batch.append(window)
                ndcs.append(ndx)

                if len(batch) >= batch_size or ndx+1 == x.shape[0]:
                    batch = np.array(batch).astype(np.float32)
                    if "CUDA" in self.onnx_execution_provider:
                        result = self.embedding_model_predict(batch)
                    elif pool:
                        chunksize = batch.shape[0]//ncpu if batch.shape[0] >= ncpu else 1
                        result = np.array(pool.map(self._get_embeddings_from_melspec,
                                          batch, chunksize=chunksize))
                    else:
                        # Single-threaded processing (used on macOS or when ncpu=1)
                        result = np.array([self._get_embeddings_from_melspec(sample) for sample in batch])

                    for j, ndx2 in zip(range(0, result.shape[0], n_frames), ndcs):
                        embeddings[ndx2, :, :] = result[j:j+n_frames]

                    batch = []
                    ndcs = []

            # Cleanup ThreadPool
            if pool:
                pool.close()

            return embeddings

        # Replace the methods
        AudioFeatures._get_melspectrogram_batch = patched_get_melspectrogram_batch
        AudioFeatures._get_embeddings_batch = patched_get_embeddings_batch
        logger.debug("Patched AudioFeatures batch methods")

    except ImportError:
        logger.warning("Could not import AudioFeatures for patching")
    except Exception as e:
        logger.warning(f"Failed to patch AudioFeatures: {e}")
