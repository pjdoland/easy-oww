# Model Testing and Validation

This module provides comprehensive testing and validation for trained wake word models, including real-time detection and accuracy metrics.

## Overview

The testing system enables:

1. **Real-Time Testing**: Test models with live microphone input
2. **Clip Evaluation**: Evaluate model accuracy on test datasets
3. **Metrics Calculation**: Comprehensive performance metrics
4. **Detection Visualization**: Live detection feedback

## Components

### ModelDetector (`detector.py`)

Real-time wake word detector with ONNX model support.

**Features:**
- Load and run ONNX models
- Real-time audio streaming
- Buffered detection with configurable threshold
- OpenWakeWord model support
- Detection event tracking

**Example:**
```python
from easy_oww.testing import ModelDetector
from pathlib import Path

# Initialize detector
detector = ModelDetector(
    model_path=Path("models/my_wake_word.onnx"),
    sample_rate=16000,
    detection_threshold=0.5
)

# Load model
detector.load_model()

# Start real-time detection
detections = detector.start_detection(
    duration=60,  # 60 seconds
    on_detection=lambda d: print(f"Detected! Score: {d['score']}")
)

print(f"Total detections: {len(detections)}")
```

### DetectionMetrics (`metrics.py`)

Comprehensive performance metrics for classification.

**Metrics Calculated:**
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity / True positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **False Positive Rate**: 1 - Specificity
- **False Negative Rate**: 1 - Recall

**Example:**
```python
from easy_oww.testing import DetectionMetrics

metrics = DetectionMetrics()

# Add test results
metrics.add_result(predicted=True, ground_truth=True)   # TP
metrics.add_result(predicted=True, ground_truth=False)  # FP
metrics.add_result(predicted=False, ground_truth=True)  # FN
metrics.add_result(predicted=False, ground_truth=False) # TN

# Display metrics
print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"Precision: {metrics.precision:.2%}")
print(f"Recall: {metrics.recall:.2%}")
print(f"F1 Score: {metrics.f1_score:.2%}")

# Display with rich table
metrics.display()
```

### MetricsTracker (`metrics.py`)

Tracks metrics across multiple test runs.

**Features:**
- Accumulate results over time
- Score distribution analysis
- Result persistence (JSON)
- Comprehensive summaries

**Example:**
```python
from easy_oww.testing import MetricsTracker

tracker = MetricsTracker()

# Add multiple results
for prediction, truth, score in test_results:
    tracker.add_result(
        predicted=prediction,
        ground_truth=truth,
        score=score
    )

# Display summary
tracker.display_summary()

# Save results
tracker.save_results(Path("test_results.json"))
```

## CLI Usage

### Real-Time Testing

Test your model with live microphone input:

```bash
# Default 60-second test
easy-oww test my_wake_word

# Custom duration
easy-oww test my_wake_word --duration 120

# Interactive test type selection
easy-oww test my_wake_word
> Select test type: 1. Real-time microphone test
```

### Clip Evaluation

Evaluate model accuracy on test clips:

```bash
easy-oww test my_wake_word
> Select test type: 2. Evaluate on test clips
```

### Combined Testing

Run both real-time and clip evaluation:

```bash
easy-oww test my_wake_word
> Select test type: 3. Both
```

## Test Workflow

### 1. Real-Time Microphone Test

**What It Does:**
- Loads trained ONNX model
- Opens microphone stream
- Processes audio in real-time
- Displays detections as they occur
- Shows detection rate and timing

**Best For:**
- Quick functionality verification
- User experience testing
- Real-world performance assessment

**Example Output:**
```
┌─────────────────────────────────────────┐
│  Real-Time Wake Word Detection Test    │
│                                         │
│  Model: my_wake_word.onnx               │
│  Duration: 60 seconds                   │
│  Threshold: 0.5                         │
│                                         │
│  Speak your wake word to test          │
└─────────────────────────────────────────┘

✓ Model loaded successfully

Starting detection...
Press Ctrl+C to stop

┌── Detection Status ──────────────────┐
│ Elapsed Time    │ 15.3s              │
│ Remaining Time  │ 44.7s              │
│ Detections      │ 3                  │
│ Last Detection  │ 2.1s ago           │
└──────────────────────────────────────┘
```

### 2. Clip Evaluation

**What It Does:**
- Loads positive and negative test clips
- Runs model prediction on each clip
- Calculates comprehensive metrics
- Displays confusion matrix
- Saves detailed results

**Best For:**
- Quantitative accuracy measurement
- Model comparison
- Identifying weaknesses

**Example Output:**
```
┌── Detection Metrics ────────────────┐
│ Total Samples  │ 200                │
│ Accuracy       │ 92.50%             │
│ Precision      │ 90.00%             │
│ Recall         │ 94.74%             │
│ F1 Score       │ 92.31%             │
│                                     │
│ True Positives  │ 90                │
│ True Negatives  │ 95                │
│ False Positives │ 10                │
│ False Negatives │ 5                 │
└─────────────────────────────────────┘

Confusion Matrix:
┌────────────────┬──────────────────┬──────────────────┐
│                │ Predicted Pos    │ Predicted Neg    │
├────────────────┼──────────────────┼──────────────────┤
│ Actual Pos     │ 90 (TP)          │ 5 (FN)           │
│ Actual Neg     │ 10 (FP)          │ 95 (TN)          │
└────────────────┴──────────────────┴──────────────────┘

✓ Results saved to test_results.json
```

## Configuration Options

### Detection Threshold

Controls sensitivity vs. specificity trade-off:

**Lower Threshold (e.g., 0.3):**
- More detections
- Higher recall (fewer misses)
- More false positives
- Good for: High-importance wake words

**Higher Threshold (e.g., 0.7):**
- Fewer detections
- Higher precision (fewer false alarms)
- More false negatives
- Good for: Low-tolerance scenarios

**Default: 0.5** - Balanced trade-off

### Test Duration

For real-time tests:
- **Quick test**: 30 seconds
- **Standard**: 60 seconds (default)
- **Thorough**: 120+ seconds

Longer duration provides more opportunities to test edge cases.

### Sample Limits

For clip evaluation, limit samples for faster testing:
- **Quick**: 50 samples per class
- **Standard**: 100 samples per class (default)
- **Thorough**: All available samples

## Understanding Metrics

### Confusion Matrix

```
                 Predicted
              Positive  Negative
Actual  Pos      TP        FN
        Neg      FP        TN
```

- **TP (True Positive)**: Correctly detected wake word
- **TN (True Negative)**: Correctly rejected non-wake-word
- **FP (False Positive)**: Incorrectly detected wake word
- **FN (False Negative)**: Missed wake word

### Key Metrics

**Accuracy**: Overall correctness
```
Accuracy = (TP + TN) / Total
```
Good for: Balanced datasets

**Precision**: Quality of positive predictions
```
Precision = TP / (TP + FP)
```
Good for: Minimizing false alarms

**Recall**: Coverage of actual positives
```
Recall = TP / (TP + FN)
```
Good for: Minimizing misses

**F1 Score**: Balance of precision and recall
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Good for: Overall performance

## Troubleshooting

### No Detections in Real-Time Test

**Possible Causes:**
- Threshold too high
- Microphone not working
- Background noise too loud
- Model not well-trained

**Solutions:**
```bash
# Test with lower threshold
easy-oww test my_wake_word
> Customize threshold? Yes
> Enter threshold: 0.3

# Check microphone
# Speak louder and closer
# Reduce background noise
```

### Low Accuracy on Clips

**If Low Precision (many false positives):**
- Increase detection threshold
- Add more diverse negative samples
- Train longer

**If Low Recall (many false negatives):**
- Decrease detection threshold
- Add more positive sample variations
- Use more TTS voices

### Model Loading Fails

```
Error: Failed to load model
```

**Solutions:**
1. Ensure ONNX model exists:
   ```bash
   ls ~/.easy-oww/projects/my_wake_word/models/
   ```

2. Install onnxruntime:
   ```bash
   pip install onnxruntime
   ```

3. For OpenWakeWord models:
   ```bash
   pip install openwakeword
   ```

## Performance Optimization

### Reduce Latency

- Use smaller buffer sizes
- Increase chunk processing frequency
- Use optimized ONNX runtime

### Improve Accuracy

**Data Quality:**
- Record in quiet environment
- Use good microphone
- Vary recording conditions

**Model Quality:**
- Train with more samples
- Use multiple TTS voices
- Apply strong augmentation

**Threshold Tuning:**
- Test multiple thresholds
- Find optimal F1 score
- Consider use case requirements

## Advanced Usage

### Custom Detection Logic

```python
from easy_oww.testing import ModelDetector

detector = ModelDetector(model_path=Path("model.onnx"))
detector.load_model()

# Custom callback
def my_callback(detection):
    score = detection['score']
    timestamp = detection['timestamp']
    print(f"Wake word detected! Score: {score:.2f} at {timestamp}")

    # Trigger action
    if score > 0.8:
        # High confidence action
        pass
    elif score > 0.5:
        # Medium confidence action
        pass

# Run with callback
detector.start_detection(on_detection=my_callback)
```

### Batch Evaluation

```python
from easy_oww.testing import ModelDetector, evaluate_model_on_dataset
from pathlib import Path

# Prepare test sets
positive_clips = list(Path("test_data/positive").glob("*.wav"))
negative_clips = list(Path("test_data/negative").glob("*.wav"))

# Create detector
detector = ModelDetector(model_path=Path("model.onnx"))

# Evaluate
tracker = evaluate_model_on_dataset(
    detector=detector,
    positive_clips=positive_clips,
    negative_clips=negative_clips
)

# Display results
tracker.display_summary()

# Get specific metrics
metrics = tracker.get_metrics()
print(f"F1 Score: {metrics.f1_score:.2%}")
```

### Threshold Optimization

```python
import numpy as np
from easy_oww.testing import ModelDetector, MetricsTracker

# Test multiple thresholds
thresholds = np.arange(0.1, 1.0, 0.1)
results = []

for threshold in thresholds:
    detector = ModelDetector(
        model_path=Path("model.onnx"),
        detection_threshold=threshold
    )

    # Evaluate
    tracker = evaluate_model_on_dataset(detector, pos_clips, neg_clips)
    metrics = tracker.get_metrics()

    results.append({
        'threshold': threshold,
        'f1_score': metrics.f1_score,
        'precision': metrics.precision,
        'recall': metrics.recall
    })

# Find optimal threshold
best = max(results, key=lambda x: x['f1_score'])
print(f"Optimal threshold: {best['threshold']:.2f}")
print(f"F1 Score: {best['f1_score']:.2%}")
```

## Testing Best Practices

### Before Deployment

1. **Test in target environment**
   - Use actual deployment hardware
   - Test in realistic conditions
   - Measure real-world performance

2. **Stress test**
   - Test with background noise
   - Test with multiple speakers
   - Test with variations in pronunciation

3. **Long-duration testing**
   - Run for extended periods
   - Monitor false positive rate
   - Track detection consistency

### Continuous Testing

- Test after each training iteration
- Compare metrics across versions
- Keep test results for regression testing
- Build test dataset from real usage

## Testing

Run testing module tests:

```bash
pytest tests/test_testing.py -v
```

## Future Enhancements

Planned features:
- ROC curve generation
- Threshold optimization tools
- Multi-model comparison
- Real-time visualization dashboard
- Cloud-based testing
- A/B testing framework
