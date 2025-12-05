"""
Tests for testing/validation functionality
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
from scipy.io import wavfile
from unittest.mock import Mock, patch, MagicMock

from easy_oww.testing.detector import ModelDetector
from easy_oww.testing.metrics import DetectionMetrics, MetricsTracker


class TestDetectionMetrics:
    """Tests for DetectionMetrics class"""

    def test_init(self):
        """Test metrics initialization"""
        metrics = DetectionMetrics()
        assert metrics.true_positives == 0
        assert metrics.true_negatives == 0
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0

    def test_add_result_true_positive(self):
        """Test adding true positive result"""
        metrics = DetectionMetrics()
        metrics.add_result(predicted=True, ground_truth=True)
        assert metrics.true_positives == 1
        assert metrics.total_samples == 1

    def test_add_result_false_positive(self):
        """Test adding false positive result"""
        metrics = DetectionMetrics()
        metrics.add_result(predicted=True, ground_truth=False)
        assert metrics.false_positives == 1

    def test_add_result_true_negative(self):
        """Test adding true negative result"""
        metrics = DetectionMetrics()
        metrics.add_result(predicted=False, ground_truth=False)
        assert metrics.true_negatives == 1

    def test_add_result_false_negative(self):
        """Test adding false negative result"""
        metrics = DetectionMetrics()
        metrics.add_result(predicted=False, ground_truth=True)
        assert metrics.false_negatives == 1

    def test_accuracy(self):
        """Test accuracy calculation"""
        metrics = DetectionMetrics()
        metrics.true_positives = 8
        metrics.true_negatives = 7
        metrics.false_positives = 2
        metrics.false_negatives = 3

        accuracy = metrics.accuracy
        assert accuracy == 15 / 20  # (8 + 7) / 20

    def test_precision(self):
        """Test precision calculation"""
        metrics = DetectionMetrics()
        metrics.true_positives = 8
        metrics.false_positives = 2

        precision = metrics.precision
        assert precision == 8 / 10  # 8 / (8 + 2)

    def test_recall(self):
        """Test recall calculation"""
        metrics = DetectionMetrics()
        metrics.true_positives = 8
        metrics.false_negatives = 3

        recall = metrics.recall
        assert recall == 8 / 11  # 8 / (8 + 3)

    def test_f1_score(self):
        """Test F1 score calculation"""
        metrics = DetectionMetrics()
        metrics.true_positives = 8
        metrics.false_positives = 2
        metrics.false_negatives = 3

        # Precision = 8/10 = 0.8
        # Recall = 8/11 ≈ 0.727
        # F1 = 2 * (0.8 * 0.727) / (0.8 + 0.727) ≈ 0.762

        f1 = metrics.f1_score
        assert 0.76 < f1 < 0.77

    def test_specificity(self):
        """Test specificity calculation"""
        metrics = DetectionMetrics()
        metrics.true_negatives = 7
        metrics.false_positives = 2

        specificity = metrics.specificity
        assert specificity == 7 / 9  # 7 / (7 + 2)

    def test_metrics_with_no_samples(self):
        """Test metrics with zero samples"""
        metrics = DetectionMetrics()
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary"""
        metrics = DetectionMetrics()
        metrics.true_positives = 5
        metrics.false_positives = 2

        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert 'accuracy' in metrics_dict
        assert 'precision' in metrics_dict
        assert 'recall' in metrics_dict
        assert metrics_dict['true_positives'] == 5


class TestMetricsTracker:
    """Tests for MetricsTracker class"""

    def test_init(self):
        """Test tracker initialization"""
        tracker = MetricsTracker()
        assert len(tracker.runs) == 0
        assert tracker.current_metrics.total_samples == 0

    def test_add_result(self):
        """Test adding results"""
        tracker = MetricsTracker()

        tracker.add_result(predicted=True, ground_truth=True, score=0.8)
        tracker.add_result(predicted=False, ground_truth=False, score=0.2)

        assert len(tracker.runs) == 2
        assert tracker.current_metrics.true_positives == 1
        assert tracker.current_metrics.true_negatives == 1

    def test_reset(self):
        """Test resetting tracker"""
        tracker = MetricsTracker()
        tracker.add_result(predicted=True, ground_truth=True)
        tracker.reset()

        assert len(tracker.runs) == 0
        assert tracker.current_metrics.total_samples == 0

    def test_get_score_distribution(self):
        """Test score distribution"""
        tracker = MetricsTracker()

        tracker.add_result(predicted=True, ground_truth=True, score=0.8)
        tracker.add_result(predicted=True, ground_truth=True, score=0.9)
        tracker.add_result(predicted=False, ground_truth=False, score=0.2)
        tracker.add_result(predicted=False, ground_truth=False, score=0.3)

        dist = tracker.get_score_distribution()

        assert len(dist['positive']) == 2
        assert len(dist['negative']) == 2
        assert 0.8 in dist['positive']
        assert 0.2 in dist['negative']

    def test_save_results(self):
        """Test saving results to file"""
        tracker = MetricsTracker()
        tracker.add_result(predicted=True, ground_truth=True, score=0.8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'results.json'
            tracker.save_results(output_path)

            assert output_path.exists()

            # Verify contents
            import json
            with open(output_path) as f:
                data = json.load(f)

            assert 'metrics' in data
            assert 'runs' in data
            assert len(data['runs']) == 1


class TestModelDetector:
    """Tests for ModelDetector class"""

    def test_init(self):
        """Test detector initialization"""
        detector = ModelDetector(
            model_path=Path("test_model.onnx"),
            sample_rate=16000,
            detection_threshold=0.5
        )

        assert detector.model_path == Path("test_model.onnx")
        assert detector.sample_rate == 16000
        assert detector.detection_threshold == 0.5
        assert not detector.model_loaded

    def test_audio_buffer_initialization(self):
        """Test audio buffer setup"""
        detector = ModelDetector()
        assert len(detector.audio_buffer) == 0
        assert detector.audio_buffer.maxlen == detector.buffer_samples

    def test_process_audio_chunk_insufficient_data(self):
        """Test processing with insufficient buffer data"""
        detector = ModelDetector()

        # Add small chunk
        chunk = np.zeros(100, dtype=np.int16)
        result = detector.process_audio_chunk(chunk)

        # Should return None (not enough data yet)
        assert result is None

    @patch('easy_oww.testing.detector.ModelDetector.predict')
    def test_process_audio_chunk_with_detection(self, mock_predict):
        """Test audio chunk processing with detection"""
        detector = ModelDetector(detection_threshold=0.5)

        # Mock prediction to return high score
        mock_predict.return_value = 0.8

        # Fill buffer
        total_samples = detector.buffer_samples
        chunk = np.random.randint(-1000, 1000, total_samples, dtype=np.int16)

        result = detector.process_audio_chunk(chunk)

        # Should detect
        assert result is not None
        assert result['score'] == 0.8
        assert 'timestamp' in result
        assert 'audio' in result

    @patch('easy_oww.testing.detector.ModelDetector.predict')
    def test_process_audio_chunk_no_detection(self, mock_predict):
        """Test audio chunk processing without detection"""
        detector = ModelDetector(detection_threshold=0.5)

        # Mock prediction to return low score
        mock_predict.return_value = 0.2

        # Fill buffer
        total_samples = detector.buffer_samples
        chunk = np.random.randint(-1000, 1000, total_samples, dtype=np.int16)

        result = detector.process_audio_chunk(chunk)

        # Should not detect
        assert result is None

    def test_stop_detection(self):
        """Test stopping detection"""
        detector = ModelDetector()
        detector.is_running = True

        detector.stop_detection()

        assert not detector.is_running


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
