"""
Model testing and validation for wake word detection
"""
from easy_oww.testing.detector import ModelDetector, run_realtime_test
from easy_oww.testing.metrics import DetectionMetrics, MetricsTracker

__all__ = [
    'ModelDetector',
    'run_realtime_test',
    'DetectionMetrics',
    'MetricsTracker'
]
