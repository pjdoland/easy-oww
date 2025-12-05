"""
Detection metrics and evaluation
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class DetectionMetrics:
    """Metrics for wake word detection performance"""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def add_result(self, predicted: bool, ground_truth: bool):
        """
        Add a detection result

        Args:
            predicted: Model prediction (detected or not)
            ground_truth: Actual label (wake word present or not)
        """
        if predicted and ground_truth:
            self.true_positives += 1
        elif not predicted and not ground_truth:
            self.true_negatives += 1
        elif predicted and not ground_truth:
            self.false_positives += 1
        else:  # not predicted and ground_truth
            self.false_negatives += 1

    @property
    def total_samples(self) -> int:
        """Total number of samples"""
        return self.true_positives + self.true_negatives + self.false_positives + self.false_negatives

    @property
    def accuracy(self) -> float:
        """Overall accuracy"""
        if self.total_samples == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total_samples

    @property
    def precision(self) -> float:
        """Precision (positive predictive value)"""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        """Recall (sensitivity, true positive rate)"""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def f1_score(self) -> float:
        """F1 score (harmonic mean of precision and recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def specificity(self) -> float:
        """Specificity (true negative rate)"""
        denominator = self.true_negatives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_negatives / denominator

    @property
    def false_positive_rate(self) -> float:
        """False positive rate"""
        return 1.0 - self.specificity

    @property
    def false_negative_rate(self) -> float:
        """False negative rate"""
        return 1.0 - self.recall

    def to_dict(self) -> Dict[str, float]:
        """
        Convert metrics to dictionary

        Returns:
            Dictionary with all metrics
        """
        return {
            'total_samples': self.total_samples,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'specificity': self.specificity,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }

    def display(self, title: str = "Detection Metrics"):
        """
        Display metrics in rich table

        Args:
            title: Table title
        """
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        # Overall metrics
        table.add_row("Total Samples", str(self.total_samples))
        table.add_row("Accuracy", f"{self.accuracy:.2%}")
        table.add_row("Precision", f"{self.precision:.2%}")
        table.add_row("Recall", f"{self.recall:.2%}")
        table.add_row("F1 Score", f"{self.f1_score:.2%}")

        # Confusion matrix
        table.add_row("", "")  # Separator
        table.add_row("True Positives", str(self.true_positives))
        table.add_row("True Negatives", str(self.true_negatives))
        table.add_row("False Positives", str(self.false_positives))
        table.add_row("False Negatives", str(self.false_negatives))

        console.print(table)

    def display_confusion_matrix(self):
        """Display confusion matrix"""
        console.print("\n[bold]Confusion Matrix:[/bold]")

        matrix_table = Table(show_header=True, header_style="bold")
        matrix_table.add_column("", style="cyan")
        matrix_table.add_column("Predicted Positive", justify="center")
        matrix_table.add_column("Predicted Negative", justify="center")

        matrix_table.add_row(
            "Actual Positive",
            f"[green]{self.true_positives}[/green]",
            f"[red]{self.false_negatives}[/red]"
        )
        matrix_table.add_row(
            "Actual Negative",
            f"[yellow]{self.false_positives}[/yellow]",
            f"[green]{self.true_negatives}[/green]"
        )

        console.print(matrix_table)


class MetricsTracker:
    """Tracks metrics over multiple test runs"""

    def __init__(self):
        """Initialize metrics tracker"""
        self.runs: List[Dict] = []
        self.current_metrics = DetectionMetrics()

    def add_result(
        self,
        predicted: bool,
        ground_truth: bool,
        score: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a detection result

        Args:
            predicted: Model prediction
            ground_truth: Actual label
            score: Detection score
            metadata: Additional metadata
        """
        self.current_metrics.add_result(predicted, ground_truth)

        # Store individual result
        result = {
            'predicted': predicted,
            'ground_truth': ground_truth,
            'correct': predicted == ground_truth
        }

        if score is not None:
            result['score'] = score

        if metadata:
            result.update(metadata)

        self.runs.append(result)

    def get_metrics(self) -> DetectionMetrics:
        """
        Get current metrics

        Returns:
            DetectionMetrics instance
        """
        return self.current_metrics

    def reset(self):
        """Reset metrics"""
        self.runs = []
        self.current_metrics = DetectionMetrics()

    def get_score_distribution(self) -> Dict[str, List[float]]:
        """
        Get distribution of scores by class

        Returns:
            Dictionary with 'positive' and 'negative' score lists
        """
        positive_scores = []
        negative_scores = []

        for run in self.runs:
            if 'score' in run:
                if run['ground_truth']:
                    positive_scores.append(run['score'])
                else:
                    negative_scores.append(run['score'])

        return {
            'positive': positive_scores,
            'negative': negative_scores
        }

    def display_summary(self):
        """Display comprehensive summary"""
        console.print("\n" + "=" * 60)
        console.print(Panel.fit(
            "[bold cyan]Test Summary[/bold cyan]",
            title="Results"
        ))

        # Display metrics
        self.current_metrics.display()

        # Display confusion matrix
        self.current_metrics.display_confusion_matrix()

        # Score distribution
        if any('score' in run for run in self.runs):
            dist = self.get_score_distribution()

            console.print("\n[bold]Score Distribution:[/bold]")

            if dist['positive']:
                import numpy as np
                pos_mean = np.mean(dist['positive'])
                pos_std = np.std(dist['positive'])
                console.print(f"  Positive samples: {pos_mean:.3f} ± {pos_std:.3f}")

            if dist['negative']:
                import numpy as np
                neg_mean = np.mean(dist['negative'])
                neg_std = np.std(dist['negative'])
                console.print(f"  Negative samples: {neg_mean:.3f} ± {neg_std:.3f}")

    def save_results(self, output_path: Path):
        """
        Save results to file

        Args:
            output_path: Path to save results
        """
        import json

        results = {
            'metrics': self.current_metrics.to_dict(),
            'runs': self.runs
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[green]✓[/green] Results saved to {output_path}")


def evaluate_model_on_dataset(
    detector,
    positive_clips: List[Path],
    negative_clips: List[Path]
) -> MetricsTracker:
    """
    Evaluate model on dataset

    Args:
        detector: ModelDetector instance
        positive_clips: List of positive sample paths
        negative_clips: List of negative sample paths

    Returns:
        MetricsTracker with results
    """
    from rich.progress import Progress

    tracker = MetricsTracker()

    console.print("\n[bold]Evaluating model on dataset...[/bold]")

    total_samples = len(positive_clips) + len(negative_clips)

    with Progress(console=console) as progress:
        task = progress.add_task("Testing samples...", total=total_samples)

        # Test positive samples
        for clip_path in positive_clips:
            result = detector.test_model(clip_path, ground_truth=True)
            tracker.add_result(
                predicted=result['predicted'],
                ground_truth=result['ground_truth'],
                score=result['score'],
                metadata={'file': str(clip_path)}
            )
            progress.update(task, advance=1)

        # Test negative samples
        for clip_path in negative_clips:
            result = detector.test_model(clip_path, ground_truth=False)
            tracker.add_result(
                predicted=result['predicted'],
                ground_truth=result['ground_truth'],
                score=result['score'],
                metadata={'file': str(clip_path)}
            )
            progress.update(task, advance=1)

    return tracker
