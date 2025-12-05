"""
Dataset download orchestration and management
"""
from pathlib import Path
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from easy_oww.datasets.acav100m_features import ACAV100MDownloader
from easy_oww.datasets.rir import RIRDownloader
from easy_oww.datasets.fsd50k import FSD50kDownloader
from easy_oww.datasets.cache import CacheManager


console = Console()


class DatasetManager:
    """Manages all dataset downloads"""

    def __init__(self, datasets_dir: str, cache_dir: str):
        """
        Initialize dataset manager

        Args:
            datasets_dir: Directory to store datasets
            cache_dir: Directory for cache metadata
        """
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        self.cache = CacheManager(cache_dir)

        # Initialize downloaders
        self.acav100m = ACAV100MDownloader(str(self.datasets_dir))
        self.rir = RIRDownloader(str(self.datasets_dir))
        self.fsd50k = FSD50kDownloader(str(self.datasets_dir))

    def get_dataset_info(self) -> List[Dict]:
        """
        Get information about all datasets

        Returns:
            List of dataset info dictionaries
        """
        return [
            {
                'name': 'ACAV100M Training',
                'key': 'acav100m_train',
                'size_gb': ACAV100MDownloader.TRAIN_SIZE_GB,
                'priority': 'Critical',
                'cached': self.acav100m.is_training_cached(),
                'description': 'Negative training features'
            },
            {
                'name': 'ACAV100M Validation',
                'key': 'acav100m_val',
                'size_gb': ACAV100MDownloader.VAL_SIZE_GB,
                'priority': 'Critical',
                'cached': self.acav100m.is_validation_cached(),
                'description': 'Model validation features'
            },
            {
                'name': 'MIT Room Impulse Responses',
                'key': 'rir',
                'size_gb': RIRDownloader.SIZE_GB,
                'priority': 'Recommended',
                'cached': self.rir.is_cached(),
                'description': 'Acoustic realism'
            },
            {
                'name': 'FSD50K Sounds',
                'key': 'fsd50k',
                'size_gb': FSD50kDownloader.SIZE_GB,
                'priority': 'Optional',
                'cached': self.fsd50k.is_cached(),
                'description': 'Background sound diversity'
            },
        ]

    def show_status(self):
        """Display dataset download status"""
        table = Table(title="Dataset Status")
        table.add_column("Dataset", style="cyan")
        table.add_column("Size", style="white")
        table.add_column("Priority", style="yellow")
        table.add_column("Status", style="magenta")
        table.add_column("Description", style="white")

        for dataset in self.get_dataset_info():
            status = "[green]✓ Cached[/green]" if dataset['cached'] else "[yellow]Not downloaded[/yellow]"
            table.add_row(
                dataset['name'],
                f"{dataset['size_gb']}GB",
                dataset['priority'],
                status,
                dataset['description']
            )

        console.print(table)

        # Calculate total size
        total_size = sum(d['size_gb'] for d in self.get_dataset_info() if not d['cached'])
        if total_size > 0:
            console.print(f"\n[yellow]Total download size:[/yellow] ~{total_size}GB")

    def download_required(self):
        """Download only required (critical) datasets"""
        console.print("\n[bold]Downloading required datasets...[/bold]\n")

        # ACAV100M Training Features
        if not self.acav100m.is_training_cached():
            try:
                train_path = self.acav100m.download_training_features(resume=True)
                self.cache.add_entry('acav100m_train', str(train_path))
                console.print("[green]✓ ACAV100M training features downloaded[/green]\n")
            except Exception as e:
                console.print(f"[red]✗ Failed to download ACAV100M training: {e}[/red]\n")
        else:
            console.print("[green]✓ ACAV100M training features already cached[/green]\n")

        # ACAV100M Validation Features
        if not self.acav100m.is_validation_cached():
            try:
                val_path = self.acav100m.download_validation_features(resume=True)
                self.cache.add_entry('acav100m_val', str(val_path))
                console.print("[green]✓ ACAV100M validation features downloaded[/green]\n")
            except Exception as e:
                console.print(f"[red]✗ Failed to download ACAV100M validation: {e}[/red]\n")
        else:
            console.print("[green]✓ ACAV100M validation features already cached[/green]\n")

    def download_optional(self):
        """Download optional (recommended + optional) datasets"""
        console.print("\n[bold]Downloading optional datasets...[/bold]\n")

        # MIT RIR
        if not self.rir.is_cached():
            try:
                self.rir.download_all()
                self.cache.add_entry('rir', str(self.rir.dest_dir))
                console.print("[green]✓ MIT RIR downloaded[/green]\n")
            except Exception as e:
                console.print(f"[red]✗ Failed to download MIT RIR: {e}[/red]\n")
        else:
            console.print("[green]✓ MIT RIR already cached[/green]\n")

        # FSD50K
        if not self.fsd50k.is_cached():
            try:
                self.fsd50k.download_all()
                self.cache.add_entry('fsd50k', str(self.fsd50k.dest_dir))
                console.print("[green]✓ FSD50K downloaded[/green]\n")
            except Exception as e:
                console.print(f"[red]✗ Failed to download FSD50K: {e}[/red]\n")
        else:
            console.print("[green]✓ FSD50K already cached[/green]\n")

    def download_all(self, required_only: bool = False):
        """
        Download all datasets

        Args:
            required_only: If True, only download critical datasets
        """
        self.show_status()

        # Download required
        self.download_required()

        # Download optional if requested
        if not required_only:
            self.download_optional()

        console.print("\n[bold green]Download complete![/bold green]")
        self.show_status()

    def verify_critical_datasets(self) -> bool:
        """
        Verify that critical datasets are available

        Returns:
            True if all critical datasets are cached
        """
        critical = [
            self.acav100m.is_training_cached(),
            self.acav100m.is_validation_cached()
        ]
        return all(critical)

    def get_dataset_paths(self) -> Dict[str, Path]:
        """
        Get paths to all datasets

        Returns:
            Dictionary mapping dataset keys to paths
        """
        return {
            'acav100m_train': self.datasets_dir / 'acav100m_train.npy',
            'acav100m_val': self.datasets_dir / 'acav100m_val.npy',
            'rir': self.datasets_dir / 'mit_rir',
            'fsd50k': self.datasets_dir / 'fsd50k',
        }
