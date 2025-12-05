"""
Progress tracking utilities for easy-oww
"""
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    TaskID
)
from rich.console import Console
from typing import Optional


class ProgressTracker:
    """Manages progress bars and status updates"""

    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
        self._active = False

    def __enter__(self):
        """Start progress tracking"""
        if not self._active:
            self.progress.start()
            self._active = True
        return self

    def __exit__(self, *args):
        """Stop progress tracking"""
        if self._active:
            self.progress.stop()
            self._active = False

    def add_task(self, description: str, total: Optional[int] = None) -> TaskID:
        """
        Add new progress task

        Args:
            description: Task description
            total: Total units for the task (None for indeterminate)

        Returns:
            Task ID
        """
        return self.progress.add_task(description, total=total)

    def update(self, task_id: TaskID, advance: int = 1, **kwargs):
        """
        Update task progress

        Args:
            task_id: Task ID to update
            advance: Amount to advance
            **kwargs: Additional update parameters
        """
        self.progress.update(task_id, advance=advance, **kwargs)

    def log(self, message: str, style: str = None):
        """
        Log message to console

        Args:
            message: Message to log
            style: Optional rich style
        """
        if style:
            self.console.print(message, style=style)
        else:
            self.console.print(message)


class DownloadProgressTracker:
    """Specialized progress tracker for downloads"""

    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        self._active = False

    def __enter__(self):
        """Start progress tracking"""
        if not self._active:
            self.progress.start()
            self._active = True
        return self

    def __exit__(self, *args):
        """Stop progress tracking"""
        if self._active:
            self.progress.stop()
            self._active = False

    def add_download(self, description: str, total_bytes: int) -> TaskID:
        """
        Add new download task

        Args:
            description: Download description
            total_bytes: Total bytes to download

        Returns:
            Task ID
        """
        return self.progress.add_task(description, total=total_bytes)

    def update(self, task_id: TaskID, advance: int):
        """
        Update download progress

        Args:
            task_id: Task ID to update
            advance: Bytes downloaded
        """
        self.progress.update(task_id, advance=advance)

    def log(self, message: str, style: str = None):
        """
        Log message to console

        Args:
            message: Message to log
            style: Optional rich style
        """
        if style:
            self.console.print(message, style=style)
        else:
            self.console.print(message)
