"""
Path management for easy-oww workspace and projects
"""
import os
from pathlib import Path
from typing import Optional


class PathManager:
    """Manages workspace paths and project directories"""

    def __init__(self, workspace: Optional[str] = None):
        """
        Initialize path manager

        Args:
            workspace: Custom workspace path (default: ~/.easy-oww)
        """
        if workspace is None:
            workspace = os.path.expanduser('~/.easy-oww')

        self.workspace = Path(workspace)
        self.datasets = self.workspace / 'datasets'
        self.projects = self.workspace / 'projects'
        self.voices = self.workspace / 'voices'
        self.piper = self.workspace / 'piper-sample-generator'
        self.cache = self.workspace / '.cache'

    def ensure_structure(self):
        """Create workspace directory structure"""
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(exist_ok=True)
        self.projects.mkdir(exist_ok=True)
        self.voices.mkdir(exist_ok=True)
        self.cache.mkdir(exist_ok=True)

    def get_project_path(self, project_name: str) -> Path:
        """
        Get path for specific project

        Args:
            project_name: Name of the project

        Returns:
            Path to project directory
        """
        return self.projects / project_name

    def project_exists(self, project_name: str) -> bool:
        """
        Check if project exists

        Args:
            project_name: Name of the project

        Returns:
            True if project exists
        """
        return self.get_project_path(project_name).exists()

    def create_project_structure(self, project_name: str):
        """
        Create project directory structure

        Args:
            project_name: Name of the project
        """
        project = self.get_project_path(project_name)
        project.mkdir(exist_ok=True)
        (project / 'recordings').mkdir(exist_ok=True)
        (project / 'clips').mkdir(exist_ok=True)
        (project / 'features').mkdir(exist_ok=True)
        (project / 'models').mkdir(exist_ok=True)

    def get_dataset_path(self, dataset_name: str) -> Path:
        """
        Get path for specific dataset

        Args:
            dataset_name: Name of the dataset

        Returns:
            Path to dataset directory
        """
        return self.datasets / dataset_name

    def get_voice_path(self, voice_name: str) -> Path:
        """
        Get path for specific voice model

        Args:
            voice_name: Name of the voice model

        Returns:
            Path to voice model file
        """
        return self.voices / f"{voice_name}.onnx"
