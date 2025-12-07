"""
Configuration manager for easy-oww

Stores and retrieves user configuration like workspace path
"""
import json
from pathlib import Path
from typing import Optional


class ConfigManager:
    """Manages user configuration"""

    # Store config in user's home directory
    CONFIG_PATH = Path.home() / '.easy-oww-config.json'

    @classmethod
    def save_workspace(cls, workspace_path: str):
        """
        Save workspace path to config

        Args:
            workspace_path: Path to workspace directory
        """
        config = cls._load_config()
        config['workspace'] = str(workspace_path)
        cls._save_config(config)

    @classmethod
    def get_workspace(cls) -> Optional[str]:
        """
        Get saved workspace path

        Returns:
            Workspace path if configured, None otherwise
        """
        config = cls._load_config()
        return config.get('workspace')

    @classmethod
    def _load_config(cls) -> dict:
        """Load configuration from file"""
        if not cls.CONFIG_PATH.exists():
            return {}

        try:
            with open(cls.CONFIG_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    @classmethod
    def _save_config(cls, config: dict):
        """Save configuration to file"""
        try:
            with open(cls.CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
        except IOError:
            pass  # Silently fail if we can't write config
