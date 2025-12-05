"""
System requirement checks for easy-oww
"""
import shutil
import psutil
import platform
import sys
from typing import Dict, Tuple


class SystemChecker:
    """Check system requirements for easy-oww"""

    @staticmethod
    def check_python_version() -> Tuple[bool, str]:
        """
        Check if Python version >= 3.7

        Returns:
            Tuple of (is_valid, version_string)
        """
        version = sys.version_info
        is_valid = version.major >= 3 and version.minor >= 7
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        return is_valid, version_str

    @staticmethod
    def check_disk_space(path: str, required_gb: int = 60) -> Tuple[bool, float]:
        """
        Check available disk space

        Args:
            path: Path to check disk space for
            required_gb: Required space in GB

        Returns:
            Tuple of (has_enough_space, available_gb)
        """
        stat = shutil.disk_usage(path)
        available_gb = stat.free / (1024**3)
        has_enough = available_gb >= required_gb
        return has_enough, available_gb

    @staticmethod
    def check_memory(required_gb: int = 8) -> Tuple[bool, float]:
        """
        Check available RAM

        Args:
            required_gb: Required RAM in GB

        Returns:
            Tuple of (has_enough_memory, available_gb)
        """
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        has_enough = available_gb >= required_gb
        return has_enough, available_gb

    @staticmethod
    def check_gpu() -> Dict[str, any]:
        """
        Check for CUDA GPU availability

        Returns:
            Dictionary with GPU information
        """
        try:
            import torch
            return {
                'available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            }
        except ImportError:
            return {
                'available': False,
                'device_count': 0,
                'device_name': None,
                'torch_not_installed': True
            }

    @staticmethod
    def get_platform_info() -> Dict[str, str]:
        """
        Get platform information

        Returns:
            Dictionary with platform details
        """
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }

    @staticmethod
    def check_all_requirements(workspace_path: str) -> Dict[str, any]:
        """
        Check all system requirements

        Args:
            workspace_path: Path to workspace directory

        Returns:
            Dictionary with all check results
        """
        python_ok, python_version = SystemChecker.check_python_version()
        disk_ok, disk_gb = SystemChecker.check_disk_space(workspace_path)
        memory_ok, memory_gb = SystemChecker.check_memory()
        gpu_info = SystemChecker.check_gpu()
        platform_info = SystemChecker.get_platform_info()

        return {
            'python': {
                'valid': python_ok,
                'version': python_version
            },
            'disk': {
                'sufficient': disk_ok,
                'available_gb': round(disk_gb, 1)
            },
            'memory': {
                'sufficient': memory_ok,
                'available_gb': round(memory_gb, 1)
            },
            'gpu': gpu_info,
            'platform': platform_info
        }
