"""Dataset management and downloading for easy-oww"""

from easy_oww.datasets.cache import CacheManager
from easy_oww.datasets.acav100m_features import ACAV100MDownloader
from easy_oww.datasets.rir import RIRDownloader
from easy_oww.datasets.fsd50k import FSD50kDownloader
from easy_oww.datasets.manager import DatasetManager

__all__ = [
    'CacheManager',
    'ACAV100MDownloader',
    'RIRDownloader',
    'FSD50kDownloader',
    'DatasetManager'
]
