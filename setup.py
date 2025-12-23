"""
easy-oww: Simplified OpenWakeWord ONNX model creation tool
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="easy-oww",
    version="0.1.0",
    author="Easy-OWW Contributors",
    description="A CLI tool that simplifies OpenWakeWord ONNX model creation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/easy-oww",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Core dependencies
        "openwakeword>=0.5.0",
        "sounddevice>=0.4.6",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "tqdm>=4.64.0",
        # CLI and UI
        "click>=8.1.0",
        "questionary>=1.10.0",
        "rich>=13.0.0",
        # System utilities
        "psutil>=5.9.0",
        # Dataset downloads
        "datasets>=2.0.0",
        "soundfile>=0.12.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "torchcodec>=0.1.0",
        # Model training
        "torchinfo>=1.8.0",
        "torchmetrics>=0.11.0",
        "pronouncing>=0.2.0",
        "Metaphone>=0.6",
        "audiomentations>=0.30.0",
        "torch-audiomentations>=0.11.0,<0.12.0",
        "speechbrain>=1.0.0,<2.0.0",
        "librosa>=0.10.0",
        "resampy>=0.4.0",
        "mutagen>=1.45.0",
        "acoustics>=0.2.0",
        "onnx>=1.15.0",
        "onnxscript>=0.1.0",
        # TTS (Text-to-Speech)
        "piper-tts>=1.3.0",
        "openai>=1.0.0",
        # ONNX Runtime (required for model inference)
        "onnxruntime>=1.15.0",
    ],
    extras_require={
        "gpu": [
            "onnxruntime-gpu>=1.15.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "easy-oww=easy_oww.cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "easy_oww": ["templates/*"],
    },
)
