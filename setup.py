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
        "openwakeword>=0.5.0",
        "sounddevice>=0.4.6",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "tqdm>=4.64.0",
        "click>=8.1.0",
        "questionary>=1.10.0",
        "rich>=13.0.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "gpu": [
            "torch>=2.0.0",
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
