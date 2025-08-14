#!/usr/bin/env python3
"""
Setup script for Enhanced AI Music Generation Suite
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="enhanced-music-generator",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced AI-powered music generation suite with ML capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/enhanced-music-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "ml": ["tensorflow>=2.6.0", "magenta>=2.1.0"],
        "dev": ["pytest>=6.0.0", "pytest-asyncio>=0.15.0", "black>=21.0.0", "flake8>=3.9.0", "mypy>=0.910"],
        "docs": ["mkdocs>=1.2.0", "mkdocs-material>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "music-generator=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
    },
    keywords="music generation ai ml audio synthesis algorithmic composition",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/enhanced-music-generator/issues",
        "Source": "https://github.com/yourusername/enhanced-music-generator",
        "Documentation": "https://enhanced-music-generator.readthedocs.io/",
    },
)
