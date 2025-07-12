#!/usr/bin/env python3
"""Setup file for Enhanced CSP Network package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Enhanced CSP System - Advanced Computing Systems Platform"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "cryptography>=41.0.0",
        "base58>=2.1.0",
    ]

setup(
    name="enhanced-csp",
    version="1.0.0",
    author="Enhanced CSP Team",
    author_email="team@enhanced-csp.com",
    description="Advanced Computing Systems Platform with AI, Quantum, and Distributed Computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enhanced-csp/network",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Networking",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "flake8>=6.0.0",
        ],
        "performance": [
            "msgpack>=1.0.0",
            "lz4>=4.0.0",
            "zstandard>=0.21.0",
            "uvloop>=0.17.0",
            "psutil>=5.9.0",
        ],
        "quantum": [
            "qiskit>=0.44.0",
            "cirq>=1.2.0",
        ],
        "ai": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "numpy>=1.24.0",
        ],
        "web": [
            "aiohttp>=3.8.0",
            "websockets>=11.0.0",
            "jinja2>=3.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "enhanced-csp=enhanced_csp.main:main",
            "csp-network=enhanced_csp.network.main:main",
            "csp-test=enhanced_csp.network.test_optimizations:main",
        ],
    },
    include_package_data=True,
    package_data={
        "enhanced_csp": ["*.txt", "*.md", "*.yaml", "*.json"],
        "enhanced_csp.network": ["*.txt", "*.md", "*.yaml", "*.json"],
    },
    zip_safe=False,
)