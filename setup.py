"""
Setup script for SimpleMem — Efficient Lifelong Memory for LLM Agents (Text Memory Core).
"""

import re
from pathlib import Path
from setuptools import setup, find_packages


_HERE = Path(__file__).parent
readme = (_HERE / "README.md")
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""


def _read_version() -> str:
    """Single source of truth: simplemem/__init__.py __version__."""
    init = (_HERE / "simplemem" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init, re.M)
    if not match:
        raise RuntimeError("Unable to find __version__ in simplemem/__init__.py")
    return match.group(1)


# Text memory core dependencies
INSTALL_REQUIRES = [
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "lancedb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "numpy>=1.24.0",
    "dateparser>=1.1.0",
    "pyarrow>=12.0.0",
    "tantivy>=0.20.0",
    "tqdm>=4.60.0",
]


EXTRAS = {
    # Benchmark / evaluation extras
    "benchmark": [
        "datasets>=2.0.0",
        "bert-score>=0.3.0",
        "rouge-score>=0.1.0",
        "nltk>=3.8.0",
    ],
    # Dev tooling
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
    ],
}
EXTRAS["all"] = sorted({pkg for group in EXTRAS.values() for pkg in group})


setup(
    name="simplemem",
    version=_read_version(),
    author="SimpleMem Team",
    description="Efficient Lifelong Memory for LLM Agents — Text Memory Core",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aiming-lab/SimpleMem",
    packages=find_packages(include=["simplemem", "simplemem.*"]),
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
