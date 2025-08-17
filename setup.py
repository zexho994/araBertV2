#!/usr/bin/env python3
"""Unified Setup script for AraBERTv2 - DAPT and NER CLI tools"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we're in the right directory
here = Path(__file__).parent.absolute()

# Read the README file
readme_file = here / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "AraBERTv2 - A comprehensive toolkit for Arabic NLP with Named Entity Recognition (NER) capabilities."

# Read requirements
requirements_file = here / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                requirements.append(line)
else:
    # Fallback requirements if file doesn't exist
    requirements = [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.62.0",
        "click>=8.0.0",
        "seqeval>=1.2.2",
        "rich>=12.0.0",
        "arabertv2>=1.0.0",
    ]

# Version information
version = "1.0.0"  # Default version

# Try to get version from NER module first, then DAPT
for module_path in ["src/ner/__init__.py"]:
    version_file = here / module_path
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    version = line.split("=")[1].strip().strip('"').strip("'")
                    break
        break

# Development requirements
dev_requirements = [
    "pytest>=6.2.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
    "isort>=5.10.0",
    "mypy>=0.950",
]

# Documentation requirements
docs_requirements = [
    "sphinx>=4.0",
    "sphinx-rtd-theme>=0.5",
    "myst-parser>=0.15",
]

# Optional requirements for different features
optional_requirements = {
    # Core modules
    "ner": [
        "seqeval>=1.2.2",
        "rich>=12.0.0",
    ],
    # Feature-based extras
    "gpu": ["nvidia-ml-py3>=7.352.0"],
    "visualization": ["matplotlib>=3.5.0", "seaborn>=0.11.0"],
    "api": ["fastapi>=0.75.0", "uvicorn>=0.17.0"],
    "web": ["streamlit>=1.8.0"],
    "notebook": ["jupyter>=1.0.0", "ipywidgets>=7.6.0"],
    "profiling": ["memory-profiler>=0.60.0"],
    "advanced-nlp": ["polyglot>=16.7.4", "langdetect>=1.0.9"],
    "dev": dev_requirements,
    "docs": docs_requirements,
}

# All optional requirements combined
optional_requirements["all"] = [
    req for reqs in optional_requirements.values() for req in reqs
]

setup(
    name="arabertv2-toolkit",
    version=version,
    author="AraBERTv2 Development Team",
    author_email="contact@arabertv2.com",
    description="AraBERTv2 Toolkit - Named Entity Recognition for Arabic NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/araBertv2",
    project_urls={
        "Bug Reports": "https://github.com/your-org/araBertv2/issues",
        "Source": "https://github.com/your-org/araBertv2",
        "NER Documentation": "https://ner-cli.readthedocs.io/",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=optional_requirements,
    entry_points={
        "console_scripts": [
            # NER CLI commands
            "ner=ner_cli:main",
        ],
    },
    scripts=["ner_cli.py"],
    include_package_data=True,
    package_data={
        "ner": [
            "configs/*.json",
            "configs/**/*.json",
            "templates/*.json",
            "templates/**/*.json",
        ],
    },
    data_files=[
        # NER data files
        ("data/ner/configs/templates", ["data/ner/configs/templates/default.json", "data/ner/configs/templates/address_ner.json"]),
        ("data/ner/configs/countries", ["data/ner/configs/countries/uae.json"]),
    ],
    zip_safe=False,
    keywords=[
        # Core technologies
        "nlp",
        "arabic",
        "bert",
        "transformer",
        "arabertv2",
        # NER specific
        "ner",
        "named-entity-recognition",
        "entity-extraction",
        # Applications
        "address-parsing",
        "text-processing",
        "multilingual",
        # Tools
        "machine-learning",
        "cli",
        "command-line",
        "natural-language-processing",
    ],
    license="MIT",
    platforms=["any"],
    cmdclass={},
)

# Post-installation setup
if __name__ == "__main__":
    # This section runs when setup.py is executed directly
    import subprocess
    import sys
    
    def post_install():
        """Post-installation setup tasks"""
        print("\n" + "="*60)
        print("AraBERTv2 Toolkit Installation Complete!")
        print("="*60)
        
        # Create necessary directories for both modules
        directories = [
            # NER directories
            "data/ner",
            "data/ner/models",
            "data/ner/datasets",
            "data/ner/outputs",
            "data/ner/logs",
            "data/ner/checkpoints",
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        
        print("\n  NER (Named Entity Recognition):")
        print("    ner --help                     # Show NER help")
        print("    ner train --country uae        # Train NER model")
        print("    ner predict --text 'text'      # Predict entities")
        
        print("\nFor UAE address parsing:")
        print("  # NER training:")
        print("  ner train --country uae --data-path your_data.json")
        
        print("\nDocumentation:")
        print("  NER:  https://ner-cli.readthedocs.io/")
        print("="*60 + "\n")
    
    # Check if this is an installation (not just importing)
    if "install" in sys.argv or "develop" in sys.argv:
        try:
            post_install()
        except Exception as e:
            print(f"Warning: Post-installation setup failed: {e}")
            print("You may need to create directories manually.")