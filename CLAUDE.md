# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AraBERTv2 is a comprehensive Arabic NLP toolkit focused on Named Entity Recognition (NER), particularly optimized for UAE address parsing and multi-country Arabic text processing. The project supports LoRA training, detailed evaluation reports, and comprehensive CLI tools.

## Project Structure

The codebase follows a modular architecture:

- **Entry Points**: 
  - `ner_cli.py` - Main CLI entry point for all NER operations
  - `python ner_cli.py` or `ner` command for all operations

- **Core Modules** (`src/`):
  - `src/ner/` - Main NER module with training, evaluation, and prediction
  - `src/ner/cli/` - CLI command implementations
  - `src/ner/config/` - Configuration management and validation
  - `src/ner/data/` - Data processing, loading, and conversion
  - `src/ner/models/` - Model management and wrappers
  - `src/ner/training/` - Training engine with LoRA support
  - `src/ner/evaluation/` - Evaluation metrics and report generation
  - `src/ner/preprocess/` - Text preprocessing pipelines

- **Data Organization** (`data/ner/`):
  - `configs/countries/` - Country-specific configurations (e.g., `uae.json`)
  - `configs/templates/` - Configuration templates
  - `models/` - Trained models and checkpoints
  - `datasets/` - Training and evaluation data
  - `logs/` - Training and evaluation logs

## Common Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Testing
```bash
# Run tests (pytest is available)
pytest tests/

# Run specific test file
pytest tests/ner/test_processor_integration.py

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/ tests/

# Check code style  
flake8 src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### NER Operations

#### Training
```bash
# Train UAE model with LoRA
python ner_cli.py train --country uae --data-path ./data/train.json --lora

# Train with custom LoRA parameters
python ner_cli.py train --country uae --lora --lora-r 16 --lora-alpha 32

# Resume training from checkpoint
python ner_cli.py train --country uae --resume ./models/checkpoint-1000
```

#### Evaluation
```bash
# Standard evaluation
python ner_cli.py evaluate --model-path ./models/uae_model --data-path ./data/test.json --country uae

# Detailed Excel report with entity focus
python ner_cli.py evaluate --model-path ./models/uae_model --data-path ./data/test.json --country uae --detailed-report --entity "country,city"

# Evaluation using predict() method (serving-like)
python ner_cli.py evaluate-predict --model-path ./models/uae_model --data-path ./data/test.json --country uae
```

#### Prediction
```bash
# Interactive prediction mode
python ner_cli.py predict

# Single text prediction
python ner_cli.py predict --model-path ./models/uae_model --text "123 Sheikh Zayed Road, Dubai"

# Batch prediction from file
python ner_cli.py predict --model-path ./models/uae_model --file ./texts.txt
```

#### Data Processing
```bash
# Validate data format
python ner_cli.py data validate --data-path ./data/train.json

# Convert CSV to JSONL with validation
python ner_cli.py data convert --input-path data.csv --output-path data.jsonl --country uae

# Only validate CSV without conversion
python ner_cli.py data convert --input-path data.csv --country uae --only-validate

# Split dataset
python ner_cli.py data split --data-path ./data/full.json --output-dir ./splits
```

#### Configuration Management
```bash
# List available configs
python ner_cli.py config list

# Show specific config
python ner_cli.py config show uae

# Create new country config
python ner_cli.py config create --country egypt --template address_ner

# Validate config
python ner_cli.py config validate uae
```

## Architecture Highlights

### Configuration System
- JSON-based country-specific configurations in `data/ner/configs/countries/`
- Template system for creating new country configurations
- Automatic validation and schema checking

### LoRA Training Support
- Built-in LoRA (Low-Rank Adaptation) training for efficient fine-tuning
- Configurable LoRA parameters (r, alpha, dropout, target modules)
- Support for incremental training with base adapters

### Evaluation and Reporting
- Token-level and entity-level metrics
- Excel reports with detailed entity analysis and error highlighting
- Entity-focused evaluation with problematic sample extraction
- Support for both standard evaluation and predict()-style evaluation

### Data Processing Pipeline
- Multi-format support (JSON, JSONL, CSV, CoNLL)
- Country-specific preprocessing with Arabic text normalization
- Quality validation and anomaly detection
- Automatic data splitting capabilities

### UAE Address Parsing
- Specialized for 23 UAE address entity types (EMIRATE, CITY, AREA, STREET, etc.)
- Optimized for Gulf Arabic dialect
- Comprehensive address component extraction

## Key Features

- **Multi-country Support**: Extensible configuration system for different Arabic countries/dialects
- **Advanced Training**: LoRA support for efficient model adaptation
- **Comprehensive Evaluation**: Detailed reporting with Excel output and error analysis
- **Data Quality**: Built-in validation and conversion tools
- **CLI Interface**: Complete command-line interface for all operations
- **Preprocessing**: Country-specific text preprocessing pipelines

## Development Notes

- All CLI commands support verbose logging with `--verbose` flag
- Log files are automatically organized by country and timestamp
- Model checkpoints are saved during training for recovery
- Configuration files use JSON format with comprehensive validation
- The system supports both CPU and GPU training/inference