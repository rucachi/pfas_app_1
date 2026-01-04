# ONFRA PFAS

Non-Target Screening Analysis Platform for PFAS (Per- and Polyfluoroalkyl Substances)

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## Overview

ONFRA PFAS is a desktop application for analyzing LC-MS/MS data to detect and prioritize PFAS compounds. The application processes mzML files through a three-stage pipeline:

1. **Feature Finding**: Detect chromatographic features using OpenMS FeatureFinderMetabo
2. **PFAS Prioritization**: Score and rank features using multiple evidence types
3. **Visualization**: Interactive EIC, spectrum, and correlation plots

## Features

- **GPU Acceleration**: Optional CuPy-based GPU acceleration for large datasets
- **Memory Efficient**: Uses OnDiscMSExperiment for large mzML files
- **Comprehensive Scoring**:
  - Kendrick Mass Defect (KMD) homologous series detection
  - MD/C (mass defect per carbon) filtering
  - Diagnostic fragment matching
  - Neutral loss (Δm) rule matching
  - Suspect list screening
- **Checkpoint System**: Save/resume analysis at each stage
- **Modern GUI**: PySide6-based interface with dark theme

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/onfra-pfas.git
cd onfra-pfas

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Optional: Install GPU support
pip install -e ".[gpu]"
```

### Requirements

- Python 3.10+
- pyOpenMS 3.0+
- PySide6 6.5+
- NumPy, Pandas, PyArrow
- Optional: CuPy (for GPU acceleration)

## Usage

### GUI Application

```bash
# Run the GUI
python -m onfra_pfas.app.main
```

Or after installation:

```bash
onfra-pfas-gui
```

### Programmatic Usage

```python
from pathlib import Path
from onfra_pfas.core.config import PipelineConfig, get_default_config
from onfra_pfas.core.featurefinding import run_feature_finder_metabo
from onfra_pfas.core.pfas_prioritization import run_prioritization

# Load configuration
config = get_default_config()

# Run feature finding
result = run_feature_finder_metabo(
    Path("sample.mzML"),
    config.feature_finder,
)

# Run prioritization
prioritized = run_prioritization(
    result.features_df,
    ms2_data=None,  # Optional MS2 data
    config=config.prioritization,
)

# Export results
prioritized.to_excel("results.xlsx", index=False)
```

## Project Structure

```
PFAS_APP/
├── pyproject.toml           # Project configuration
├── src/onfra_pfas/
│   ├── core/
│   │   ├── config.py        # Pydantic configuration
│   │   ├── backend.py       # NumPy/CuPy selection
│   │   ├── io_mzml.py       # mzML loading
│   │   ├── featurefinding.py
│   │   ├── pfas_prioritization.py
│   │   ├── visualization.py
│   │   ├── checkpoints.py
│   │   └── ...
│   └── app/
│       ├── main.py          # GUI entry point
│       ├── splash.py
│       └── tabs/
├── assets/
│   └── onfra_logo.png
├── scripts/
│   └── build_pyinstaller.py
└── tests/
```

## GPU Acceleration

The application supports GPU acceleration for compute-intensive operations:

- EIC extraction and smoothing
- Correlation analysis
- Batch scoring

### GPU Modes

- `AUTO`: Use GPU if data size exceeds thresholds
- `FORCE_GPU`: Always use GPU (fails if unavailable)
- `FORCE_CPU`: Never use GPU

```python
from onfra_pfas.core.config import GPUMode

config = get_default_config()
config.gpu.mode = GPUMode.AUTO
config.gpu.feature_count_threshold = 5000
```

### Handling VRAM OOM

The application automatically handles CUDA out-of-memory errors by:
1. Reducing chunk size
2. Retrying up to 3 times
3. Falling back to CPU if all retries fail

## Building Standalone Executable

```bash
# Check dependencies
python scripts/build_pyinstaller.py --check-deps

# Build onedir (recommended)
python scripts/build_pyinstaller.py --mode onedir

# Build onefile
python scripts/build_pyinstaller.py --mode onefile
```

The built application will be in `dist/ONFRA_PFAS/`.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_prioritization_math.py -v

# Run with coverage
pytest tests/ --cov=onfra_pfas --cov-report=html
```

## Configuration

Configuration is managed through Pydantic models. Key settings:

### Feature Finding
- `mass_trace_mz_tolerance`: m/z tolerance for mass trace detection (default: 0.005 Da)
- `noise_threshold_int`: Intensity threshold (default: 1000)
- `chrom_peak_snr`: Signal-to-noise ratio (default: 3.0)

### Prioritization
- `kmd.repeat_units`: ["CF2", "CF2O", "C2F4", "C2F4O"]
- `suspect_screening.ppm_tolerance`: 5.0 ppm
- `scoring.weights`: Evidence weight dictionary

### Presets
```python
from onfra_pfas.core.config import (
    get_default_config,
    get_high_sensitivity_config,
    get_high_specificity_config,
)

# For maximum detection (more false positives)
config = get_high_sensitivity_config()

# For high confidence results (fewer hits)
config = get_high_specificity_config()
```

## Output Files

Each analysis run creates a structured output folder:

```
runs/<run_id>/
├── input/              # Links to input files
├── checkpoints/
│   ├── 01_featurefinding.parquet
│   ├── 01_featurefinding_meta.json
│   ├── 02_prioritization.parquet
│   └── ...
├── reports/
│   ├── summary.html
│   └── plots.html
└── logs/
    ├── run.log
    └── config_snapshot.json
```

## Developer Information

**개발**: 김태형  
**연락처**: 010-9411-7143

## License

MIT License - See LICENSE file for details.
