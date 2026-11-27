---
description: Project Execution Environment Instructions
---

# Project Execution Environment

This project utilizes a specific Python virtual environment.

## Environment Details
- **Path**: `./venv_315`
- **Binary**: `./venv_315/bin/python`
- **Python Version**: 3.15.2 (Homebrew `python@3.15`)

## Usage
Run scripts using the absolute or relative path to the venv binary:
`./venv_315/bin/python [script_name].py`

## Rebuilding the environment
The venv is not committed (see `.gitignore`). To recreate it from scratch:

```bash
/opt/homebrew/bin/python3.15 -m venv venv_315
./venv_315/bin/pip install -r requirements.txt
./venv_315/bin/pip install scikit-learn torch xgboost lightgbm hmmlearn optuna pygad tqdm requests plotly pytest
# catboost has no cp315 wheel on PyPI — it must be built from source.
# --no-widget skips the Jupyter widget build (requires yarn/node, unrelated to this project).
./venv_315/bin/pip install --no-binary catboost --config-settings="--build-option=--no-widget" catboost
```

> Note: `catboost` is only needed for Method F (`models/method_f.py`). The Streamlit
> app (`app.py`) does not import `models/` at all, so the app works without it.
