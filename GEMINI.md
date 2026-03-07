# Lottery Analysis & Prediction Project (Streamlit App)

## Project Overview
This project is a comprehensive Streamlit application designed for collecting, analyzing, and predicting results for various Chinese lotteries (SSQ, DLT, KL8, etc.). The application is intended to be deployed via GitHub to Streamlit Cloud.

### Core Goals
- **Data Collection**: Automated scraping and updating of historical lottery data.
- **Statistical Analysis**: Providing visualizations and metrics (cold/hot numbers, trends, ratios).
- **AI Prediction**: Leveraging various AI models (Gemini, NVIDIA, MiniMax, DashScope) for data-driven predictions.
- **ML Backtesting**: Evaluating performance of Statistical, RF, XGB, and LSTM models.

---

## Technical Stack & Environment
- **Language**: Python 3.12
- **Framework**: [Streamlit](https://streamlit.io/) (Entry point: `app.py`)
- **Data Handling**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualization**: [Altair](https://altair-viz.github.io/), [Matplotlib](https://matplotlib.org/)
- **AI/ML**: `google-genai`, `openai`, `scikit-learn`, `xgboost`, `pytorch`.

---

## Directory Structure (Modular Architecture)
The project follows a modular design to ensure maintainability and scalability:

- `app.py`: The main entry point. Handles navigation, global CSS, and high-level routing.
- `config.py`: Central configuration for all supported lottery types (ranges, rules, pools).
- `data/`: CSV storage for historical draw data and backtest results.
- `views/`: UI Rendering Modules (View Layer):
    - `components.py`: Shared UI elements (sidebar, metrics, data loading).
    - `trend_analysis.py`: Charts for frequency, odd/even, consecutive, etc.
    - `ai_assistant.py`: AI prediction interface and history comparison.
    - `morphology.py`: Morphological filtering and scoring UI.
    - `backtest.py`: Detailed backtesting reports and model comparisons.
- `funcs/`: Core Logic Modules (Model/Controller Layer):
    - `ai_helper.py`: Unified interface for AI model predictions.
    - `functions.py`: Statistical analysis and data processing helpers.
    - `ball_filter.py`: Morphological scoring algorithms.
- `skills/`: Project-specific knowledge base and specialized agent skills.
- `request_data_*.py`: Scripts for automated data synchronization and validation.
- `multi_model.py`: Local Machine Learning prediction engine.

---

## Development & Deployment Workflow

### 1. Local Development
- **Environment**: Use the pre-configured virtual environment in `./venv_312`.
- **Running the App**: `streamlit run app.py` (ensure you use the `venv_312` binary).
- **Data Sync**: Run `python request_data_update.py` to fetch the latest draw results.

### 2. Modeling Logic (Dual-Pool vs Single-Pool)
- **Separate Pool (Separate Pool = True)**: For SSQ, DLT, XQXC. Red and Blue balls are treated as independent distributions.
- **Single Pool (Separate Pool = False)**: For QLC, KL8. All balls (including special ones) are treated as a single interconnected sequence.

### 3. Deployment (Streamlit Cloud)
- **Source Control**: Push changes to the GitHub repository.
- **Secrets Management**: Configure Streamlit Cloud "Secrets" to match the variables in `.Renviron`.

---

## Specialized Skills
Refer to the `skills/` directory for detailed documentation on:
- [AI API Configurations](./skills/ai_api_configs/SKILL.md)
- [AI Prediction Workflow](./skills/ai_prediction_workflow/SKILL.md)
- [Lottery Rules](./skills/lottery_rules/SKILL.md)
