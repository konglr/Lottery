# Lottery Analysis & Prediction Project (Streamlit App)

## Project Overview
This project is a comprehensive Streamlit application designed for collecting, analyzing, and predicting results for various Chinese lotteries (SSQ, DLT, KL8, etc.). The application is intended to be deployed via GitHub to Streamlit Cloud.

### Core Goals
- **Data Collection**: Automated scraping and updating of historical lottery data.
- **Statistical Analysis**: Providing visualizations and metrics (cold/hot numbers, trends, ratios).
- **AI Prediction**: Leveraging various AI models (Gemini, NVIDIA, MiniMax, DashScope) for data-driven predictions.
- **Backtesting**: Evaluating model performance against historical data.

---

## Technical Stack & Environment
- **Language**: Python 3.12
- **Framework**: [Streamlit](https://streamlit.io/) (Entry point: `app.py`)
- **Data Handling**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualization**: [Altair](https://altair-viz.github.io/), [Matplotlib](https://matplotlib.org/)
- **AI Integration**: `google-genai`, `openai` (for compatible APIs)
- **Environment Management**: `.Renviron` for API keys and configuration.

---

## Directory Structure & Key Files
- `app.py`: The main Streamlit application entry point.
- `config.py`: Central configuration for all supported lottery types (ranges, rules, metrics).
- `data/`: CSV storage for historical draw data and backtest results.
- `funcs/`: Core logic modules:
    - `ai_helper.py`: Unified interface for AI model predictions.
    - `functions.py`: Statistical analysis and data processing helpers.
- `skills/`: Project-specific knowledge base and specialized agent skills.
- `request_data_*.py`: Scripts for automated data synchronization and validation.
- `requirements.txt`: Dependencies for local development and Streamlit Cloud deployment.

---

## Development & Deployment Workflow

### 1. Local Development
- **Environment**: Use the pre-configured virtual environment in `./venv_312`.
- **Running the App**: `streamlit run app.py` (ensure you use the `venv_312` binary).
- **Data Sync**: Run `python request_data_update.py` to fetch the latest draw results.

### 2. Configuration & AI APIs
- **API Keys**: Stored in `.Renviron`.
- **Skills**: Refer to `skills/ai_api_configs/SKILL.md` for detailed API specifications and model support.
- **Lottery Rules**: Refer to `skills/lottery_rules/SKILL.md` for individual lottery mechanics.

### 3. Deployment (Streamlit Cloud)
- **Source Control**: Push changes to the GitHub repository.
- **Requirements**: Ensure all dependencies (including `streamlit` and `altair`) are listed in `requirements.txt`.
- **Secrets Management**: Configure Streamlit Cloud "Secrets" to match the variables in `.Renviron`.

---

## Specialized Skills
This project utilizes sub-skills for specific domains:
- [AI API Configurations](./skills/ai_api_configs/SKILL.md): Protocols and models for Gemini, NVIDIA NIM, MiniMax, and DashScope.
- [Lottery Rules](./skills/lottery_rules/SKILL.md): Official rules and prize structures for SSQ, KL8, DLT, etc.
- [Execution Environment](./skills/environment.md): Details on the local Python environment.

---

## Coding Standards
- **Chinese UI/UX**: All user-facing text in the Streamlit app should be in Simplified Chinese.
- **Modular Design**: Keep visualization logic in `app.py` or dedicated helpers, and data processing in `funcs/`.
- **Error Handling**: Use the logging configuration defined in `app.py`.
- **Environment Isolation**: Always prioritize loading configuration from `.Renviron` or system environment variables.
