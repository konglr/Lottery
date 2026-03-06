---
name: AI Prediction Workflow
description: Logic and process for AI-driven lottery analysis and prediction.
---

# AI Prediction Workflow

This skill documents the technical flow and logic for AI model integration within the Lottery project.

## 1. System Components
- **UI Interaction**: `app.py` -> `render_ai()`
- **Logic Helper**: `funcs/ai_helper.py` -> `prepare_lottery_data_text()`, `generate_ai_prediction()`, `_build_prediction_prompt()`
- **API Clients**: `google-genai` (Gemini), `openai` (NVIDIA, MiniMax, DashScope)

## 2. Logic Flow

### Step 1: Data Preparation
The user selects a lottery and the number of periods (N) to analyze. `app.py` extracts the latest N rows from the historical CSV data.

### Step 2: Data Textualization
`funcs/ai_helper.py` converts the raw dataframe rows into a human-readable text format suitable for Large Language Models (LLMs). This includes draw numbers, red balls, and blue balls.

### Step 3: Prompt Engineering
The system constructs a detailed "Prompt" (instruction book) for the AI. Key components:
- **Expert Role**: Assigns the AI as a professional lottery analyst.
- **Constraints**: Enforces lottery-specific rules (e.g., SSQ ranges 1-33).
- **Morphology Focus**: Explicitly instructs the AI to consider "Morphological distributions" such as consecutive numbers (连号), jump numbers (跳号), and repeated numbers (重号).

### Step 4: API Dispatch
The system routes the prompt to the selected AI provider:
- **Gemini**: Direct integration via `Vertex AI` or `Gemini API`.
- **OpenAI Compatible Providers**: Routes via `OpenAI` client using provider-specific base URLs and API keys.

### Step 5: Result Rendering
The AI returns a Markdown report containing:
- Deep analysis of historical patterns.
- 5 groups of recommended numbers (for standard lotteries) or 20 numbers (for KL8).
- Rationale for the selections.

## 3. Key Configuration
API keys must be stored in the `.Renviron` file or system environment variables:
- `GEMINI_API_KEY`
- `NV_API_KEY` (NVIDIA)
- `MINIMAX_API_KEY`
- `ALIYUNCS_API_KEY` (DashScope)

---

## 4. Automated Batch Archival (`ai_batch_predict.py`)

The project includes a standalone script for automated, multi-model batch predictions and structured archival.

### Execution Flow
1. **Configuration**: `BATCH_CONFIG` defines the target lotteries and the list of model tuples `(Brand, ModelName)` to run.
2. **Data Synchronization**:
    - Loads historical draw data from `data/`.
    - Normalizes variety in column names (e.g., `issue`, `period` standardized to `期号`).
    - Calculates the `target_period` (Latest + 1).
3. **AI Logic Invoke**:
    - Prepares the textual prompt with the last $N$ periods.
    - Sequentially calls `generate_ai_prediction` for each configured model.
4. **Structured Parsing**:
    - Calls `parse_ai_recommendations` to convert the AI's Markdown report into standardized JSON.
5. **Archival**:
    - Appends results to `data/ai_predictions_history.csv`.
    - Saves metadata: `timestamp`, `lottery`, `model`, `target_period`, `input_periods`, `recommendations` (JSON), and `raw_response` (Original Text).

### Usage
Run the batch engine manually via terminal to refresh archival data:
```bash
PYTHONPATH=. python3 ai_batch_predict.py
```
