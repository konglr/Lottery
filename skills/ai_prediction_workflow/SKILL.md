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
