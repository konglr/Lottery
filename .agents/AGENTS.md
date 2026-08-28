# Workspace Customizations

## Lottery Project Prediction Data Writing Policy (Lottery 项目预测数据写入策略)

1. **data/ai_predictions_history.csv (Main history, Streamlit rendering)**
   - **Writer**: `ai_batch_predict.py:archive_prediction()`
   - **Applicable to**: External LLM (Gemini/Qwen/Kimi/DeepSeek/Llama/MiniMax API)
   - **Format**: 8 columns (`timestamp`, `lottery`, `model`, `target_period`, `input_periods`, `recommendations`, `raw_response`, `_备注_source`)

2. **data/backtest/*.json (Lucky local algorithms)**
   - **Writer**: Lucky (MiniMax-M3) / Local execution
   - **Applicable to**: FSW / V3-V13 / adaptive / morphological constraint, etc. (Local computing solutions)
   - **Format**: One independent JSON per issue, including meta + multiple configurations + backtest data
   - **Filename Convention**: `<lottery_code>_<period>_<variant>.json`
     - *Example*: `fsw_26083_prediction.json`, `2026196_predictions_v10.json`
