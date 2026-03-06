import os
import pandas as pd
import json
from datetime import datetime
import logging
from funcs.ai_helper import load_renviron, get_brand_models, prepare_lottery_data_text, generate_ai_prediction, parse_ai_recommendations
from config import LOTTERY_CONFIG

# --- Configuration Section ---
# Set the lotteries and models you want to run in batch
BATCH_CONFIG = {
    "lotteries": [ "排列五", "超级大乐透"],  #"双色球", "七星彩", , "排列三", "排列五" "超级大乐透", "快乐8", "福彩3D",“七乐彩”
    "models": [
        ("DashScope", "qwen3.5-plus"),
        ("DashScope", "kimi-k2.5"),
        ("MiniMax", "MiniMax-M2.5"),
        ("DashScope", "glm-5"),
        ("Gemini", "models/gemini-3.1-flash-lite-preview")
    ],
    "input_periods": 30  # How many historical periods to provide as context
}

# --- Setup ---
load_renviron()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def archive_prediction(lottery_name, model_name, target_period, input_periods, raw_response, structured_data):
    """
    Saves the prediction results to a historical CSV file.
    """
    csv_path = f"data/ai_predictions_history.csv"
    
    # Prepare record
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lottery": lottery_name,
        "model": model_name,
        "target_period": target_period,
        "input_periods": input_periods,
        "recommendations": json.dumps(structured_data, ensure_ascii=False),
        "raw_response": raw_response.replace("\n", "\\n") # Flatten for CSV
    }
    
    df_new = pd.DataFrame([record])
    
    # Write to CSV
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
    else:
        df_new.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    logging.info(f"✅ [Archived] {lottery_name} ({model_name}) prediction saved to {csv_path}")

def run_batch():
    logging.info("🚀 Starting AI Batch Prediction Engine...")
    
    # Map brands to their env keys
    env_keys = {
        "Gemini": "GEMINI_API_KEY",
        "NVIDIA": "NV_API_KEY",
        "MiniMax": "MINIMAX_API_KEY",
        "DashScope": "ALIYUNCS_API_KEY"
    }

    for lottery_name in BATCH_CONFIG["lotteries"]:
        config = LOTTERY_CONFIG.get(lottery_name)
        if not config:
            logging.warning(f"Lottery {lottery_name} not found in config. Skipping.")
            continue
            
        # Load data
        data_file = config["data_file"]
        if not os.path.exists(data_file):
            logging.warning(f"Data file {data_file} not found. Skipping.")
            continue
            
        df_full = pd.read_csv(data_file)
        if df_full.empty:
            continue
            
        # Rename columns to standard '期号'
        column_mapping = {}
        if 'issue' in df_full.columns: column_mapping['issue'] = '期号'
        if 'period' in df_full.columns: column_mapping['period'] = '期号'
        if column_mapping: 
            df_full = df_full.rename(columns=column_mapping)
            
        if '期号' not in df_full.columns:
            logging.warning(f"Column '期号' (or 'issue'/'period') not found in {data_file}. Skipping.")
            continue

        # Sort by period descending
        df_full['期号'] = pd.to_numeric(df_full['期号'], errors='coerce').fillna(0).astype(int)
        df_full = df_full.sort_values('期号', ascending=False)
        last_period = str(df_full.iloc[0]['期号'])
        target_period = str(int(last_period) + 1)
        
        # Take N periods for context
        df_input = df_full.head(BATCH_CONFIG["input_periods"])
        data_text = prepare_lottery_data_text(df_input, config)
        
        for brand, model in BATCH_CONFIG["models"]:
            api_key = os.getenv(env_keys.get(brand, ""))
            if not api_key:
                logging.error(f"API Key for {brand} not found. Skipping {model}.")
                continue
                
            logging.info(f"🔮 Predicting {lottery_name} (Target: {target_period}) using {model}...")
            
            try:
                # 1. Generate Prediction
                raw_prediction = generate_ai_prediction(brand, model, api_key, data_text, config)
                
                # 2. Parse Results
                structured = parse_ai_recommendations(raw_prediction, config)
                
                # 3. Archive
                archive_prediction(lottery_name, model, target_period, BATCH_CONFIG["input_periods"], raw_prediction, structured)
                
            except Exception as e:
                logging.error(f"Failed to predict {lottery_name} with {model}: {e}")

if __name__ == "__main__":
    run_batch()
