import os
import pandas as pd
import json
from datetime import datetime
import logging
from funcs.ai_helper import load_renviron, get_brand_models, prepare_lottery_data_text, generate_ai_prediction, parse_ai_recommendations
from config import LOTTERY_CONFIG
from ai_batch_predict import archive_prediction, BATCH_CONFIG

# --- Test Configuration ---
TEST_CONFIG = {
    "lotteries": ["双色球"],
    "models": [
        ("DashScope", "qwen3.5-plus")
    ],
    "input_periods": 30
}

load_renviron()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    logging.info("🧪 Running AI Archival System Test...")
    
    lottery_name = TEST_CONFIG["lotteries"][0]
    brand, model = TEST_CONFIG["models"][0]
    config = LOTTERY_CONFIG[lottery_name]
    
    # Load data
    df_full = pd.read_csv(config["data_file"])
    
    # Rename columns to standard '期号'
    column_mapping = {}
    if 'issue' in df_full.columns: column_mapping['issue'] = '期号'
    if 'period' in df_full.columns: column_mapping['period'] = '期号'
    if column_mapping: 
        df_full = df_full.rename(columns=column_mapping)
        
    df_full['期号'] = pd.to_numeric(df_full['期号'], errors='coerce').fillna(0).astype(int)
    df_full = df_full.sort_values('期号', ascending=False)
    target_period = str(int(df_full.iloc[0]['期号']) + 1)
    
    data_text = prepare_lottery_data_text(df_full.head(TEST_CONFIG["input_periods"]), config)
    api_key = os.getenv("ALIYUNCS_API_KEY")
    
    if not api_key:
        print("Error: No GEMINI_API_KEY found.")
        return

    print(f"Calling {model} for {lottery_name}...")
    try:
        raw = generate_ai_prediction(brand, model, api_key, data_text, config)
        print("--- RAW RESPONSE START ---")
        print(raw[:500] + "...")
        print("--- RAW RESPONSE END ---")
        
        structured = parse_ai_recommendations(raw, config)
        print("\n--- STRUCTURED DATA ---")
        print(json.dumps(structured, indent=2, ensure_ascii=False))
        
        archive_prediction(lottery_name, model, target_period, TEST_CONFIG["input_periods"], raw, structured)
        print("\n✅ Test complete. Check data/ai_predictions_history.csv")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    run_test()
