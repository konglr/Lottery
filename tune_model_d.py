import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
import json
import itertools
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

# Ensure funcs and models can be imported
sys.path.append(os.getcwd())
from config import LOTTERY_CONFIG
from models.method_d import train_predict_lstm

# --- Configuration ---
LOTTERY_NAME = "双色球"
BACKTEST_LEN = 15  # Further reduced to save time
N_ITER = 10 # 10 random samples per pool

# Suppress verbose logging
logging.basicConfig(level=logging.WARNING)

def load_data(conf):
    try:
        df = pd.read_csv(conf['data_file'])
        if 'issue' in df.columns:
            df = df.rename(columns={'issue': '期号'})
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce').fillna(0).astype(int).astype(str)
        if int(df['期号'].iloc[0]) > int(df['期号'].iloc[-1]):
            df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def evaluate_model(df, model_config, lottery_config, pool_type='red'):
    """
    Backtests Model D.
    Metric: Average rank of the actual drawn balls.
    """
    total_rank_sum = 0
    total_hits = 0
    
    start_idx = len(df) - BACKTEST_LEN
    
    for i in range(start_idx, len(df)):
        train_df = df.iloc[:i]
        
        if pool_type == 'red':
            conf_local = lottery_config.copy()
            conf_local['separate_pool'] = False
            conf_local['total_numbers'] = 33
            conf_local['num_list'] = list(range(1, 34))
            conf_local['red_num_list'] = conf_local['num_list']
            conf_local['red_cols'] = [f"{lottery_config['red_col_prefix']}{k}" for k in range(1, 7)]
            conf_local['blue_cols'] = []
            actual_balls = df.iloc[i][conf_local['red_cols']].values.astype(int)
        else:
            conf_local = lottery_config.copy()
            conf_local['separate_pool'] = False
            conf_local['total_numbers'] = 16
            conf_local['num_list'] = list(range(1, 17))
            conf_local['red_num_list'] = conf_local['num_list']
            conf_local['red_cols'] = ['蓝球']
            conf_local['blue_cols'] = []
            actual_balls = df.iloc[i][conf_local['red_cols']].values.astype(int)

        probs = train_predict_lstm(train_df, model_config, conf_local)
        
        num_list = conf_local['num_list']
        num_prob = {num: probs[idx] for idx, num in enumerate(num_list)}
        sorted_nums = sorted(num_prob.items(), key=lambda x: x[1], reverse=True)
        rank_map = {num: rank + 1 for rank, (num, prob) in enumerate(sorted_nums)}
        
        current_ranks = [rank_map[num] for num in actual_balls if num in rank_map]
        total_rank_sum += sum(current_ranks)
        total_hits += len(current_ranks)
        
    avg_rank = total_rank_sum / total_hits if total_hits > 0 else (len(conf_local['num_list'])/2)
    return avg_rank

def run_tuning():
    print(f"🚀 开始为 [{LOTTERY_NAME}] 的模型 D (LSTM) 进行随机搜索调优...")
    
    conf = LOTTERY_CONFIG[LOTTERY_NAME].copy()
    conf['name'] = LOTTERY_NAME

    df = load_data(conf)
    if df is None: return

    param_grid = {
        'embedding_dim': [16, 32],
        'hidden_dim': [32, 64],
        'num_layers': [1, 2],
        'dropout': [0.2, 0.3],
        'lr': [0.001, 0.005],
        'epochs': [50, 80]
    }

    best_results = {}

    for pool in ['red', 'blue']:
        print(f"\n--- 正在调优 {pool.upper()} 球参数 ( samples={N_ITER} )...")
        param_list = list(ParameterSampler(param_grid, n_iter=N_ITER, random_state=42))
        
        best_score = np.inf
        best_params = None
        
        for params in tqdm(param_list, desc=f"Scanning {pool.upper()}"):
            score = evaluate_model(df, params, conf, pool)
            if score < best_score:
                best_score = score
                best_params = params
        
        print(f"✅ {pool.upper()} 球最佳 Avg Rank: {best_score:.4f}")
        print(f"最佳参数: {best_params}")
        best_results[pool] = best_params

    print("\n" + "="*50)
    print("🎉 调优完成！建议的 MODEL_CONFIG (SSQ - Model D):")
    print("="*50)
    print(json.dumps(best_results, indent=4))

if __name__ == "__main__":
    run_tuning()
