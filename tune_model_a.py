import os
import sys
import pandas as pd
import numpy as np
import optuna
import itertools
import logging
from tqdm import tqdm
import json

# Ensure funcs and models can be imported
sys.path.append(os.getcwd())
from models.method_a import predict_similarity
from config import LOTTERY_CONFIG

# --- Configuration ---
LOTTERY_NAME = "双色球"
BACKTEST_PERIODS = 50  # Use last 50 periods for evaluation of one param set
OPTUNA_TRIALS_PER_GRID = 30 # Number of trials for weight optimization for each grid point

# Suppress Optuna's INFO messages
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- Helper Functions (adapted from multi_model.py) ---

def load_data(conf):
    """Loads and prepares lottery data."""
    try:
        df = pd.read_csv(conf['data_file'])
        if 'issue' in df.columns:
            df = df.rename(columns={'issue': '期号'})
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce').fillna(0).astype(int).astype(str)
        if int(df['期号'].iloc[0]) > int(df['期号'].iloc[-1]):
            df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def get_omission_matrix(df, red_num_list, blue_num_list, red_cols, blue_cols):
    """Calculates omission matrix."""
    n_rows = len(df)
    total_red = len(red_num_list)
    total_blue = len(blue_num_list) if blue_num_list else 0
    
    omission_matrix = np.zeros((n_rows, total_red + total_blue), dtype=int)
    current_red_omiss = np.zeros(total_red, dtype=int)
    current_blue_omiss = np.zeros(total_blue, dtype=int) if total_blue > 0 else np.array([])
    
    omission_cols = [f'Omission_{i}' for i in red_num_list]
    if total_blue > 0:
        omission_cols += [f'Omission_{i}' for i in blue_num_list]

    for i in range(n_rows):
        omission_matrix[i, :total_red] = current_red_omiss
        if total_blue > 0:
            omission_matrix[i, total_red:] = current_blue_omiss
        
        row_red = set(df.loc[i, red_cols].values)
        row_blue = set(df.loc[i, blue_cols].values) if blue_cols else set()
        
        for idx, num in enumerate(red_num_list):
            if num in row_red: current_red_omiss[idx] = 0
            else: current_red_omiss[idx] += 1
        if total_blue > 0:
            for idx, num in enumerate(blue_num_list):
                if num in row_blue: current_blue_omiss[idx] = 0
                else: current_blue_omiss[idx] += 1
        
    return pd.DataFrame(omission_matrix, columns=omission_cols)


def evaluate_performance(params, full_df, conf, pool_type='red'):
    """Performs backtesting for a given set of parameters."""
    total_hits = 0
    
    red_cols = [f"{conf['red_col_prefix']}{i}" for i in range(1, conf['red_count'] + 1)]
    blue_cols = [f"{conf['blue_col_name']}{i}" for i in range(1, conf['blue_count'] + 1)] if conf['blue_count'] > 1 else [conf['blue_col_name']]

    for i in range(len(full_df) - BACKTEST_PERIODS, len(full_df)):
        train_df = full_df.iloc[:i]
        actual_row = full_df.iloc[i]

        if pool_type == 'red':
            lottery_conf = {
                'window_size': conf['window_size'], 'separate_pool': False,
                'total_numbers': conf['red_range'][1] - conf['red_range'][0] + 1, 'num_list': list(range(conf['red_range'][0], conf['red_range'][1] + 1)),
                'red_cols': red_cols, 'blue_cols': [], 'red_range': conf['red_range']
            }
            actual_balls = set(actual_row[red_cols].values)
            top_n = conf['red_count']
        else: # blue
            lottery_conf = {
                'window_size': conf['window_size'], 'separate_pool': False,
                'total_numbers': conf['blue_range'][1] - conf['blue_range'][0] + 1, 'num_list': list(range(conf['blue_range'][0], conf['blue_range'][1] + 1)),
                'red_cols': blue_cols, 'blue_cols': [], 'red_range': conf['blue_range']
            }
            actual_balls = set(actual_row[blue_cols].values)
            top_n = conf['blue_count']

        probs = predict_similarity(train_df, params, lottery_conf)

        if probs is not None and probs.sum() > 0:
            top_indices = probs.argsort()[::-1][:top_n]
            pred_balls = set([lottery_conf['num_list'][idx] for idx in top_indices])
            hits = len(actual_balls & pred_balls)
            total_hits += hits

    return total_hits / (BACKTEST_PERIODS * top_n) if (BACKTEST_PERIODS * top_n) > 0 else 0

def objective(trial, full_df, conf, search_limit, top_matches, pool_type):
    """Optuna objective function to optimize weights."""
    weights = {
        'overlap': trial.suggest_float('overlap', 0.1, 20.0),
        'sum': trial.suggest_float('sum', 0.0, 2.0),
        'ac': trial.suggest_float('ac', 0.0, 2.0),
        'consecutive': trial.suggest_float('consecutive', 0.0, 5.0),
        'neighbor': trial.suggest_float('neighbor', 0.0, 5.0),
        'repeat': trial.suggest_float('repeat', 0.0, 5.0),
        'jump': trial.suggest_float('jump', 0.0, 2.0),
        'omission': trial.suggest_float('omission', 0.0, 5.0)
    }
    
    params = {'search_limit': search_limit, 'top_matches': top_matches, 'weights': weights}
    return evaluate_performance(params, full_df, conf, pool_type)

def run_tuning():
    """Main function to run the tuning process."""
    logging.info(f"🚀 开始为 [{LOTTERY_NAME}] 的模型 A 进行参数调优...")
    
    conf = LOTTERY_CONFIG[LOTTERY_NAME]
    df = load_data(conf)
    if df is None: return

    red_num_list = list(range(conf['red_range'][0], conf['red_range'][1] + 1))
    blue_num_list = list(range(conf['blue_range'][0], conf['blue_range'][1] + 1))
    red_cols = [f"{conf['red_col_prefix']}{i}" for i in range(1, conf['red_count'] + 1)]
    blue_cols = [f"{conf['blue_col_name']}{i}" for i in range(1, conf['blue_count'] + 1)] if conf['blue_count'] > 1 else [conf['blue_col_name']]
    
    omission_df = get_omission_matrix(df, red_num_list, blue_num_list, red_cols, blue_cols)
    full_df = pd.concat([df, omission_df], axis=1)

    grid_params = {'search_limit': [2000, 3000, 5000], 'top_matches': [15, 20, 25]}
    grid_combinations = list(itertools.product(grid_params['search_limit'], grid_params['top_matches']))
    
    best_config = {}

    for pool_type in ['red', 'blue']:
        logging.info(f"\n--- 正在调优 {pool_type.upper()} 球参数 ---")
        best_score = -1
        best_params = None
        
        for sl, tm in tqdm(grid_combinations, desc=f"Grid Search ({pool_type.upper()})"):
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, full_df, conf, sl, tm, pool_type), n_trials=OPTUNA_TRIALS_PER_GRID)
            
            if study.best_value > best_score:
                best_score = study.best_value
                best_params = {
                    'search_limit': sl, 'top_matches': tm, 'weights': study.best_params
                }
        best_config[pool_type] = {'score': best_score, 'params': best_params}

    print("\n" + "="*50)
    print("🎉 调优完成！模型 A 建议参数 (SSQ):")
    print("="*50)
    
    red_conf = best_config['red']
    print("\n[红球] Best Score (Avg Hit Rate): {:.2%}".format(red_conf['score']))
    print("Best Parameters:")
    print(json.dumps(red_conf['params'], indent=4, ensure_ascii=False))

    blue_conf = best_config['blue']
    print("\n[蓝球] Best Score (Avg Hit Rate): {:.2%}".format(blue_conf['score']))
    print("Best Parameters:")
    print(json.dumps(blue_conf['params'], indent=4, ensure_ascii=False))
    
    print("\n请将以上参数更新到 multi_model.py 的 MODEL_CONFIG['A']['ssq_red'] 和 MODEL_CONFIG['A']['ssq_blue'] 中。")

if __name__ == "__main__":
    run_tuning()