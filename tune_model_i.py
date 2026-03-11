import os
import sys
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import itertools

# Ensure funcs and models can be imported
sys.path.append(os.getcwd())
from config import LOTTERY_CONFIG
from models.method_i import train_predict_ga

# --- Configuration ---
LOTTERY_NAME = "双色球"
MUTATION_RATE_OPTIONS = [0.05, 0.1, 0.15, 0.2]
FITNESS_PERIODS_OPTIONS = [10, 20, 30, 50]
BACKTEST_LEN = 30 # Blue is faster, can test more
POPULATION_SIZE = 50
GENERATIONS = 30

# Suppress verbose logging from GA during tuning
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')

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

def evaluate_model_blue(df, mutation_rate, fitness_periods, conf):
    """
    Backtests Model I for BLUE BALLS.
    Metric: Average rank of the actual drawn blue ball.
    """
    total_rank_sum = 0
    total_hits = 0
    
    start_idx = len(df) - BACKTEST_LEN
    blue_cols = conf['blue_cols']
    
    # We need to hack the train_predict_ga to treat blue balls as red
    # because the GA implementation is hardcoded for 'red_cols' and 'red_range'
    
    # Prepare a fake config that makes GA think Blue is Red
    fake_conf = conf.copy()
    fake_conf['red_cols'] = blue_cols
    fake_conf['red_range'] = conf['blue_range']
    fake_conf['red_count'] = conf['blue_count']
    fake_conf['red_num_list'] = list(range(conf['blue_range'][0], conf['blue_range'][1] + 1))
    fake_conf['num_list'] = fake_conf['red_num_list'] # Add this to avoid KeyError
    fake_conf['total_numbers'] = len(fake_conf['red_num_list'])
    fake_conf['separate_pool'] = False # Single pool mode for the fake call
    
    model_config = {
        'population_size': POPULATION_SIZE,
        'generations': GENERATIONS,
        'mutation_rate': mutation_rate,
        'fitness_periods': fitness_periods
    }
    
    for i in range(start_idx, len(df)):
        train_df = df.iloc[:i]
        actual_blues = df.iloc[i][blue_cols].values.astype(int)
        
        # Predict
        probs = train_predict_ga(train_df, model_config, fake_conf)
        
        # Rank
        num_list = fake_conf['red_num_list']
        num_prob = {num: probs[idx] for idx, num in enumerate(num_list)}
        sorted_nums = sorted(num_prob.items(), key=lambda x: x[1], reverse=True)
        rank_map = {num: rank + 1 for rank, (num, prob) in enumerate(sorted_nums)}
        
        current_ranks = [rank_map[num] for num in actual_blues if num in rank_map]
        total_rank_sum += sum(current_ranks)
        total_hits += len(current_ranks)
        
    avg_rank = total_rank_sum / total_hits if total_hits > 0 else 16/2
    return avg_rank

def run_tuning():
    print(f"🚀 开始为 [{LOTTERY_NAME}] 的模型 I (GA) 进行 [蓝球] 参数调优...")
    
    conf = LOTTERY_CONFIG[LOTTERY_NAME].copy()
    # Initial setup for blue cols identification
    conf['blue_cols'] = ['蓝球'] if '蓝球' in pd.read_csv(conf['data_file']).columns else ['蓝球1']
    conf['name'] = LOTTERY_NAME

    df = load_data(conf)
    if df is None: return

    results = []
    combinations = list(itertools.product(MUTATION_RATE_OPTIONS, FITNESS_PERIODS_OPTIONS))
    
    for mr, fp in tqdm(combinations, desc="网格搜索进度"):
        avg_rank = evaluate_model_blue(df, mr, fp, conf)
        results.append(((mr, fp), avg_rank))

    # Sort by performance
    results.sort(key=lambda x: x[1])
    
    (best_mr, best_fp), best_rank = results[0]
    
    print("\n" + "="*50)
    print("🎉 调优完成！模型 I 建议 [蓝球] 参数 (SSQ):")
    print("="*50)
    print(f"Best mutation_rate: {best_mr}")
    print(f"Best fitness_periods: {best_fp}")
    print(f"Best Average Rank: {best_rank:.4f}")
    print("-" * 50)
    
    print("正在计算旧参数 (mr=0.15, fp=15) 表现以供对比...")
    old_rank = evaluate_model_blue(df, 0.15, 15, conf)
    print(f"  - 旧参数 Average Rank: {old_rank:.4f}")

if __name__ == "__main__":
    run_tuning()
