import os
import sys
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

"""
这是一个非常好的优化方向。针对双色球（SSQ）这种红蓝球分离的彩票，针对性地调优每个模型的参数可以显著提升预测效果。

由于 multi_model.py 中的代码结构依赖于全局变量且在导入时会执行初始化代码，直接导入可能会导致冲突。为了保证调优脚本的独立性和稳定性，我将创建一个独立的脚本 tune_ssq_params.py。

这个脚本将包含以下功能：

独立的数据加载与特征工程：复用 multi_model.py 的核心逻辑，但针对双色球进行固化。
参数搜索空间定义：为每个模型定义合理的超参数范围。
随机搜索 (Random Search)：相比网格搜索，随机搜索在处理多参数时效率更高。
回测评估：使用最近 30 期数据作为验证集，以“命中率”为核心指标评估每一组参数。
"""

# Ensure funcs and models can be imported
sys.path.append(os.getcwd())
from config import LOTTERY_CONFIG
from models.method_h import train_predict_evt

# --- Configuration ---
LOTTERY_NAME = "双色球"
RECENT_PERIODS_OPTIONS = [30, 50, 75, 100, 150]
BACKTEST_LEN = 50 # Evaluate on the last 50 draws

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_data(conf):
    try:
        df = pd.read_csv(conf['data_file'])
        # Sort by issue ascending
        if 'issue' in df.columns:
            df = df.rename(columns={'issue': '期号'})
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce').fillna(0).astype(int).astype(str)
        if int(df['期号'].iloc[0]) > int(df['期号'].iloc[-1]):
            df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def evaluate_model(df, recent_periods, sum_min, sum_max, conf):
    """
    Backtests Model H over the last BACKTEST_LEN periods.
    Metric: Average rank of the actual drawn red balls in the predicted distribution.
    Lower is better.
    """
    total_rank_sum = 0
    total_hits = 0
    
    # We iterate over the last BACKTEST_LEN draws
    # For each draw, we predict using all data up to that draw
    start_idx = len(df) - BACKTEST_LEN
    
    red_cols = [f"{conf['red_col_prefix']}{i}" for i in range(1, conf['red_count'] + 1)]
    
    model_config = {
        'recent_periods': recent_periods,
        'sum_min': sum_min,
        'sum_max': sum_max
    }
    
    for i in range(start_idx, len(df)):
        # Data available UP TO (but not including) index i
        train_df = df.iloc[:i]
        actual_reds = df.iloc[i][red_cols].values.astype(int)
        
        # Predict
        probs = train_predict_evt(train_df, model_config, conf)
        
        # Rank numbers (1 is highest probability)
        # Probabilities are for the entire red pool (usually 1-33)
        # Note: Model H returns probs for red_num_list
        red_num_list = conf['red_num_list']
        
        # Create a mapping of number -> probability
        num_prob = {num: probs[idx] for idx, num in enumerate(red_num_list)}
        
        # Sort by probability descending
        sorted_nums = sorted(num_prob.items(), key=lambda x: x[1], reverse=True)
        rank_map = {num: rank + 1 for rank, (num, prob) in enumerate(sorted_nums)}
        
        # Calculate ranks of actual numbers
        current_ranks = [rank_map[num] for num in actual_reds if num in rank_map]
        total_rank_sum += sum(current_ranks)
        total_hits += len(current_ranks)
        
    avg_rank = total_rank_sum / total_hits if total_hits > 0 else 33/2
    return avg_rank

def run_tuning():
    logging.info(f"🚀 开始为 [{LOTTERY_NAME}] 的模型 H (EVT) 进行参数调优...")
    
    conf = LOTTERY_CONFIG[LOTTERY_NAME].copy() # 使用副本防止污染全局配置
    
    # 补全动态字段
    conf['red_cols'] = [f"{conf['red_col_prefix']}{i}" for i in range(1, conf['red_count'] + 1)]
    if conf.get('separate_pool', False):
        conf['blue_cols'] = [conf['blue_col_name']] if 'blue_col_name' in conf else ['蓝球']
        conf['red_num_list'] = list(range(conf['red_range'][0], conf['red_range'][1] + 1))
        conf['blue_num_list'] = list(range(conf['blue_range'][0], conf['blue_range'][1] + 1))
        conf['total_red'] = len(conf['red_num_list'])
        conf['total_numbers'] = len(conf['red_num_list']) + len(conf['blue_num_list'])
    else:
        conf['num_list'] = list(range(conf['red_range'][0], conf['red_range'][1] + 1))
        conf['red_num_list'] = conf['num_list'] # H 模型内部可能用到
        conf['total_numbers'] = len(conf['num_list'])

    df = load_data(conf)
    if df is None: return

    # 1. Calculate Global 3-Sigma Thresholds
    sums = df['和值']
    mean_s, std_s = sums.mean(), sums.std()
    sum_min, sum_max = mean_s - 3 * std_s, mean_s + 3 * std_s
    
    logging.info(f"全局统计 - Mean: {mean_s:.2f}, Std: {std_s:.2f}")
    logging.info(f"3σ 极值区间: [{sum_min:.2f}, {sum_max:.2f}]")
    
    # 2. Grid Search for recent_periods
    results = []
    for rp in RECENT_PERIODS_OPTIONS:
        logging.info(f"正在测试 recent_periods = {rp} ...")
        avg_rank = evaluate_model(df, rp, sum_min, sum_max, conf)
        results.append((rp, avg_rank))
        logging.info(f"  - Average Rank: {avg_rank:.4f}")

    # 3. Sort by performance
    results.sort(key=lambda x: x[1])
    
    best_rp, best_rank = results[0]
    
    print("\n" + "="*50)
    print("🎉 调优完成！模型 H 建议参数 (SSQ):")
    print("="*50)
    print(f"Best recent_periods: {best_rp}")
    print(f"Best Average Rank: {best_rank:.4f}")
    print(f"Calculated sum_min: {int(sum_min)}")
    print(f"Calculated sum_max: {int(sum_max)}")
    print("-" * 50)
    
    # Also test against the OLD parameters to see if we improved
    logging.info("正在测试旧参数 (recent_periods=50, sum_min=75, sum_max=135) 作为对比...")
    old_rank = evaluate_model(df, 50, 75, 135, conf)
    logging.info(f"  - Old Parameters Average Rank: {old_rank:.4f}")

if __name__ == "__main__":
    run_tuning()
