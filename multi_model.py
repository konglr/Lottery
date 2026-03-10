import logging
import sys
import os
import time

# --- Immediate Feedback ---
print("🚀 [System] 启动模型分析引擎...", flush=True)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("正在加载核心预测引擎, 请稍候 (首次运行可能较慢)...")

import pandas as pd
import numpy as np
import argparse
import csv
import json
from datetime import datetime
import warnings
# Suppress LightGBM/Sklearn feature name warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# --- Heavy Imports (might be slow) ---
import matplotlib
matplotlib.use('Agg') # 关键：禁用 GUI 后端，防止在无显示器环境下挂起
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import LOTTERY_CONFIG

from models.method_a import predict_similarity
from models.method_b import train_predict_rf
from models.method_c import train_predict_xgb
from models.method_d import train_predict_lstm
from models.method_e import train_predict_lgbm
from models.method_f import train_predict_catboost
from models.method_g import train_predict_hmm
from models.method_h import train_predict_evt
from models.method_i import train_predict_ga
from models.method_j import train_predict_poisson
# --- Reproducibility ---
np.random.seed(42)

# --- Global Configuration ---
def init_config():
    parser = argparse.ArgumentParser(description="Multi-Model Lottery Prediction")
    parser.add_argument("--lottery", type=str, default="all", help="彩票名称，支持多个以逗号分隔或输入 'all'")
    parser.add_argument("--method", type=str, choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'all'], default='all', help="分析方法")
    parser.add_argument("--eval_size", type=int, default=10, help="回测期数")
    args = parser.parse_args()
    
    lotteries = []
    if args.lottery.lower() == 'all':
        lotteries = list(LOTTERY_CONFIG.keys())
    else:
        lotteries = [l.strip() for l in args.lottery.split(',') if l.strip() in LOTTERY_CONFIG]
    
    if not lotteries:
        print(f"Error: 找不到有效的彩票类型或名称 '{args.lottery}'")
        sys.exit(1)
        
    return args, lotteries

def update_lottery_config(lottery_name, df=None):
    global LOTTERY_NAME, DATA_FILE, RED_COUNT, BLUE_COUNT, RED_RANGE, BLUE_RANGE, RED_COL_PREFIX, RED_COLS, BLUE_COLS, SEPARATE_POOL
    global TOTAL_RED, TOTAL_BLUE, RED_NUM_LIST, BLUE_NUM_LIST, WINDOW_SIZE, BACKTEST_CSV, STAT_COLS, STAT_MAP, TOTAL_NUMBERS, NUM_LIST
    
    conf = LOTTERY_CONFIG[lottery_name]
    LOTTERY_NAME = lottery_name
    DATA_FILE = conf['data_file']
    RED_COUNT = conf['red_count']
    BLUE_COUNT = conf.get('blue_count', 0)
    RED_RANGE = conf['red_range'] 
    BLUE_RANGE = conf.get('blue_range', (0, 0))
    RED_COL_PREFIX = conf['red_col_prefix']
    SEPARATE_POOL = conf.get('separate_pool', False)
    
    # Initialize Lists
    RED_COLS = [f"{RED_COL_PREFIX}{i}" for i in range(1, RED_COUNT + 1)]
    BLUE_COLS = []
    
    if conf.get('has_blue'):
        blue_prefix = conf.get('blue_col_name', '蓝球')
        if BLUE_COUNT > 1:
            for i in range(1, BLUE_COUNT + 1):
                col_found = False
                if df is not None:
                    for p in [blue_prefix, '蓝球']:
                        col_name = f"{p}{i}"
                        if col_name in df.columns:
                            BLUE_COLS.append(col_name)
                            col_found = True
                            break
                if not col_found:
                    BLUE_COLS.append(f"{blue_prefix}{i}")
        else:
            col_found = False
            if df is not None:
                for col_name in [blue_prefix, '蓝球']:
                    if col_name in df.columns:
                        BLUE_COLS.append(col_name)
                        col_found = True
                        break
            if not col_found:
                BLUE_COLS.append(blue_prefix)

    # Logic for Model Targets
    if SEPARATE_POOL:
        # Dual Pool: Red and Blue are independent
        RED_NUM_LIST = list(range(RED_RANGE[0], RED_RANGE[1] + 1))
        BLUE_NUM_LIST = list(range(BLUE_RANGE[0], BLUE_RANGE[1] + 1))
        TOTAL_RED = len(RED_NUM_LIST)
        TOTAL_BLUE = len(BLUE_NUM_LIST)
        TOTAL_NUMBERS = TOTAL_RED + TOTAL_BLUE
        NUM_LIST = RED_NUM_LIST + BLUE_NUM_LIST # concatenated for vector indexing
    else:
        # Single Pool (like QLC): Red and Blue come from same range
        # We combine them into a single list of columns for analysis
        # Use dict.fromkeys to merge and uniqueify while preserving order
        all_cols = RED_COLS + BLUE_COLS
        RED_COLS = list(dict.fromkeys(all_cols))
        BLUE_COLS = [] # Reset blue cols as they are merged into red
        
        RED_NUM_LIST = list(range(RED_RANGE[0], RED_RANGE[1] + 1))
        BLUE_NUM_LIST = []
        TOTAL_RED = len(RED_NUM_LIST)
        TOTAL_BLUE = 0
        TOTAL_NUMBERS = TOTAL_RED
        NUM_LIST = RED_NUM_LIST
            
    WINDOW_SIZE = conf.get('window_size', 4)
    BACKTEST_CSV = f"data/{conf['code']}_backtest.csv"
    STAT_COLS = []
    STAT_MAP = {}
    print(f"\n{'='*20} [切换彩票: {LOTTERY_NAME} ({'独立池' if SEPARATE_POOL else '单池'})] {'='*20}")
    return conf

args, LOTTERIES_TO_RUN = init_config()
# Pre-initialize globals with the first lottery
conf = update_lottery_config(LOTTERIES_TO_RUN[0])
DEFAULT_EVAL_SIZE = args.eval_size

# --- Globals for Dynamic Features ---
STAT_COLS = []
STAT_MAP = {}

def determine_stat_features(df):
    """Dynamically identify relevant statistical columns from the dataframe."""
    global STAT_COLS, STAT_MAP
    
    # Core stats that should always be checked
    core_stats = ['和值', 'AC', '跨度', '奇数', '大号', '重号', '邻号', '孤号', '一区', '二区', '三区']
    STAT_COLS = [c for c in core_stats if c in df.columns]
    
    # Add all '连' (consecutive) and '跳' (jump) columns
    pattern_cols = [c for c in df.columns if c.endswith('连') or c.endswith('跳')]
    # Sort them to maintain consistent feature order (e.g., 二连, 三连...)
    # Note: Chinese sorting might be tricky, but consistent relative order is enough.
    STAT_COLS.extend(sorted(pattern_cols))
    
    # Translation map for plots
    translation = {
        '和值': 'Sum', 'AC': 'AC', '跨度': 'Span', '奇数': 'OddCnt', 
        '大号': 'BigCnt', '重号': 'Repeat', '邻号': 'Neighbor', '孤号': 'Isolated',
        '一区': 'Zone1', '二区': 'Zone2', '三区': 'Zone3'
    }
    
    # Chinese number to Arabic map for Consec/Jump
    cn_num_map = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
        '十一': '11', '十二': '12', '十三': '13', '十四': '14', '十五': '15', '十六': '16', '十七': '17', '十八': '18', '十九': '19', '二十': '20'
    }
    
    for col in STAT_COLS:
        if col in translation:
            STAT_MAP[col] = translation[col]
        elif '连' in col:
            num_str = col.replace('连', '')
            eng_num = cn_num_map.get(num_str, num_str)
            STAT_MAP[col] = f"Consec_{eng_num}"
        elif '跳' in col:
            num_str = col.replace('跳', '')
            eng_num = cn_num_map.get(num_str, num_str)
            STAT_MAP[col] = f"Jump_{eng_num}"
        else:
            STAT_MAP[col] = col
    
    logging.info(f"动态特征识别完成: {len(STAT_COLS)} 个统计特征 [{', '.join(STAT_COLS)}]")

# --- MODEL HYPERPARAMETERS ---
MODEL_CONFIG = {
    'A': {
        'search_limit': 3000,
        'top_matches': 20,
        'weights': {
            'overlap': 10,
            'sum': 0.1,
            'ac': 0.5,
            'consecutive': 2.0,
            'neighbor': 1.5,
            'repeat': 1.0,
            'jump': 0.5,
            'omission': 1.0
        }
    },
    'B': {
        'n_estimators': 2000,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 10,
        'max_features': 'log2',
        'random_state': 42,
        'n_jobs': -1
    },
    'C': {
        'n_estimators': 1000,
        'max_depth': 4,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'scale_pos_weight': 5,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    },
    'D': {
        'embedding_dim': 16,
        'hidden_dim': 32,
        'num_layers': 1,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 50,
    },
    'E': {
        'n_estimators': 400,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'random_state': 42,
        'n_jobs': -1,
        # High-dimension specific (e.g., for '快乐8'): 'feature_fraction': 0.8, 'bagging_fraction': 0.8
        # Imbalance specific (e.g., for '双色球'): 'is_unbalance': True
    },
    'F': {
        'iterations': 500,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_state': 42,
        # For '快乐8', depth will be dynamically set to 4.
    },
    'G': {
        'n_components': 4,
        'covariance_type': 'diag',
        'train_size': 800,
        'random_state': 42
    },
    'H': {
        'recent_periods': 50,
        # 'sum_min' and 'sum_max' will be set dynamically based on lottery type
        # Default fallback: 3-sigma calculated in model
    },
    'I': {
        'population_size': 100,
        'generations': 50,
        'mutation_rate': 0.1,
        'fitness_periods': 10 # How many recent periods to use for fitness evaluation
    },
    'J': {
        # Poisson model is parameter-free mostly, relying on theoretical probabilities
    }
}

# --- 1. Data Loading & Preprocessing ---

def recalculate_stats(df):
    """Recalculate statistical columns based on the current RED_COLS (e.g., 7+1 for QLC)."""
    logging.info(f"正在重算 {LOTTERY_NAME} 的统计特征 (基于 {len(RED_COLS)} 个号码)...")
    
    # 1. Prep boundaries
    midpoint = TOTAL_NUMBERS / 2
    z1_limit = TOTAL_NUMBERS / 3.0
    z2_limit = (TOTAL_NUMBERS * 2) / 3.0
    
    # Chinese Keys for Consec/Jump
    CN_KEYS = ["", "", "二", "三", "四", "五", "六", "七", "八", "九", "十", 
               "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十"]
    
    # Reset temporal stats and consec/jump targets
    temporal_cols = ['重号', '邻号', '孤号']
    pattern_cols = [f"{CN_KEYS[k]}连" for k in range(2, 7)] + [f"{CN_KEYS[k]}跳" for k in range(2, 7)]
    for c in temporal_cols + pattern_cols:
        if c in df.columns: df[c] = 0
    
    # 2. Row by Row Stats
    for idx in df.index:
        try:
            nums = sorted([int(v) for v in df.loc[idx, RED_COLS].values if pd.notna(v)])
            if not nums: continue
            n = len(nums)
            
            # Basic Stats
            df.loc[idx, '和值'] = sum(nums)
            df.loc[idx, '跨度'] = max(nums) - min(nums) if n >= 2 else 0
            df.loc[idx, '奇数'] = sum(1 for v in nums if v % 2 != 0)
            df.loc[idx, '大号'] = sum(1 for v in nums if v > midpoint)
            
            # AC
            if n >= 2:
                diffs = set(abs(a - b) for i, a in enumerate(nums) for b in nums[i+1:])
                df.loc[idx, 'AC'] = len(diffs) - (n - 1)
            
            # Zones
            z1, z2, z3 = 0, 0, 0
            for v in nums:
                if v <= z1_limit: z1 += 1
                elif v <= z2_limit: z2 += 1
                else: z3 += 1
            df.loc[idx, '一区'], df.loc[idx, '二区'], df.loc[idx, '三区'] = z1, z2, z3
            
            # Patterns (Consecutive)
            i_c = 0
            while i_c < n - 1:
                if nums[i_c] + 1 == nums[i_c + 1]:
                    length = 2
                    while i_c + length < n and nums[i_c + length - 1] + 1 == nums[i_c + length]:
                        length += 1
                    if length <= 6:
                        key = f"{CN_KEYS[length]}连"
                        if key in df.columns: df.loc[idx, key] += 1
                    i_c += length
                else: i_c += 1
                
            # Patterns (Jumps)
            i_j = 0
            while i_j < n - 1:
                diff = nums[i_j + 1] - nums[i_j]
                if diff >= 2:
                    length = 1
                    while i_j + length < n - 1 and nums[i_j + length + 1] - nums[i_j + length] == diff:
                        length += 1
                    jump_key = None
                    if diff == 2:
                        jump_key = {1:'二跳', 2:'三跳', 3:'四跳', 4:'五跳', 5:'六跳'}.get(length)
                    elif diff in [3, 4, 5, 6] and (length == diff - 1 or length == diff):
                        jump_key = {2:'三跳', 3:'四跳', 4:'五跳', 5:'六跳'}.get(length)
                    if jump_key and jump_key in df.columns:
                        df.loc[idx, jump_key] += 1
                        i_j += (length + 1)
                    else: i_j += 1
                else: i_j += 1
        except: continue

    # 3. Temporal Pass (Oldest to Newest)
    for i in range(1, len(df)):
        try:
            curr = set([int(v) for v in df.loc[i, RED_COLS].values if pd.notna(v)])
            prev = set([int(v) for v in df.loc[i-1, RED_COLS].values if pd.notna(v)])
            if not curr or not prev: continue
            rep = len(curr & prev)
            adj = sum(1 for n in curr if (n - 1 in prev or n + 1 in prev))
            df.loc[i, '重号'], df.loc[i, '邻号'], df.loc[i, '孤号'] = rep, adj, len(curr) - rep - adj
        except: continue
        
    return df


def load_data(file_path=DATA_FILE):
    try:
        df = pd.read_csv(file_path)
        if 'issue' in df.columns:
            df = df.rename(columns={'issue': '期号'})
        # Ensure '期号' is read as integer then string to remove .0 decimals
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce').fillna(0).astype(int).astype(str)
        # Ensure oldest to newest
        if int(df['期号'].iloc[0]) > int(df['期号'].iloc[-1]):
            df = df.iloc[::-1].reset_index(drop=True)
            
        # Recalculate stats for QLC (7+1=8 balls)
        if LOTTERY_NAME == "七乐彩":
            df = recalculate_stats(df)
            
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def get_omission_matrix(df):
    """Calculate separate omission matrices for red and blue if SEPARATE_POOL."""
    n_rows = len(df)
    omission_matrix = np.zeros((n_rows, TOTAL_NUMBERS), dtype=int)
    current_red_omiss = np.zeros(TOTAL_RED, dtype=int)
    current_blue_omiss = np.zeros(TOTAL_BLUE, dtype=int)
    
    for i in range(n_rows):
        # Current state before this draw
        if SEPARATE_POOL:
            omission_matrix[i, :TOTAL_RED] = current_red_omiss
            omission_matrix[i, TOTAL_RED:] = current_blue_omiss
            
            # Update after draw
            row_red = set(df.loc[i, RED_COLS].values)
            row_blue = set(df.loc[i, BLUE_COLS].values)
            
            for idx, num in enumerate(RED_NUM_LIST):
                if num in row_red: current_red_omiss[idx] = 0
                else: current_red_omiss[idx] += 1
            for idx, num in enumerate(BLUE_NUM_LIST):
                if num in row_blue: current_blue_omiss[idx] = 0
                else: current_blue_omiss[idx] += 1
        else:
            omission_matrix[i] = current_red_omiss # combined in single pool
            row_nums = set(df.loc[i, RED_COLS].values)
            for idx, num in enumerate(RED_NUM_LIST):
                if num in row_nums: current_red_omiss[idx] = 0
                else: current_red_omiss[idx] += 1
    
    # Generate Col Names
    cols = []
    if SEPARATE_POOL:
        cols.extend([f'Red_Omiss_{i}' for i in RED_NUM_LIST])
        cols.extend([f'Blue_Omiss_{i}' for i in BLUE_NUM_LIST])
        next_omiss = np.concatenate([current_red_omiss, current_blue_omiss])
    else:
        cols.extend([f'Omission_{i}' for i in RED_NUM_LIST])
        next_omiss = current_red_omiss
        
    return pd.DataFrame(omission_matrix, columns=cols), next_omiss

def extract_features(df, window_data, next_omission, pool_type='all'):
    """Unified feature extraction for ML models."""
    features = []

    # New logic for separate training pools
    if SEPARATE_POOL and pool_type != 'all':
        if pool_type == 'red':
            # A. Red Freq
            red_balls = window_data[RED_COLS].values.flatten()
            red_balls = red_balls[~np.isnan(red_balls)].astype(int)
            r_freq = np.zeros(TOTAL_RED)
            for b in red_balls:
                if RED_RANGE[0] <= b <= RED_RANGE[1]: r_freq[b - RED_RANGE[0]] += 1
            features.extend(r_freq)
            
            # B. Red Omission
            features.extend(next_omission[:TOTAL_RED])
            
            # C. Stats (Mean and Last) - these are red-ball specific
            w_data = window_data.copy()
            for col in STAT_COLS:
                if col not in w_data.columns: w_data[col] = 0
            features.extend(w_data[STAT_COLS].mean().values)
            features.extend(w_data[STAT_COLS].iloc[-1].values)
            
            # D. Tail Frequency (Red only)
            red_balls_for_tail = window_data[RED_COLS].values.flatten()
            red_balls_for_tail = red_balls_for_tail[~np.isnan(red_balls_for_tail)].astype(int)
            tails = [int(n) % 10 for n in red_balls_for_tail if n >= 0]
            tail_counts = np.bincount(tails, minlength=10)
            features.extend(tail_counts)
            
        elif pool_type == 'blue':
            # A. Blue Freq
            if BLUE_COLS:
                blue_balls = window_data[BLUE_COLS].values.flatten()
                blue_balls = blue_balls[~np.isnan(blue_balls)].astype(int)
                b_freq = np.zeros(TOTAL_BLUE)
                for b in blue_balls:
                    if BLUE_RANGE[0] <= b <= BLUE_RANGE[1]: b_freq[b - BLUE_RANGE[0]] += 1
                features.extend(b_freq)
            else:
                features.extend(np.zeros(TOTAL_BLUE))
            
            # B. Blue Omission
            if len(next_omission) >= TOTAL_RED + TOTAL_BLUE:
                features.extend(next_omission[TOTAL_RED:TOTAL_RED+TOTAL_BLUE])
            else:
                features.extend(np.zeros(TOTAL_BLUE))
            
            # C. Blue Advanced Stats (Parity, Size, Prime, Diff)
            if BLUE_COLS:
                blue_vals = window_data[BLUE_COLS].values.flatten()
                blue_vals = blue_vals[~np.isnan(blue_vals)].astype(int)
                
                if len(blue_vals) > 0:
                    # 1. Parity (Odd ratio)
                    odd_ratio = np.mean([1 if x % 2 != 0 else 0 for x in blue_vals])
                    
                    # 2. Size (Big ratio)
                    mid_point = (BLUE_RANGE[1] - BLUE_RANGE[0]) / 2 + BLUE_RANGE[0]
                    big_ratio = np.mean([1 if x > mid_point else 0 for x in blue_vals])
                    
                    # 3. Prime (Prime ratio)
                    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47} # Covers most blue ranges
                    prime_ratio = np.mean([1 if x in primes else 0 for x in blue_vals])
                    
                    # 4. Diff (Abs diff with prev)
                    if len(blue_vals) >= 2:
                        last_diff = abs(blue_vals[-1] - blue_vals[-2])
                    else:
                        last_diff = 0
                    
                    features.extend([odd_ratio, big_ratio, prime_ratio, last_diff])
                else:
                    features.extend([0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0])
                
            # D. Red Stats for Blue (Red Sum, Red Span) - Cross-pool correlation
            # Use the last row of window_data for the most recent red stats
            if not window_data.empty:
                last_row = window_data.iloc[-1]
                # Calculate Red Sum
                red_sum = 0
                red_span = 0
                try:
                    # Extract red numbers from the last row
                    current_reds = [int(last_row[c]) for c in RED_COLS if pd.notna(last_row.get(c))]
                    if current_reds:
                        red_sum = sum(current_reds)
                        red_span = max(current_reds) - min(current_reds)
                except: pass
                features.extend([red_sum, red_span])
            else:
                features.extend([0, 0])
        
        return np.array(features)

    # A. Hot/Cold (Freq in window)
    if SEPARATE_POOL:
        red_balls = window_data[RED_COLS].values.flatten()
        red_balls = red_balls[~np.isnan(red_balls)].astype(int)  # 过滤 NaN
        # Offset frequencies for bincount based on range start
        r_freq = np.zeros(TOTAL_RED)
        for b in red_balls:
            if RED_RANGE[0] <= b <= RED_RANGE[1]: r_freq[b - RED_RANGE[0]] += 1
        features.extend(r_freq)
        
        # Blue balls (only if BLUE_COLS exists)
        if BLUE_COLS:
            blue_balls = window_data[BLUE_COLS].values.flatten()
            blue_balls = blue_balls[~np.isnan(blue_balls)].astype(int)  # 过滤 NaN
            b_freq = np.zeros(TOTAL_BLUE)
            for b in blue_balls:
                if BLUE_RANGE[0] <= b <= BLUE_RANGE[1]: b_freq[b - BLUE_RANGE[0]] += 1
            features.extend(b_freq)
    else:
        all_balls = window_data[RED_COLS].values.flatten().astype(int)
        freq = np.zeros(TOTAL_RED)
        for b in all_balls:
            if RED_RANGE[0] <= b <= RED_RANGE[1]: freq[b - RED_RANGE[0]] += 1
        features.extend(freq)
    
    # B. Next Omission State
    features.extend(next_omission)
    
    # C. Stats (Mean and Last)
    w_data = window_data.copy()
    for col in STAT_COLS:
        if col not in w_data.columns: w_data[col] = 0
    features.extend(w_data[STAT_COLS].mean().values)
    features.extend(w_data[STAT_COLS].iloc[-1].values)
    
    # D. Tail Frequency (Red only usually enough for patterns)
    red_balls_for_tail = window_data[RED_COLS].values.flatten().astype(int)
    tails = [int(n) % 10 for n in red_balls_for_tail if n >= 0]
    tail_counts = np.bincount(tails, minlength=10)
    features.extend(tail_counts)
    
    return np.array(features)

def get_feature_names(pool_type='all'):
    """Map indices to human-readable names."""
    names = []

    if SEPARATE_POOL and pool_type != 'all':
        if pool_type == 'red':
            names.extend([f"Red_Freq_{i:02d}" for i in RED_NUM_LIST])
            names.extend([f"Red_Omiss_{i:02d}" for i in RED_NUM_LIST])
            names.extend([f"Avg_{STAT_MAP[c]}" for c in STAT_COLS])
            names.extend([f"Last_{STAT_MAP[c]}" for c in STAT_COLS])
            names.extend([f"Tail_{i}" for i in range(10)])
        elif pool_type == 'blue':
            names.extend([f"Blue_Freq_{i:02d}" for i in BLUE_NUM_LIST])
            names.extend([f"Blue_Omiss_{i:02d}" for i in BLUE_NUM_LIST])
            names.extend(["Blue_Odd_Ratio", "Blue_Big_Ratio", "Blue_Prime_Ratio", "Blue_Last_Diff"])
            names.extend(["Red_Sum_For_Blue", "Red_Span_For_Blue"])
        return names

    # A. Freq
    if SEPARATE_POOL:
        names.extend([f"Red_Freq_{i:02d}" for i in RED_NUM_LIST])
        names.extend([f"Blue_Freq_{i:02d}" for i in BLUE_NUM_LIST])
    else:
        names.extend([f"Freq_{i:02d}" for i in RED_NUM_LIST])
        
    # B. Omission
    if SEPARATE_POOL:
        names.extend([f"Red_Omiss_{i:02d}" for i in RED_NUM_LIST])
        names.extend([f"Blue_Omiss_{i:02d}" for i in BLUE_NUM_LIST])
    else:
        names.extend([f"Omiss_{i:02d}" for i in RED_NUM_LIST])
        
    # C. Stats
    names.extend([f"Avg_{STAT_MAP[c]}" for c in STAT_COLS])
    names.extend([f"Last_{STAT_MAP[c]}" for c in STAT_COLS])
    # D. Tails
    names.extend([f"Tail_{i}" for i in range(10)])
    return names

def plot_importance(importance, names, model_name, filename):
    """Generate R-style lollipop plot for feature importance."""
    # Use English names for plot saving and title
    model_id = "RF" if "RF" in model_name else "XGB"
    df = pd.DataFrame({'Feature': names, 'Importance': importance})
    df = df.sort_values(by='Importance', ascending=False).head(20)
    
    # Modern "ggplot-like" style
    plt.style.use('bmh') # Better than default for clean look
    plt.figure(figsize=(10, 8))
    
    # Lollipop aesthetics
    plt.hlines(y=range(len(df)), xmin=0, xmax=df['Importance'], color='skyblue', alpha=0.9, linewidth=1.5)
    plt.plot(df['Importance'], range(len(df)), "o", color='deepskyblue', markersize=10, markeredgecolor='steelblue', alpha=0.8)
    
    plt.title(f"Top 20 Feature Importance - {model_id}", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Importance Score", fontsize=12)
    plt.yticks(range(len(df)), df['Feature'], fontsize=10)
    
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    logging.info(f"Feature importance plot ({model_id}) saved to {filename}")
    return df.head(10)

def run_prediction(df, method, full_df, next_omission, conf=None):
    lottery_config = {
        'window_size': WINDOW_SIZE,
        'red_cols': RED_COLS,
        'blue_cols': BLUE_COLS,
        'num_list': NUM_LIST,
        'total_numbers': TOTAL_NUMBERS,
        'separate_pool': SEPARATE_POOL,
        'red_num_list': RED_NUM_LIST,
        'blue_num_list': BLUE_NUM_LIST,
        'total_red': TOTAL_RED,
        'red_col_prefix': RED_COL_PREFIX,
        'red_count': RED_COUNT,
        'blue_count': BLUE_COUNT,
        'red_range': RED_RANGE,
        'blue_range': BLUE_RANGE,
    }
    if method in ['B', 'C', 'E', 'F']:
        # New logic for separate pools
        if SEPARATE_POOL:
            # --- 1. Red Ball Training ---
            X_red_list, y_red_list = [], []
            omiss_cols = [c for c in full_df.columns if 'Omiss' in c]
            for i in range(WINDOW_SIZE, len(full_df)):
                win = full_df.iloc[i-WINDOW_SIZE : i]
                feat = extract_features(full_df, win, full_df.iloc[i][omiss_cols].values, pool_type='red')
                target = np.zeros(TOTAL_RED)
                for n in full_df.loc[i, RED_COLS].values:
                    try:
                        idx = RED_NUM_LIST.index(int(n))
                        target[idx] = 1
                    except (ValueError, TypeError): pass
                X_red_list.append(feat)
                y_red_list.append(target)
            
            last_win = full_df.iloc[-WINDOW_SIZE:]
            final_red_feat = extract_features(full_df, last_win, next_omission, pool_type='red')
            
            lottery_config_red = lottery_config.copy()
            lottery_config_red['total_numbers'] = TOTAL_RED
            
            train_func = {'B': train_predict_rf, 'C': train_predict_xgb, 'E': train_predict_lgbm, 'F': train_predict_catboost}[method]
            
            red_res = train_func(np.array(X_red_list), np.array(y_red_list), final_red_feat, MODEL_CONFIG[method], lottery_config_red)
            red_probs, red_importance = (red_res[0], red_res[1]) if isinstance(red_res, tuple) else (red_res, None)

            # --- 2. Blue Ball Training ---
            blue_probs = np.zeros(TOTAL_BLUE)
            if BLUE_COLS and TOTAL_BLUE > 0:
                X_blue_list, y_blue_list = [], []
                for i in range(WINDOW_SIZE, len(full_df)):
                    win = full_df.iloc[i-WINDOW_SIZE : i]
                    feat = extract_features(full_df, win, full_df.iloc[i][omiss_cols].values, pool_type='blue')
                    target = np.zeros(TOTAL_BLUE)
                    for n in full_df.loc[i, BLUE_COLS].values:
                        try:
                            idx = BLUE_NUM_LIST.index(int(n))
                            target[idx] = 1
                        except (ValueError, TypeError): pass
                    X_blue_list.append(feat)
                    y_blue_list.append(target)
                
                final_blue_feat = extract_features(full_df, last_win, next_omission, pool_type='blue')
                lottery_config_blue = lottery_config.copy()
                lottery_config_blue['total_numbers'] = TOTAL_BLUE
                
                blue_res = train_func(np.array(X_blue_list), np.array(y_blue_list), final_blue_feat, MODEL_CONFIG[method], lottery_config_blue)
                blue_probs, _ = (blue_res[0], blue_res[1]) if isinstance(blue_res, tuple) else (blue_res, None)

            # --- 3. Combine Results ---
            final_probs = np.concatenate([red_probs, blue_probs])
            return final_probs, red_importance

        # Original logic for single pool
        else:
            X_list, y_list = [], []
            omiss_cols = [c for c in full_df.columns if 'Omiss' in c]
            for i in range(WINDOW_SIZE, len(full_df)):
                win = full_df.iloc[i-WINDOW_SIZE : i]
                feat = extract_features(full_df, win, full_df.iloc[i][omiss_cols].values)
                target = np.zeros(TOTAL_NUMBERS)
                for n in full_df.loc[i, RED_COLS].values:
                    try:
                        idx = RED_NUM_LIST.index(int(n))
                        target[idx] = 1
                    except (ValueError, TypeError): pass
                X_list.append(feat)
                y_list.append(target)
            
            last_win = full_df.iloc[-WINDOW_SIZE:]
            final_feat = extract_features(full_df, last_win, next_omission)
            
            if method == 'B':
                return train_predict_rf(np.array(X_list), np.array(y_list), final_feat, MODEL_CONFIG['B'], lottery_config)
            elif method == 'C':
                return train_predict_xgb(np.array(X_list), np.array(y_list), final_feat, MODEL_CONFIG['C'], lottery_config)
            elif method == 'E':
                return train_predict_lgbm(np.array(X_list), np.array(y_list), final_feat, MODEL_CONFIG['E'], lottery_config)
            elif method == 'F':
                return train_predict_catboost(np.array(X_list), np.array(y_list), final_feat, MODEL_CONFIG['F'], lottery_config)

    elif method in ['A', 'D']:
        # Group 2: DataFrame based Models (A, D) - Now supporting Separation
        if SEPARATE_POOL:
            # --- 1. Red Ball Prediction ---
            # Create a config that looks like a single pool lottery for Red
            conf_red = lottery_config.copy()
            conf_red['separate_pool'] = False
            conf_red['total_numbers'] = TOTAL_RED
            conf_red['num_list'] = RED_NUM_LIST
            conf_red['red_num_list'] = RED_NUM_LIST # Critical for LSTM
            conf_red['red_cols'] = RED_COLS # Real Red Cols
            conf_red['blue_cols'] = []
            
            func = predict_similarity if method == 'A' else train_predict_lstm
            probs_red = func(df, MODEL_CONFIG[method], conf_red)
            
            # --- 2. Blue Ball Prediction ---
            # Create a config that looks like a single pool lottery for Blue
            conf_blue = lottery_config.copy()
            conf_blue['separate_pool'] = False
            conf_blue['total_numbers'] = TOTAL_BLUE
            conf_blue['num_list'] = BLUE_NUM_LIST
            conf_blue['red_num_list'] = BLUE_NUM_LIST # Treat Blue nums as "Red" for the model
            conf_blue['red_cols'] = BLUE_COLS # Treat Blue cols as "Red" (Primary) cols
            conf_blue['blue_cols'] = []
            
            probs_blue = func(df, MODEL_CONFIG[method], conf_blue)
            
            return np.concatenate([probs_red, probs_blue])
        else:
            # Standard call for single pool
            func = predict_similarity if method == 'A' else train_predict_lstm
            return func(df, MODEL_CONFIG[method], lottery_config)
            
    elif method == 'G':
        return train_predict_hmm(df, MODEL_CONFIG['G'], lottery_config)
    elif method == 'H':
        return train_predict_evt(df, MODEL_CONFIG['H'], lottery_config)
    elif method == 'I':
        return train_predict_ga(df, MODEL_CONFIG['I'], lottery_config)
    elif method == 'J':
        return train_predict_poisson(df, MODEL_CONFIG['J'], lottery_config)
    return np.zeros(TOTAL_NUMBERS)

def evaluate_methods(df, full_df, conf, test_size=10, active_methods=['A', 'B', 'C', 'D']):
    """Perform backtesting for all selected methods."""
    logging.info(f"开始历史回测分析 (最近 {test_size} 期)...")
    
    # Track hits: {method: {'red_6': 0, 'red_10': 0, 'blue_6': 0, 'blue_10': 0}}
    hits = {m: {'r1': 0, 'r2': 0, 'b1': 0, 'b2': 0} for m in active_methods + ['Ensemble']}
    history = {m: [] for m in active_methods + ['Ensemble']}
    
    # Prepare CSV
    with open(BACKTEST_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if os.path.getsize(BACKTEST_CSV) == 0:
            header = ['Run_Time', 'Target_Period']
            for m in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                header.append(f'Params_{m}')
                if SEPARATE_POOL:
                    for n in RED_NUM_LIST: header.append(f'Prob_{m}_R{n:02d}')
                    for n in BLUE_NUM_LIST: header.append(f'Prob_{m}_B{n:02d}')
                else:
                    for n in NUM_LIST: header.append(f'Prob_{m}_{n:02d}')
            writer.writerow(header)

    current_run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i in tqdm(range(len(df) - test_size, len(df)), desc="历史回测进度"):
        train_df = df.iloc[:i].reset_index(drop=True)
        train_full_df = full_df.iloc[:i].reset_index(drop=True)
        
        actual_red = set(df.iloc[i][RED_COLS].dropna().astype(int).values)
        actual_blue = set(df.iloc[i][BLUE_COLS].dropna().astype(int).values) if SEPARATE_POOL and BLUE_COLS else set()
        target_period = df.iloc[i]['期号']
        
        _, current_omission = get_omission_matrix(train_df)
        metrics = conf.get('eval_metrics', {"top_n_1": 6, "top_n_2": 10})
        n1, n2 = metrics['top_n_1'], metrics['top_n_2']
        
        current_probs = {}
        for m in active_methods:
            res = run_prediction(train_df, m, train_full_df, current_omission, conf)
            probs = res[0] if isinstance(res, tuple) else res
            current_probs[m] = probs
            
            if SEPARATE_POOL:
                r_p, b_p = probs[:TOTAL_RED], probs[TOTAL_RED:]
                hr1 = len(actual_red & set([RED_NUM_LIST[idx] for idx in r_p.argsort()[::-1][:n1]]))
                hr2 = len(actual_red & set([RED_NUM_LIST[idx] for idx in r_p.argsort()[::-1][:n2]]))
                hb1 = len(actual_blue & set([BLUE_NUM_LIST[idx] for idx in b_p.argsort()[::-1][:BLUE_COUNT]]))
                hits[m]['r1'] += hr1; hits[m]['r2'] += hr2; hits[m]['b1'] += hb1
                history[m].append((hr1, hr2, hb1))
            else:
                s_idx = probs.argsort()[::-1]
                h1 = len(actual_red & set([NUM_LIST[idx] for idx in s_idx[:n1]]))
                h2 = len(actual_red & set([NUM_LIST[idx] for idx in s_idx[:n2]]))
                hits[m]['r1'] += h1; hits[m]['r2'] += h2
                history[m].append((h1, h2))

        # Ensemble logic
        ens_scores = np.zeros(TOTAL_NUMBERS)
        for m in active_methods:
            p = current_probs[m]
            if p.max() > p.min():
                ens_scores += (p - p.min()) / (p.max() - p.min())
        ens_scores /= len(active_methods)
        
        if SEPARATE_POOL:
            r_p, b_p = ens_scores[:TOTAL_RED], ens_scores[TOTAL_RED:]
            hr1 = len(actual_red & set([RED_NUM_LIST[idx] for idx in r_p.argsort()[::-1][:n1]]))
            hr2 = len(actual_red & set([RED_NUM_LIST[idx] for idx in r_p.argsort()[::-1][:n2]]))
            hb1 = len(actual_blue & set([BLUE_NUM_LIST[idx] for idx in b_p.argsort()[::-1][:BLUE_COUNT]]))
            hits['Ensemble']['r1'] += hr1; hits['Ensemble']['r2'] += hr2; hits['Ensemble']['b1'] += hb1
            history['Ensemble'].append((hr1, hr2, hb1))
        else:
            s_idx = ens_scores.argsort()[::-1]
            h1 = len(actual_red & set([NUM_LIST[idx] for idx in s_idx[:n1]]))
            h2 = len(actual_red & set([NUM_LIST[idx] for idx in s_idx[:n2]]))
            hits['Ensemble']['r1'] += h1; hits['Ensemble']['r2'] += h2
            history['Ensemble'].append((h1, h2))
        
        # 写入回测数据到 CSV
        log_prediction_to_csv(current_run_time, target_period, current_probs, MODEL_CONFIG)

    # Calculate final average rates
    results = {}
    for m in active_methods + ['Ensemble']:
        r_denom = test_size * RED_COUNT
        b_denom = test_size * BLUE_COUNT if BLUE_COUNT > 0 else 1
        results[m] = {
            'avg_r1': hits[m]['r1'] / r_denom,
            'avg_r2': hits[m]['r2'] / r_denom,
            'avg_b1': hits[m]['b1'] / b_denom if BLUE_COUNT > 0 else 0,
            'history': history[m]
        }
    return results, current_run_time


def log_prediction_to_csv(run_time, target_period, results, models_config):
    """Logs a single prediction row to backtest.csv"""
    csv_file = BACKTEST_CSV
    header = ['Run_Time', 'Target_Period']
    for m in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        header.append(f'Params_{m}')
        if SEPARATE_POOL:
            for n in RED_NUM_LIST: header.append(f'Prob_{m}_R{n:02d}')
            for n in BLUE_NUM_LIST: header.append(f'Prob_{m}_B{n:02d}')
        else:
            for n in NUM_LIST: header.append(f'Prob_{m}_{n:02d}')
            
    row_data = {
        'Run_Time': run_time,
        'Target_Period': str(target_period) if target_period is not None else ''
    }
    for m in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
        if m in results:
            row_data[f'Params_{m}'] = json.dumps(models_config.get(m, {}))
            probs = results[m]
            probs = probs[0] if isinstance(probs, tuple) else probs
            if SEPARATE_POOL:
                for idx, n in enumerate(RED_NUM_LIST):
                    row_data[f'Prob_{m}_R{n:02d}'] = float(probs[idx])
                for idx, n in enumerate(BLUE_NUM_LIST):
                    row_data[f'Prob_{m}_B{n:02d}'] = float(probs[TOTAL_RED + idx])
            else:
                for idx, n in enumerate(NUM_LIST):
                    row_data[f'Prob_{m}_{n:02d}'] = float(probs[idx])
        else:
            row_data[f'Params_{m}'] = "{}"
            if SEPARATE_POOL:
                for n in RED_NUM_LIST: row_data[f'Prob_{m}_R{n:02d}'] = 0.0
                for n in BLUE_NUM_LIST: row_data[f'Prob_{m}_B{n:02d}'] = 0.0
            else:
                for n in NUM_LIST: row_data[f'Prob_{m}_{n:02d}'] = 0.0

    file_exists = os.path.isfile(csv_file)
    
    # Check for schema change (column count mismatch) to prevent corruption
    if file_exists and os.path.getsize(csv_file) > 0:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line:
                    existing_cols = len(first_line.split(','))
                    new_cols = len(header)
                    if existing_cols != new_cols:
                        print(f"⚠️ 检测到 CSV 表头字段数量变更 ({existing_cols} -> {new_cols})，正在备份旧文件并重建...")
                        backup_file = csv_file + f".bak_{int(time.time())}"
                        os.rename(csv_file, backup_file)
                        file_exists = False
        except Exception as e:
            print(f"⚠️ 检查 CSV 表头失败: {e}")

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writeheader()
        writer.writerow(row_data)

def main():
    for lottery_name in LOTTERIES_TO_RUN:
        conf = update_lottery_config(lottery_name)
        
        # Dynamically adjust LGBM parameters based on lottery type
        # Reset first to avoid carry-over
        MODEL_CONFIG['E'].pop('feature_fraction', None)
        MODEL_CONFIG['E'].pop('bagging_fraction', None)
        MODEL_CONFIG['E'].pop('is_unbalance', None)
        if lottery_name == '快乐8':
            MODEL_CONFIG['E']['feature_fraction'] = 0.8
            MODEL_CONFIG['E']['bagging_fraction'] = 0.8
        elif lottery_name in ['双色球', '超级大乐透']:
            MODEL_CONFIG['E']['is_unbalance'] = True
        
        # Dynamically adjust CatBoost parameters
        MODEL_CONFIG['F']['depth'] = 6 # Reset to default
        if lottery_name == '快乐8':
            MODEL_CONFIG['F']['depth'] = 4
        # Note: cat_features for 3D/PL3/PL5 would require feature name mapping
        # and is not implemented here as features are numeric.
            
        # Dynamically adjust EVT (Model H) parameters
        MODEL_CONFIG['H'].pop('sum_min', None)
        MODEL_CONFIG['H'].pop('sum_max', None)
        if lottery_name == '双色球':
            MODEL_CONFIG['H']['sum_min'] = 75
            MODEL_CONFIG['H']['sum_max'] = 135
        elif lottery_name == '快乐8':
            MODEL_CONFIG['H']['sum_min'] = 680
            MODEL_CONFIG['H']['sum_max'] = 940
            
        df = load_data()
        if df is None: continue
        conf = update_lottery_config(lottery_name, df=df)
        determine_stat_features(df)
        omission_df, next_omission = get_omission_matrix(df)
        full_df = pd.concat([df, omission_df], axis=1)
        active_methods = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] if args.method == 'all' else [args.method]
        
        # 1. Evaluation / Backtest
        eval_results, run_id = evaluate_methods(df, full_df, conf, test_size=args.eval_size, active_methods=active_methods)
        
        # 2. Final Prediction for Next Period
        try:
            next_period_id = int(df.iloc[-1]['期号']) + 1
        except:
            next_period_id = "Next"
            
        print("\n" + "="*65)
        print(f"🚀 正在预测下一期号码 (目标期号: {next_period_id})...")
        
        results = {}
        importances = {}
        _, current_omission = get_omission_matrix(df)
        
        for m in active_methods:
            res = run_prediction(df, m, full_df, current_omission, conf)
            if isinstance(res, tuple):
                results[m], importances[m] = res
            else:
                results[m] = res
            
        log_prediction_to_csv(run_id, next_period_id, results, MODEL_CONFIG)
        
        # 3. Display Results
        print("\n" + "="*65)
        print(f"🔮 {LOTTERY_NAME} 多模型综合分析报告 (历史回测期数: {args.eval_size})")
        print("="*65)
        print(f"最近期号: {df.iloc[-1]['期号']} | 预测目标: {next_period_id}")
        
        metrics = conf.get('eval_metrics', {"top_n_1": 6, "top_n_2": 10})
        n1, n2 = metrics['top_n_1'], metrics['top_n_2']
        
        for m in active_methods:
            probs = results[m]
            m_names = {'A': "统计相似度 SM", 'B': "机器学习 RF", 'C': "机器学习 XGB", 'D': "深度学习 LSTM", 'E': "机器学习 LGBM", 'F': "机器学习 CatBoost", 'G': "统计模型 HMM", 'H': "极值理论 EVT", 'I': "遗传算法 GA", 'J': "泊松分布 Poisson"}
            print(f"\n--- {m_names[m]} ---")
            
            # History calculation display
            if SEPARATE_POOL:
                # h = (hr1, hr2, hb1) from evaluate_methods
                history_str = ", ".join([f"R:{h[0]}/{RED_COUNT}|B:{h[2]}/{BLUE_COUNT}" for h in eval_results[m]['history']])
                print(f"历史回测平均命中率: 红球(Top {n1}/{n2}) {eval_results[m]['avg_r1']:.2%}/{eval_results[m]['avg_r2']:.2%} | 蓝球 {eval_results[m]['avg_b1']:.2%}")
                print(f"回测详情 (R/B): [{history_str}]")
                
                red_p, blue_p = probs[:TOTAL_RED], probs[TOTAL_RED:]
                top_red = [RED_NUM_LIST[i] for i in red_p.argsort()[::-1][:n1]]
                top_blue = [BLUE_NUM_LIST[i] for i in blue_p.argsort()[::-1][:max(1, BLUE_COUNT)]]
                print(f"最佳推荐: 红球 {sorted(top_red)} | 蓝球 {sorted(top_blue)}")
            else:
                # h = (h1, h2)
                history_str = ", ".join([f"{h[0]}/{RED_COUNT+BLUE_COUNT}|{h[1]}/{RED_COUNT+BLUE_COUNT}" for h in eval_results[m]['history']])
                print(f"历史回测平均命中率 (Top {n1}/{n2}): {eval_results[m]['avg_r1']:.2%}/{eval_results[m]['avg_r2']:.2%} [{history_str}]")
                
                top_indices = probs.argsort()[::-1]
                top_nums = [NUM_LIST[i] for i in top_indices]
                if LOTTERY_NAME == "七乐彩":
                    print(f"最佳 8 推荐 (7红 + 1蓝): {sorted(top_nums[:7])} | 蓝球: {top_nums[7]:02d}")
                else:
                    print(f"最佳 {n1} 推荐: {sorted(top_nums[:n1])}")
                    
            if m in importances and importances[m] is not None:
                plot_filename = f"data/{conf['code']}_importance_{m.lower()}.png"
                # For separate pool, ML models' importance is from red ball training
                f_names = get_feature_names(pool_type='red') if (SEPARATE_POOL and m in ['B', 'C', 'E', 'F']) else get_feature_names()
                plot_importance(importances[m], f_names, m_names[m], plot_filename)
    
        if args.method == 'all':
            # Ensemble (Simplified version)
            ensemble_scores = np.zeros(TOTAL_NUMBERS)
            # 调整权重计算：移除 15% 的硬性门槛，让所有模型根据自身表现贡献权重。
            # 表现差的模型 (avg_r2 接近 0) 权重自然会很低，这是一种更动态的优胜劣汰。
            qualified = active_methods

            if qualified:
                for m in active_methods:
                    p = results[m]
                    weight = eval_results[m]['avg_r2']
                    if p.max() > p.min():
                        ensemble_scores += ((p - p.min()) / (p.max() - p.min())) * weight
                
                print("\n" + "="*65)
                print("🏆 多模型标准概率综合推荐 (Ensemble)")
                print("="*65)
                if SEPARATE_POOL:
                    r_p, b_p = ensemble_scores[:TOTAL_RED], ensemble_scores[TOTAL_RED:]
                    top_red = [RED_NUM_LIST[i] for i in r_p.argsort()[::-1][:n1]]
                    top_blue = [BLUE_NUM_LIST[i] for i in b_p.argsort()[::-1][:BLUE_COUNT]]
                    print(f"最终组合: 红球 {sorted(top_red)} | 蓝球 {sorted(top_blue)}")
                else:
                    top_nums = [NUM_LIST[i] for i in ensemble_scores.argsort()[::-1][:n2]]
                    if LOTTERY_NAME == "七乐彩":
                        print(f"最终组合 (7红 + 1蓝): {sorted(top_nums[:7])} | 蓝球: {top_nums[7]:02d}")
                    else:
                        print(f"最终组合: {sorted(top_nums[:n1])}")
                
                ens_res = eval_results['Ensemble']
                if SEPARATE_POOL:
                    print(f"🏆 综合平均命中率: 红球(Top {n1}/{n2}) {ens_res['avg_r1']:.2%}/{ens_res['avg_r2']:.2%} | 蓝球 {ens_res['avg_b1']:.2%}")
                else:
                    print(f"🏆 综合平均命中率: {ens_res['avg_r1']:.2%}/{ens_res['avg_r2']:.2%}")
                print("="*65)

if __name__ == "__main__":
    main()
