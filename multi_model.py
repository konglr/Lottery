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

# --- Heavy Imports (might be slow) ---
import matplotlib
matplotlib.use('Agg') # 关键：禁用 GUI 后端，防止在无显示器环境下挂起
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import LOTTERY_CONFIG

# --- Reproducibility ---
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Global Configuration ---
def init_config():
    parser = argparse.ArgumentParser(description="Multi-Model Lottery Prediction")
    parser.add_argument("--lottery", type=str, default="all", help="彩票名称，支持多个以逗号分隔或输入 'all'") #"双色球", "七星彩", , "排列三", "排列五" "超级大乐透", "快乐8", "福彩3D",“七乐彩”
    parser.add_argument("--method", type=str, choices=['A', 'B', 'C', 'D', 'all'], default='all', help="分析方法")
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
        'n_estimators': 1000,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 10,
        'max_features': 'log2',
        'random_state': 42,
        'n_jobs': -1
    },
    'C': {
        'n_estimators': 500,
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
        'epochs': 50
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

def extract_features(df, window_data, next_omission):
    """Unified feature extraction for ML models."""
    features = []
    # A. Hot/Cold (Freq in window)
    if SEPARATE_POOL:
        red_balls = window_data[RED_COLS].values.flatten().astype(int)
        blue_balls = window_data[BLUE_COLS].values.flatten().astype(int)
        # Offset frequencies for bincount based on range start
        r_freq = np.zeros(TOTAL_RED)
        b_freq = np.zeros(TOTAL_BLUE)
        for b in red_balls:
            if RED_RANGE[0] <= b <= RED_RANGE[1]: r_freq[b - RED_RANGE[0]] += 1
        for b in blue_balls:
            if BLUE_RANGE[0] <= b <= BLUE_RANGE[1]: b_freq[b - BLUE_RANGE[0]] += 1
        features.extend(r_freq)
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

def get_feature_names():
    """Map indices to human-readable names."""
    names = []
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

# --- Pattern Feature Helpers ---

def count_consecutive(numbers):
    """Count consecutive number pairs in a draw."""
    sorted_nums = sorted(numbers)
    count = 0
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i+1] - sorted_nums[i] == 1:
            count += 1
    return count

def count_neighbor(current_nums, prev_draw_nums):
    """Count numbers that are neighbors (+/- 1) of previous draw."""
    count = 0
    for num in current_nums:
        for prev_num in prev_draw_nums:
            if abs(num - prev_num) == 1:
                count += 1
                break
    return count

def count_repeat(current_nums, recent_history):
    """Count numbers that repeat from recent history (last 3 draws)."""
    recent_nums = set()
    for draw in recent_history:
        recent_nums.update(draw)
    return len(set(current_nums) & recent_nums)

def count_jumps(numbers):
    """Count multi-step jump patterns (2-jump to 6-jump)."""
    sorted_nums = sorted(numbers)
    jump_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for i in range(len(sorted_nums) - 1):
        diff = sorted_nums[i+1] - sorted_nums[i]
        if diff in jump_counts:
            jump_counts[diff] += 1
    return sum(jump_counts.values())

# --- Method A: Statistical Similarity (Property Range Prediction) ---

def predict_similarity(df):
    """
    1. Find similar fragments in history (Pattern Matching).
    2. Analyze 'Property Range' (Sum, AC, etc.) of the following draw.
    3. Use property distributions to weight frequencies.
    4. Enhanced with pattern features: consecutive, neighbor, repeat, jump.
    """
    logging.info("Method A: 执行模式匹配与属性范围分析（含模式特征）...")
    target_window = df.tail(WINDOW_SIZE)
    target_nums = set(target_window[RED_COLS].values.flatten())
    target_sum = target_window['和值'].mean()
    target_ac = target_window['AC'].mean()
    
    # Omission similarity
    omission_cols = [f'Omission_{i}' for i in NUM_LIST]
    has_omission = all(c in df.columns for c in omission_cols)
    if has_omission:
        target_omission = target_window.iloc[-1][omission_cols].values.astype(float)
    
    # Calculate target pattern features
    target_last_draw = target_window.iloc[-1][RED_COLS].values
    target_consecutive = count_consecutive(target_last_draw)
    target_recent_history = [target_window.iloc[i][RED_COLS].values for i in range(max(0, len(target_window)-3), len(target_window))]
    
    history_matches = []
    # Search history based on CONFIG
    conf = MODEL_CONFIG['A']
    search_limit = min(conf['search_limit'], len(df) - WINDOW_SIZE - 2)
    for i in range(max(0, len(df) - search_limit), len(df) - WINDOW_SIZE - 1):
        window = df.iloc[i : i + WINDOW_SIZE]
        next_row = df.iloc[i + WINDOW_SIZE]
        
        # Basic Similarity Factors
        win_nums = set(window[RED_COLS].values.flatten())
        overlap = len(target_nums & win_nums)
        sum_err = abs(window['和值'].mean() - target_sum)
        ac_err = abs(window['AC'].mean() - target_ac)
        
        # Pattern Similarity Factors
        win_last_draw = window.iloc[-1][RED_COLS].values
        win_consecutive = count_consecutive(win_last_draw)
        consecutive_match = 1.0 / (1.0 + abs(target_consecutive - win_consecutive))
        
        # Neighbor: compare next_row with window's last draw
        neighbor_count = count_neighbor(next_row[RED_COLS].values, win_last_draw)
        
        # Repeat: compare next_row with window's recent history
        win_recent_history = [window.iloc[j][RED_COLS].values for j in range(max(0, len(window)-3), len(window))]
        repeat_count = count_repeat(next_row[RED_COLS].values, win_recent_history)
        
        # Jump patterns
        jump_count = count_jumps(next_row[RED_COLS].values)
        target_jump_count = count_jumps(target_last_draw)
        jump_match = 1.0 / (1.0 + abs(target_jump_count - jump_count))
        
        # Combined score using configurable weights
        score = (overlap * conf['weights']['overlap'] - 
                 sum_err * conf['weights']['sum'] - 
                 ac_err * conf['weights']['ac'] +
                 consecutive_match * conf['weights']['consecutive'] +
                 neighbor_count * conf['weights']['neighbor'] +
                 repeat_count * conf['weights']['repeat'] +
                 jump_match * conf['weights']['jump'])
        
        if has_omission:
            win_omission = window.iloc[-1][omission_cols].values.astype(float)
            omiss_err = np.mean(np.abs(target_omission - win_omission))
            omiss_match = 1.0 / (1.0 + omiss_err)
            score += omiss_match * conf['weights']['omission']
        
        history_matches.append({
            'score': score,
            'draw': next_row[RED_COLS].values,
            'sum_val': next_row['和值'],
            'ac_val': next_row['AC'],
            'span': next_row['跨度']
        })
    
    # Take top matches based on CONFIG
    history_matches.sort(key=lambda x: x['score'], reverse=True)
    top_matches = history_matches[:conf['top_matches']]
    
    # Analyze Property Ranges (Targeting next draw)
    avg_next_sum = np.mean([m['sum_val'] for m in top_matches])
    avg_next_ac = np.mean([m['ac_val'] for m in top_matches])
    
    logging.info(f"模式匹配预测下期属性: 预计和值(avg)={avg_next_sum:.1f}, AC(avg)={avg_next_ac:.1f}")
    
    # Calculate Probabilities
    pred_counts = np.zeros(TOTAL_NUMBERS)
    for m in top_matches:
        for n in m['draw']:
            try:
                idx = NUM_LIST.index(int(n))
                pred_counts[idx] += 1.0
            except ValueError:
                pass
                
    probs = pred_counts / pred_counts.sum() if pred_counts.sum() > 0 else np.zeros(TOTAL_NUMBERS)
    return probs

# --- Method B: RandomForest (Classification) ---

def train_predict_rf(X, y, final_feature):
    logging.info("Method B: 训练 RandomForest 分类模型...")
    # Parameters from Centralized CONFIG
    conf = MODEL_CONFIG['B']
    model = RandomForestClassifier(
        n_estimators=conf['n_estimators'], 
        max_depth=conf.get('max_depth'),
        min_samples_split=conf.get('min_samples_split', 2),
        min_samples_leaf=conf.get('min_samples_leaf', 1),
        max_features=conf.get('max_features', 'sqrt'),
        class_weight='balanced_subsample',
        random_state=conf['random_state'], 
        n_jobs=conf['n_jobs']
    )
    model.fit(X, y)
    
    # Predict probabilities for final_feature
    raw_probs = model.predict_proba(final_feature.reshape(1, -1))
    probs = np.zeros(TOTAL_NUMBERS)
    
    # Robust probability extraction (handling potential missing classes '0' or '1' in training)
    for i in range(TOTAL_NUMBERS):
        if hasattr(raw_probs[i], "shape") and raw_probs[i].shape[1] == 2:
            probs[i] = raw_probs[i][0, 1]
        elif hasattr(raw_probs[i], "shape") and raw_probs[i].shape[1] == 1:
            # If only one class exists in training data for this number
            if model.classes_[i][0] == 1:
                probs[i] = 1.0
            else:
                probs[i] = 0.0
            
    return probs, model.feature_importances_

# --- Method C: XGBoost (Classification) ---

def train_predict_xgb(X, y, final_feature):
    logging.info("Method C: 训练 XGBoost 分类模型...")
    probs = np.zeros(TOTAL_NUMBERS)
    conf = MODEL_CONFIG['C']
    all_importances = []
    # Train 33 independent binary classifiers with progress bar
    for i in tqdm(range(TOTAL_NUMBERS), desc="XGBoost 训练", leave=False):
        if len(np.unique(y[:, i])) > 1:
            clf = xgb.XGBClassifier(
                n_estimators=conf['n_estimators'],
                max_depth=conf['max_depth'],
                learning_rate=conf['learning_rate'],
                subsample=conf.get('subsample', 1.0),
                colsample_bytree=conf.get('colsample_bytree', 1.0),
                gamma=conf.get('gamma', 0),
                scale_pos_weight=conf.get('scale_pos_weight', 1),
                random_state=conf['random_state'],
                n_jobs=conf['n_jobs'],
                eval_metric=conf['eval_metric']
            )
            clf.fit(X, y[:, i])
            probs[i] = clf.predict_proba(final_feature.reshape(1, -1))[0, 1]
            all_importances.append(clf.feature_importances_)
        else:
            probs[i] = 1.0 if y[0, i] == 1 else 0.0
            
    avg_importance = np.mean(all_importances, axis=0) if all_importances else np.zeros(X.shape[1])
    return probs, avg_importance

# --- Method D: LSTM (PyTorch) ---

class LotteryLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LotteryLSTM, self).__init__()
        
        # 1. Embedding Layer: maps ball IDs (0 to vocab_size-1) to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=num_layers, 
                            dropout=dropout if num_layers > 1 else 0, 
                            batch_first=True)
        
        # 3. Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # 4. Output Layer: Logits for all possible IDs
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeds = self.embedding(x) # (batch_size, seq_len, embed_dim)
        
        lstm_out, _ = self.lstm(embeds) # (batch_size, seq_len, hidden_dim)
        
        # Take the last time step
        last_step = lstm_out[:, -1, :]
        
        out = self.dropout(last_step)
        logits = self.fc(out)
        return logits

def train_predict_lstm(df):
    try:
        torch.set_num_threads(1)
        logging.info(f"Method D: 训练 LSTM 走势捕捉模型 (内部ID映射, 词表大小={TOTAL_NUMBERS})...")
        
        # 1. Data Preparation using Internal IDs
        n_rows = len(df)
        ball_seqs = []
        for i in range(n_rows):
            ids = []
            # Red balls
            row_red = [int(n) for n in df.iloc[i][RED_COLS].values if pd.notna(n)]
            for val in row_red:
                if val in RED_NUM_LIST:
                    ids.append(RED_NUM_LIST.index(val))
            # Blue balls (with offset)
            if SEPARATE_POOL:
                row_blue = [int(n) for n in df.iloc[i][BLUE_COLS].values if pd.notna(n)]
                for val in row_blue:
                    if val in BLUE_NUM_LIST:
                        ids.append(TOTAL_RED + BLUE_NUM_LIST.index(val))
            ball_seqs.append(sorted(ids))
            
        X_list, y_list = [], []
        for i in range(WINDOW_SIZE, n_rows):
            x_seq = []
            for j in range(i - WINDOW_SIZE, i):
                x_seq.extend(ball_seqs[j])
            
            for target_id in ball_seqs[i]:
                X_list.append(x_seq)
                y_list.append(target_id)
                
        if not X_list:
            return np.zeros(TOTAL_NUMBERS)
            
        X_train = torch.LongTensor(X_list)
        y_train = torch.LongTensor(y_list)
        
        # 2. Model Initialization
        vocab_size = TOTAL_NUMBERS
        conf_d = MODEL_CONFIG['D']
        model = LotteryLSTM(
            vocab_size, 
            conf_d['embedding_dim'], 
            conf_d['hidden_dim'], 
            conf_d['num_layers'], 
            conf_d['dropout']
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=conf_d['lr'])
        
        # 3. Training
        epochs = conf_d['epochs']
        model.train()
        pbar = tqdm(range(epochs), desc="LSTM 训练", leave=False)
        for _ in pbar:
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            if torch.isnan(loss): break
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # 4. Prediction
        model.eval()
        last_x = []
        for j in range(n_rows - WINDOW_SIZE, n_rows):
            last_x.extend(ball_seqs[j])
        
        last_x_tensor = torch.LongTensor([last_x])
        with torch.no_grad():
            logits = model(last_x_tensor)
            final_probs = torch.softmax(logits, dim=1).numpy()[0]
            
        return final_probs
    except Exception as e:
        logging.error(f"Method D (LSTM) Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return np.zeros(TOTAL_NUMBERS)

def run_prediction(df, method, full_df, next_omission, conf=None):
    if method == 'A':
        return predict_similarity(full_df)
    elif method in ['B', 'C']:
        X_list, y_list = [], []
        # Find all omission columns
        omiss_cols = [c for c in full_df.columns if 'Omiss' in c]
        
        for i in range(WINDOW_SIZE, len(full_df)):
            win = full_df.iloc[i-WINDOW_SIZE : i]
            feat = extract_features(full_df, win, full_df.iloc[i][omiss_cols].values)
            
            # Construct Target Vector
            target = np.zeros(TOTAL_NUMBERS)
            if SEPARATE_POOL:
                # Red Segment
                for n in full_df.loc[i, RED_COLS].values:
                    try:
                        idx = RED_NUM_LIST.index(int(n))
                        target[idx] = 1
                    except ValueError: pass
                # Blue Segment
                for n in full_df.loc[i, BLUE_COLS].values:
                    try:
                        idx = BLUE_NUM_LIST.index(int(n))
                        target[TOTAL_RED + idx] = 1
                    except ValueError: pass
            else:
                for n in full_df.loc[i, RED_COLS].values:
                    try:
                        idx = RED_NUM_LIST.index(int(n))
                        target[idx] = 1
                    except ValueError: pass
            X_list.append(feat)
            y_list.append(target)
            
        last_win = full_df.iloc[-WINDOW_SIZE:]
        final_feat = extract_features(full_df, last_win, next_omission)
        if method == 'B':
            return train_predict_rf(np.array(X_list), np.array(y_list), final_feat)
        else:
            return train_predict_xgb(np.array(X_list), np.array(y_list), final_feat)
    elif method == 'D':
        return train_predict_lstm(df)
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
            for m in ['A', 'B', 'C', 'D']:
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
        
        actual_red = set(df.iloc[i][RED_COLS].values.astype(int))
        actual_blue = set(df.iloc[i][BLUE_COLS].values.astype(int)) if SEPARATE_POOL else set()
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
    for m in ['A', 'B', 'C', 'D']:
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
    for m in ['A', 'B', 'C', 'D']:
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
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writeheader()
        writer.writerow(row_data)

def main():
    for lottery_name in LOTTERIES_TO_RUN:
        conf = update_lottery_config(lottery_name)
        df = load_data()
        if df is None: continue
        conf = update_lottery_config(lottery_name, df=df)
        determine_stat_features(df)
        omission_df, next_omission = get_omission_matrix(df)
        full_df = pd.concat([df, omission_df], axis=1)
        active_methods = ['A', 'B', 'C', 'D'] if args.method == 'all' else [args.method]
        
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
            m_names = {'A': "统计相似度", 'B': "机器学习 RF", 'C': "机器学习 XGB", 'D': "深度学习 LSTM"}
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
                    
            if m in importances:
                plot_filename = f"data/{conf['code']}_importance_{m.lower()}.png"
                plot_importance(importances[m], get_feature_names(), m_names[m], plot_filename)
    
        if args.method == 'all':
            # Ensemble (Simplified version)
            ensemble_scores = np.zeros(TOTAL_NUMBERS)
            # Use avg_r2 (Red Top N2 hit rate) as the qualification metric
            qualified = [m for m in active_methods if eval_results[m]['avg_r2'] >= 0.15] 
            if qualified:
                for m in qualified:
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
