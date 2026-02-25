import pandas as pd
import numpy as np
import logging
import argparse
import sys
import csv
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Reproducibility ---
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- CONFIG ---
DATA_FILE = "data/双色球_lottery_data.csv"
RED_COLS = [f'红球{i}' for i in range(1, 7)]
WINDOW_SIZE = 4
TOTAL_NUMBERS = 33
DEFAULT_EVAL_SIZE = 10

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
        'min_samples_leaf': 5,
        'max_features': 'log2',
        'random_state': 42,
        'n_jobs': -1
    },
    'C': {
        'n_estimators': 500,
        'max_depth': 8,
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
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'lr': 0.005,
        'epochs': 100
    }
}

# --- 1. Data Loading & Preprocessing ---

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
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def get_omission_matrix(df):
    n_rows = len(df)
    omission_matrix = np.zeros((n_rows, TOTAL_NUMBERS), dtype=int)
    current_omission = np.zeros(TOTAL_NUMBERS, dtype=int)
    for i in range(n_rows):
        omission_matrix[i] = current_omission.copy()
        current_draw = set(df.loc[i, RED_COLS].values)
        for num in range(1, TOTAL_NUMBERS + 1):
            if num in current_draw:
                current_omission[num-1] = 0
            else:
                current_omission[num-1] += 1
    cols = [f'Omission_{i}' for i in range(1, TOTAL_NUMBERS + 1)]
    return pd.DataFrame(omission_matrix, columns=cols), current_omission

def extract_features(df, window_data, next_omission):
    """Unified feature extraction for ML models."""
    features = []
    # A. Hot/Cold (Freq in window)
    all_balls = window_data[RED_COLS].values.flatten().astype(int)
    freq = np.bincount(all_balls, minlength=TOTAL_NUMBERS+1)[1:]
    features.extend(freq)
    
    # B. Next Omission State
    features.extend(next_omission)
    
    # C. Stats
    stat_cols = [
        '和值', 'AC', '跨度', '奇数', '大号', '重号', '邻号', 
        '二连', '三连', '二跳', '三跳', '四跳', '五跳', '六跳'
    ]
    w_data = window_data.copy()
    for col in stat_cols:
        if col not in w_data.columns: w_data[col] = 0
    
    features.extend(w_data[stat_cols].mean().values)
    features.extend(w_data[stat_cols].iloc[-1].values)
    
    # D. Tail Frequency
    all_tails = [int(n) % 10 for n in all_balls if n > 0]
    tail_counts = np.bincount(all_tails, minlength=10)
    features.extend(tail_counts)
    
    return np.array(features)

def get_feature_names():
    """Map indices to human-readable names (English-safe for plots)."""
    names = []
    # A. Freq
    names.extend([f"Freq_{i:02d}" for i in range(1, 34)])
    # B. Omission
    names.extend([f"Omission_{i:02d}" for i in range(1, 34)])
    # C. Stats (Translated for plots)
    stat_map = {
        '和值': 'Sum', 'AC': 'AC', '跨度': 'Span', '奇数': 'OddCnt', 
        '大号': 'BigCnt', '重号': 'Repeat', '邻号': 'Neighbor', 
        '二连': 'Consec2', '三连': 'Consec3', '二跳': 'Jump2', 
        '三跳': 'Jump3', '四跳': 'Jump4', '五跳': 'Jump5', '六跳': 'Jump6'
    }
    stat_cols = [
        '和值', 'AC', '跨度', '奇数', '大号', '重号', '邻号', 
        '二连', '三连', '二跳', '三跳', '四跳', '五跳', '六跳'
    ]
    names.extend([f"Avg_{stat_map[c]}" for c in stat_cols])
    names.extend([f"Last_{stat_map[c]}" for c in stat_cols])
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
    omission_cols = [f'Omission_{i}' for i in range(1, TOTAL_NUMBERS + 1)]
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
            if 1 <= int(n) <= 33:
                # Basic frequency increment
                weight = 1.0
                # Could add property weighting here (e.g. if num fits the sum range better)
                pred_counts[int(n)-1] += weight
                
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

class SSQLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(SSQLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

def train_predict_lstm(df):
    try:
        torch.set_num_threads(1) # Potential fix for Mac hangs
        logging.info("Method D: 训练 LSTM 走势捕捉模型...")
        # Vectorized creation of seq_data
        n_rows = len(df)
        seq_data = np.zeros((n_rows, TOTAL_NUMBERS), dtype=np.float32)
        red_vals = df[RED_COLS].values.astype(int)
        for i in range(n_rows):
            for n in red_vals[i]:
                if 1 <= n <= 33:
                    seq_data[i, n-1] = 1.0
        
        # Prepare X, y sequences
        n_sequences = n_rows - WINDOW_SIZE
        if n_sequences <= 0:
            logging.warning("数据量不足, 无法训练 LSTM.")
            return np.zeros(TOTAL_NUMBERS)
            
        logging.info(f"LSTM 数据准备完成: 序列数={n_sequences}")
        X_seq_np = np.zeros((n_sequences, WINDOW_SIZE, TOTAL_NUMBERS), dtype=np.float32)
        y_seq_np = np.zeros((n_sequences, TOTAL_NUMBERS), dtype=np.float32)
        
        for i in range(n_sequences):
            X_seq_np[i] = seq_data[i : i + WINDOW_SIZE]
            y_seq_np[i] = seq_data[i + WINDOW_SIZE]
            
        # Validate target range
        if not ((y_seq_np >= 0).all() and (y_seq_np <= 1).all()):
            logging.error("LSTM 目标数据超出 [0, 1] 范围.")
            return np.zeros(TOTAL_NUMBERS)

        X_seq = torch.from_numpy(X_seq_np)
        y_seq = torch.from_numpy(y_seq_np)
        
        # Hyperparameters from CONFIG
        conf = MODEL_CONFIG['D']
        conf = MODEL_CONFIG['D']
        model = SSQLSTM(
            TOTAL_NUMBERS, 
            conf['hidden_dim'], 
            TOTAL_NUMBERS, 
            num_layers=conf.get('num_layers', 1), 
            dropout=conf.get('dropout', 0)
        )
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'])
        
        epochs = conf['epochs']
        model.train()
        logging.info(f"开始 LSTM 训练 ({epochs} epochs)...")
        pbar = tqdm(range(epochs), desc="LSTM 训练", leave=False)
        for _ in pbar:
            optimizer.zero_grad()
            outputs = model(X_seq)
            loss = criterion(outputs, y_seq)
            if torch.isnan(loss):
                logging.warning(f"LSTM loss is NaN, stopping early.")
                break
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        model.eval()
        last_seq = torch.from_numpy(seq_data[-WINDOW_SIZE:]).unsqueeze(0)
        with torch.no_grad():
            probs = model(last_seq).numpy()[0]
        return probs
    except Exception as e:
        logging.error(f"Method D (LSTM) Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return np.zeros(TOTAL_NUMBERS)

def run_prediction(df, method, full_df, next_omission):
    if method == 'A':
        return predict_similarity(full_df)
    elif method in ['B', 'C']:
        X_list, y_list = [], []
        for i in range(WINDOW_SIZE, len(full_df)):
            win = full_df.iloc[i-WINDOW_SIZE : i]
            feat = extract_features(full_df, win, full_df.iloc[i][[f'Omission_{k}' for k in range(1, 34)]].values)
            target = np.zeros(TOTAL_NUMBERS)
            for n in full_df.loc[i, RED_COLS].values:
                if 1 <= int(n) <= 33: target[int(n)-1] = 1
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

def evaluate_methods(df, full_df, test_size=10, active_methods=['A', 'B', 'C', 'D']):
    """Perform backtesting for all selected methods."""
    logging.info(f"开始历史回测分析 (最近 {test_size} 期)...")
    hit_counts_10 = {m: 0 for m in active_methods}
    hit_counts_6 = {m: 0 for m in active_methods}
    hit_history = {m: [] for m in active_methods}
    
    ensemble_hits_10 = 0
    ensemble_hits_6 = 0
    ensemble_history = []
    
    union_hits = 0
    union_size_sum = 0
    union_history = []
    
    voting_hits_6 = 0
    voting_history = []
    
    
    # Check if backtest.csv exists to write header
    csv_file = 'backtest.csv'
    file_exists = os.path.isfile(csv_file)
    
    # Prepare CSV Header
    header = ['Run_Time', 'Target_Period']
    for m in ['A', 'B', 'C', 'D']:
        header.append(f'Params_{m}')
        for n in range(1, TOTAL_NUMBERS + 1):
            header.append(f'Prob_{m}_{n:02d}')
            
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writerow(header)

    # Overall backtest progress bar
    current_run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i in tqdm(range(len(df) - test_size, len(df)), desc="历史回测进度"):
        train_df = df.iloc[:i].reset_index(drop=True)
        train_full_df = full_df.iloc[:i].reset_index(drop=True)
        
        # Get target info
        target_period = df.iloc[i]['期号']
        actual_draw = set(df.iloc[i][RED_COLS].values.astype(int))
        _, current_omission = get_omission_matrix(train_df)
        
        # Prepare Log Row
        row_data = {
            'Run_Time': current_run_time,
            'Target_Period': target_period
        }
        
        current_probs = {} # Store probs for logging
        
        for m in active_methods:
            res = run_prediction(train_df, m, train_full_df, current_omission)
            probs = res[0] if isinstance(res, tuple) else res
            sorted_idx = probs.argsort()[::-1]
            
            # Top 10
            top_10 = sorted_idx[:10]
            hits_10 = len(actual_draw & set([int(idx+1) for idx in top_10]))
            hit_counts_10[m] += hits_10
            
            # Top 6
            top_6 = sorted_idx[:6]
            hits_6 = len(actual_draw & set([int(idx+1) for idx in top_6]))
            hit_counts_6[m] += hits_6
            
            hit_history[m].append((hits_6 / 6.0, hits_10 / 6.0))
            
            # Save probs for logging
            current_probs[m] = probs
            
        # Log to CSV
        log_row = [row_data['Run_Time'], row_data['Target_Period']]
        
        # Add Params and Probs for A, B, C, D (fill with 0 if not active)
        for m in ['A', 'B', 'C', 'D']:
            # Params
            if m in active_methods:
                log_row.append(json.dumps(MODEL_CONFIG[m], ensure_ascii=False))
            else:
                log_row.append("{}")
            
            # Probs
            if m in current_probs:
                p = current_probs[m]
                log_row.extend([f"{val:.4f}" for val in p])
            else:
                log_row.extend(['0.0000'] * TOTAL_NUMBERS)
                
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(log_row)
        
        # --- Ensemble Calculation for this Period ---
        # 1. Normalize probabilities for each method
        # 2. Filter methods that have a "rolling" good hit rate? 
        #    For complexity, we will just use all active methods and weight by their *current* period performance? No.
        #    We will use a simplified ensemble: Average of Min-Max scaled probs of ALL active methods.
        
        ens_scores = np.zeros(TOTAL_NUMBERS)
        valid_methods = 0
        
        for m in active_methods:
            p = res[0] if isinstance(res, tuple) else res
            p_min, p_max = p.min(), p.max()
            if p_max > p_min:
                # Standardize
                std_p = (p - p_min) / (p_max - p_min)
                ens_scores += std_p
                valid_methods += 1
        
        if valid_methods > 0:
            ens_scores /= valid_methods
            
            # Check hits
            ens_sorted_idx = ens_scores.argsort()[::-1]
            ens_top_10 = ens_sorted_idx[:10]
            ens_hits_10 = len(actual_draw & set([int(idx+1) for idx in ens_top_10]))
            ensemble_hits_10 += ens_hits_10
            
            ens_top_6 = ens_sorted_idx[:6]
            ens_hits_6 = len(actual_draw & set([int(idx+1) for idx in ens_top_6]))
            ensemble_hits_6 += ens_hits_6
            
            ensemble_history.append((ens_hits_6 / 6.0, ens_hits_10 / 6.0))
            
            # --- Union Ensemble (Top 10 of All Qualified) ---
            # Union of all active methods' Top 10
            union_set = set()
            voting_dict = {} # Number -> (Count, Sum_Std_Prob)
            
            for m in active_methods:
                p = current_probs[m]
                p_min, p_max = p.min(), p.max()
                std_p = (p - p_min) / (p_max - p_min) if p_max > p_min else p
                
                m_top_10 = p.argsort()[-10:][::-1]
                for idx in m_top_10:
                    num = int(idx + 1)
                    union_set.add(num)
                    
                    if num not in voting_dict:
                        voting_dict[num] = [0, 0.0]
                    voting_dict[num][0] += 1
                    voting_dict[num][1] += std_p[idx]
            
            # Calculate Union Hits
            u_hits = len(actual_draw & union_set)
            union_hits += u_hits
            union_size_sum += len(union_set)
            union_history.append((u_hits, len(union_set)))
            
            # --- Voting Ensemble (Top 6 most frequent) ---
            # Sort by Count (desc), then Sum_Std_Prob (desc)
            sorted_votes = sorted(voting_dict.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
            voting_top_6 = [x[0] for x in sorted_votes[:6]]
            v_hits = len(actual_draw & set(voting_top_6))
            voting_hits_6 += v_hits
            voting_history.append(v_hits / 6.0)
            
        else:
            ensemble_history.append((0, 0))
            union_history.append((0, 0))
            voting_history.append(0)

    results = {
        m: {
            'avg_10': hit_counts_10[m] / (test_size * 6),
            'avg_6': hit_counts_6[m] / (test_size * 6),
            'history': hit_history[m]
        } for m in active_methods
    }
    
    # Add Ensemble Results
    results['Ensemble'] = {
        'avg_10': ensemble_hits_10 / (test_size * 6),
        'avg_6': ensemble_hits_6 / (test_size * 6),
        'history': ensemble_history
    }
    
    results['Union'] = {
        'avg_hits': union_hits / test_size,
        'avg_size': union_size_sum / test_size,
        'history': union_history
    }
    
    results['Voting'] = {
        'avg_6': voting_hits_6 / (test_size * 6),
        'history': voting_history
    }
    
    return results, current_run_time

def log_prediction_to_csv(run_time, target_period, results, models_config):
    """Logs a single prediction row to backtest.csv"""
    csv_file = 'backtest.csv'
    header = ['Run_Time', 'Target_Period']
    for m in ['A', 'B', 'C', 'D']:
        header.append(f'Params_{m}')
        for n in range(1, TOTAL_NUMBERS + 1):
            header.append(f'Prob_{m}_{n:02d}')
            
    row_data = {
        'Run_Time': run_time,
        'Target_Period': int(float(target_period)) if target_period and str(target_period).replace('.','').isdigit() else target_period
    }
    
    for m in ['A', 'B', 'C', 'D']:
        if m in results:
            row_data[f'Params_{m}'] = json.dumps(models_config.get(m, {}))
            probs = results[m]
            probs = probs[0] if isinstance(probs, tuple) else probs
            for n in range(1, TOTAL_NUMBERS + 1):
                row_data[f'Prob_{m}_{n:02d}'] = float(probs[n-1])
        else:
            row_data[f'Params_{m}'] = "{}"
            for n in range(1, TOTAL_NUMBERS + 1):
                row_data[f'Prob_{m}_{n:02d}'] = 0.0

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writeheader()
        writer.writerow(row_data)

def main():
    parser = argparse.ArgumentParser(description="SSQ Multi-Model Prediction")
    parser.add_argument("--method", type=str, choices=['A', 'B', 'C', 'D', 'all'], default='all', help="分析方法: A-统计, B-RF, C-XGB, D-LSTM, all-全对比")
    parser.add_argument("--eval_size", type=int, default=DEFAULT_EVAL_SIZE, help=f"回测期数 (默认 {DEFAULT_EVAL_SIZE})")
    args = parser.parse_args()
    
    df = load_data()
    if df is None: return

    omission_df, next_omission = get_omission_matrix(df)
    full_df = pd.concat([df, omission_df], axis=1)
    
    active_methods = ['A', 'B', 'C', 'D'] if args.method == 'all' else [args.method]
    
    # 1. Evaluation / Backtest
    eval_results, run_id = evaluate_methods(df, full_df, test_size=args.eval_size, active_methods=active_methods)
    
    # 2. Final Prediction for Next Period
    next_period_id = int(df.iloc[-1]['期号']) + 1 # Assuming period IDs are sequential integers
    print("\n" + "="*65)
    print(f"🚀 正在预测下一期号码 (目标期号: {next_period_id})...")
    
    results = {}
    importances = {}
    qualified_methods = active_methods 
    methods_to_run = active_methods
    _, current_omission = get_omission_matrix(df)
    
    for m in qualified_methods:
        res = run_prediction(df, m, full_df, current_omission)
        if isinstance(res, tuple):
            results[m], importances[m] = res
        else:
            results[m] = res
        
    # Log Final Prediction to CSV
    log_prediction_to_csv(run_id, next_period_id, results, MODEL_CONFIG)
    
    # 3. Display Results
    # The original loop for methods_to_run is now replaced by the above final prediction logic
    # and the display logic below.
    
    # Output Results
    print("\n" + "="*65)
    print(f"🔮 双色球多模型综合分析报告 (历史回测期数: {args.eval_size})")
    print("="*65)
    print(f"最近期号: {df.iloc[-1]['期号']} | 预测目标: 正常下一期")
    
    for m in methods_to_run:
        probs = results[m]
        top_10_idx = probs.argsort()[-10:][::-1]
        top_10_nums = [int(idx + 1) for idx in top_10_idx]
        
        # Calculate Min/Max for individual model standardization display
        p_min, p_max = probs.min(), probs.max()
        
        m_names = {'A': "统计相似度 (Method A)", 'B': "机器学习 RF (Method B)", 'C': "机器学习 XGB (Method C)", 'D': "深度学习 LSTM (Method D)"}
        print("\n" + f"--- {m_names[m]} ---")
        history_str = ", ".join([f"{h6:.0%}/{h10:.0%}" for h6, h10 in eval_results[m]['history']])
        print(f"历史回测平均命中率 (Top 6/10 覆盖率): {eval_results[m]['avg_6']:.2%}/{eval_results[m]['avg_10']:.2%} [{history_str}]")
        print(f"最佳 6 红推荐: {sorted(top_10_nums[:6])}")
        print(f"推荐 Top 10 号码 (原始概率 | 标准得分):")
        for idx in top_10_idx:
            std_val = (probs[idx] - p_min) / (p_max - p_min) if p_max > p_min else 0.0
            print(f"  号码: {int(idx+1):02d} - 概率: {probs[idx]:.2%} | 标准得分: {std_val:.4f}")
            
        # Display Feature Importance if available
        if m in importances:
            f_names = get_feature_names()
            top_f = plot_importance(importances[m], f_names, m_names[m], f"importance_{m.lower()}.png")
            print(f"  核心特征影响 (Top 10):")
            for _, row in top_f.iterrows():
                print(f"    - {row['Feature']}: {row['Importance']:.4f}")

    if args.method == 'all':
        # Standardized Probability Ensemble (Min-Max Scaling)
        # 1. Normalize each model's raw probabilities to [0, 1]
        # 2. Weighted average based on historical hit rates (Min 35% hit rate required)
        
        ensemble_scores = np.zeros(TOTAL_NUMBERS)
        
        # Filter methods with at least 30% hit rate
        qualified_methods = [m for m in active_methods if eval_results[m]['avg_10'] >= 0.30]
        
        if not qualified_methods:
            print("\n" + "="*65)
            print("⚠️ 没有模型的历史命中率达到 30% 阈值，跳过综合推荐。")
            print("="*65)
        else:
            total_weight = sum([eval_results[m]['avg_10'] for m in qualified_methods])  # 30% hit rate      
            
            print("\n" + "="*65)
            print("🏆 多模型标准概率综合推荐 (Ensemble: 4-Model Weighted)")
            print("="*65)
            print(f"计算逻辑: [标准化概率 = (原概率 - Min) / (Max - Min)]")
            print(f"入选模型: " + ", ".join([f"{m}={eval_results[m]['avg_10']:.2%}" for m in qualified_methods]))
            print("-" * 65)
            
            for m in qualified_methods:
                raw_probs = results[m]
                weight = eval_results[m]['avg_10']
                
                p_min = raw_probs.min()
                p_max = raw_probs.max()
                
                # Avoid division by zero if all probabilities are the same
                if p_max > p_min:
                    std_probs = (raw_probs - p_min) / (p_max - p_min)
                else:
                    std_probs = np.zeros(TOTAL_NUMBERS)
                    
                ensemble_scores += (std_probs * weight)
            
            # Average by total weight
            if total_weight > 0:
                ensemble_scores /= total_weight
                
            top_10_idx = ensemble_scores.argsort()[-10:][::-1]
            top_10_nums = [int(idx + 1) for idx in top_10_idx] 
            
            print(f"最佳 6 红组合: {sorted(top_10_nums[:6])}")
            print(f"推荐 Top 10 (标准综合得分):")
            for idx in top_10_idx:
                print(f"  号码: {int(idx+1):02d} - 标准得分: {ensemble_scores[idx]:.4f}")
            
            print("\n" + "="*65)
            # Display Ensemble History
            if 'Ensemble' in eval_results:
                ens_res = eval_results['Ensemble']
                ens_history_str = ", ".join([f"{h6:.0%}/{h10:.0%}" for h6, h10 in ens_res['history']])
                print(f"🏆 综合推荐历史命中率 (Top 6/10 覆盖率): {ens_res['avg_6']:.2%}/{ens_res['avg_10']:.2%} [{ens_history_str}]")
                
                # Display Union & Voting Stats
                if 'Union' in eval_results:
                    u_res = eval_results['Union']
                    u_hist_str = ", ".join([f"{h}({s})" for h, s in u_res['history']])
                    print(f"🌌 全模型并集 (Top 10 Union) 历史: 平均命中 {u_res['avg_hits']:.1f} 个 / 平均选号 {u_res['avg_size']:.1f} 个 [{u_hist_str}]")
                
                if 'Voting' in eval_results:
                    v_res = eval_results['Voting']
                    v_hist_str = ", ".join([f"{h:.0%}" for h in v_res['history']])
                    print(f"🗳️ 频次投票 (Best 6) 历史命中率: {v_res['avg_6']:.2%} [{v_hist_str}]")
                    
                print("="*65)
        

if __name__ == "__main__":
    main()
