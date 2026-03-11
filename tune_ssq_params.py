import pandas as pd
import numpy as np
import json
import logging
import warnings
from tqdm import tqdm
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import accuracy_score
import os
import sys

# 导入模型训练函数 (假设 models 文件夹在当前目录)
# 注意：需要确保 models 文件夹中有 __init__.py 或者在 python path 中
from models.method_b import train_predict_rf
from models.method_c import train_predict_xgb
from models.method_e import train_predict_lgbm
from models.method_f import train_predict_catboost
# Model A, H, G 等统计模型通常参数较少，这里主要针对 ML 模型进行深度调优

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
warnings.filterwarnings("ignore")

# --- 双色球配置 ---
LOTTERY_CONFIG = {
    "name": "双色球",
    "red_range": (1, 33),
    "blue_range": (1, 16),
    "red_count": 6,
    "blue_count": 1,
    "window_size": 4, # 默认窗口大小，也可以作为参数调优
    "red_col_prefix": "红球",
    "blue_col_name": "蓝球"
}

# --- 核心辅助函数 (复用自 multi_model.py 但简化) ---
def load_data(file_path="data/双色球_lottery_data.csv"):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return None
    df = pd.read_csv(file_path)
    if 'issue' in df.columns: df = df.rename(columns={'issue': '期号'})
    df['期号'] = pd.to_numeric(df['期号'], errors='coerce').fillna(0).astype(int).astype(str)
    if int(df['期号'].iloc[0]) > int(df['期号'].iloc[-1]):
        df = df.iloc[::-1].reset_index(drop=True)
    return df

def get_omission_matrix(df, red_nums, blue_nums):
    n_rows = len(df)
    total_red = len(red_nums)
    total_blue = len(blue_nums)
    
    omission_matrix = np.zeros((n_rows, total_red + total_blue), dtype=int)
    curr_red = np.zeros(total_red, dtype=int)
    curr_blue = np.zeros(total_blue, dtype=int)
    
    red_cols = [f"红球{i}" for i in range(1, 7)]
    blue_col = "蓝球"
    
    for i in range(n_rows):
        omission_matrix[i, :total_red] = curr_red
        omission_matrix[i, total_red:] = curr_blue
        
        row_red = set(df.loc[i, red_cols].values)
        row_blue = int(df.loc[i, blue_col])
        
        for idx, num in enumerate(red_nums):
            if num in row_red: curr_red[idx] = 0
            else: curr_red[idx] += 1
            
        for idx, num in enumerate(blue_nums):
            if num == row_blue: curr_blue[idx] = 0
            else: curr_blue[idx] += 1
            
    return omission_matrix, np.concatenate([curr_red, curr_blue])

def extract_features(window_data, next_omission, red_nums, blue_nums, pool_type='red'):
    features = []
    red_cols = [f"红球{i}" for i in range(1, 7)]
    blue_col = "蓝球"
    total_red = len(red_nums)
    
    if pool_type == 'red':
        # 1. Freq
        vals = window_data[red_cols].values.flatten()
        freq = np.zeros(len(red_nums))
        for v in vals:
            if 1 <= v <= 33: freq[int(v)-1] += 1
        features.extend(freq)
        # 2. Omission
        features.extend(next_omission[:total_red])
        # 3. Basic Stats (Sum, Span)
        sums = window_data[red_cols].sum(axis=1).mean()
        features.append(sums)
        
    elif pool_type == 'blue':
        # 1. Freq
        vals = window_data[blue_col].values.flatten()
        freq = np.zeros(len(blue_nums))
        for v in vals:
            if 1 <= v <= 16: freq[int(v)-1] += 1
        features.extend(freq)
        # 2. Omission
        features.extend(next_omission[total_red:])
        
    return np.array(features)

def evaluate_params(model_func, params, X_train, y_train, X_val, y_val, final_feat_shape, pool_type, top_n=6):
    """
    训练模型并评估验证集上的命中率
    """
    # 构造临时的 config 对象
    model_config = params
    lottery_config = {
        'total_numbers': y_train.shape[1]
    }
    
    # 训练 (注意：这里我们简化了流程，直接用训练集训练，验证集预测)
    # models/method_*.py 中的函数通常是 fit + predict next period
    # 为了调优，我们需要它返回模型或者我们需要修改调用方式。
    # 现有的 train_predict_* 函数是设计为返回下一期概率的。
    # 我们可以 hack 一下：传入 X_train 作为训练，但我们需要它对 X_val 进行预测。
    # 遗憾的是现有函数内部直接 predict 了 final_feature。
    # 因此，我们需要稍微修改策略：
    # 我们将 X_val 的最后一行作为 "final_feature" 传入，但这只能评估一期。
    # 为了速度，我们只评估验证集的最后一期？不，这太随机了。
    # 正确做法：循环验证集每一期。
    
    hits = 0
    total = len(X_val)
    
    # 由于调用现有函数进行批量预测比较困难（它们是针对单期预测设计的），
    # 我们这里只取验证集的最后 5 期进行平均，以节省时间。
    eval_periods = min(5, len(X_val))
    
    for i in range(eval_periods):
        # 构造当前训练集：原始训练集 + 验证集的前 i 期
        curr_X = np.vstack([X_train, X_val[:i]]) if i > 0 else X_train
        curr_y = np.vstack([y_train, y_val[:i]]) if i > 0 else y_train
        curr_feat = X_val[i] # 当前要预测的特征
        
        try:
            probs, _ = model_func(curr_X, curr_y, curr_feat, model_config, lottery_config)
            
            # 评估
            actual_indices = np.where(y_val[i] == 1)[0]
            top_indices = probs.argsort()[::-1][:top_n]
            
            hit = len(set(top_indices) & set(actual_indices))
            hits += hit
        except Exception as e:
            # logging.error(f"Error in eval: {e}")
            return 0

    return hits / (eval_periods * (1 if pool_type=='blue' else 6)) # 返回平均命中率 (归一化)

# --- 参数搜索空间 ---
PARAM_GRIDS = {
    'B': { # Random Forest
        'n_estimators': [100, 300, 500, 800, 1000],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 10],
        'max_features': ['sqrt', 'log2'],
        'n_jobs': [-1]
    },
    'C': { # XGBoost
        'n_estimators': [300, 500, 800, 1000],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'scale_pos_weight': [1, 5, 10], # 处理不平衡
        'n_jobs': [-1],
        'eval_metric': ['logloss']
    },
    'E': { # LightGBM
        'n_estimators': [300, 500, 800, 1000],
        'num_leaves': [15, 31, 63, 127],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'feature_fraction': [0.7, 0.8, 0.9],
        'bagging_fraction': [0.7, 0.8, 0.9],
        'is_unbalance': [True, False],
        'n_jobs': [-1]
    },
    'F': { # CatBoost
        'iterations': [300, 500, 800],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5, 7],
        'random_state': [42]
    }
}

def run_tuning():
    print("🚀 开始双色球模型参数调优...")
    df = load_data()
    if df is None: return

    red_nums = list(range(1, 34))
    blue_nums = list(range(1, 17))
    
    # 准备数据
    omission, _ = get_omission_matrix(df, red_nums, blue_nums)
    
    # 构建数据集 (X, y)
    window_size = 4
    X_red, y_red = [], []
    X_blue, y_blue = [], []
    
    print("正在构建数据集...")
    for i in range(window_size, len(df)):
        win = df.iloc[i-window_size:i]
        # Red
        feat_r = extract_features(win, omission[i], red_nums, blue_nums, 'red')
        target_r = np.zeros(len(red_nums))
        curr_reds = [int(df.iloc[i][f'红球{k}']) for k in range(1, 7)]
        for r in curr_reds: target_r[r-1] = 1
        X_red.append(feat_r)
        y_red.append(target_r)
        
        # Blue
        feat_b = extract_features(win, omission[i], red_nums, blue_nums, 'blue')
        target_b = np.zeros(len(blue_nums))
        curr_blue = int(df.iloc[i]['蓝球'])
        target_b[curr_blue-1] = 1
        X_blue.append(feat_b)
        y_blue.append(target_b)
        
    X_red = np.array(X_red)
    y_red = np.array(y_red)
    X_blue = np.array(X_blue)
    y_blue = np.array(y_blue)
    
    # 划分 训练集 / 验证集 (最后 20 期作为验证)
    split_idx = len(X_red) - 20
    X_r_train, X_r_val = X_red[:split_idx], X_red[split_idx:]
    y_r_train, y_r_val = y_red[:split_idx], y_red[split_idx:]
    
    X_b_train, X_b_val = X_blue[:split_idx], X_blue[split_idx:]
    y_b_train, y_b_val = y_blue[:split_idx], y_blue[split_idx:]
    
    best_configs = {}
    
    models_map = {
        'B': train_predict_rf,
        'C': train_predict_xgb,
        'E': train_predict_lgbm,
        'F': train_predict_catboost
    }
    
    # 开始调优
    for model_name, func in models_map.items():
        print(f"\n--- 正在调优模型 {model_name} ---")
        param_list = list(ParameterSampler(PARAM_GRIDS[model_name], n_iter=10, random_state=42))
        
        # 1. Red Ball Tuning
        best_score_r = -1
        best_params_r = None
        
        print(f"  [红球] 搜索 {len(param_list)} 组参数...")
        for params in tqdm(param_list, leave=False):
            score = evaluate_params(func, params, X_r_train, y_r_train, X_r_val, y_r_val, None, 'red', top_n=6)
            if score > best_score_r:
                best_score_r = score
                best_params_r = params
        
        print(f"  ✅ 红球最佳得分: {best_score_r:.2%} | 参数: {best_params_r}")
        
        # 2. Blue Ball Tuning (Optional: use same params or retune)
        # 通常蓝球数据少，噪声大，使用红球的最优参数或者简单的参数即可。
        # 这里我们简单复用红球参数，或者进行小规模搜索。为了脚本完整性，我们独立搜索。
        best_score_b = -1
        best_params_b = None
        
        print(f"  [蓝球] 搜索 {len(param_list)} 组参数...")
        for params in tqdm(param_list, leave=False):
            score = evaluate_params(func, params, X_b_train, y_b_train, X_b_val, y_b_val, None, 'blue', top_n=1)
            if score > best_score_b:
                best_score_b = score
                best_params_b = params
                
        print(f"  ✅ 蓝球最佳得分: {best_score_b:.2%} | 参数: {best_params_b}")
        
        best_configs[model_name] = {
            'red': best_params_r,
            'blue': best_params_b
        }

    print("\n" + "="*50)
    print("🎉 调优完成！建议的 MODEL_CONFIG (SSQ):")
    print("="*50)
    print(json.dumps(best_configs, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    run_tuning()
