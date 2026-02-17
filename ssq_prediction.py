import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path="data/双色球_lottery_data.csv"):
    """Load the processed lottery data."""
    try:
        df = pd.read_csv(file_path)
        if 'issue' in df.columns:
            df = df.rename(columns={'issue': '期号'})
        # Ensure '期号' is string and sorted correctly (oldest to newest for training)
        df['期号'] = df['期号'].astype(str)
        # Check if already sorted. Usually processed data is newest first.
        # We need oldest first for training sequences.
        if df['期号'].iloc[0] > df['期号'].iloc[-1]:
             df = df.iloc[::-1].reset_index(drop=True)
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None

def calculate_omission(df, red_cols):
    """
    Calculate the omission (periods since last appearance) for each number (1-33).
    Returns a DataFrame with columns 'Omission_1' to 'Omission_33'.
    The value at row `i` represents the omission count *before* the draw at row `i`.
    """
    n_rows = len(df)
    omission_matrix = np.zeros((n_rows, 33), dtype=int)
    
    # Initialize omission counts (assuming 0 omission at start or undefined, let's say 0)
    current_omission = np.zeros(33, dtype=int)
    
    for i in range(n_rows):
        # Save current omission state to matrix (state BEFORE the draw)
        omission_matrix[i] = current_omission.copy()
        
        # Current draw numbers
        current_draw = set(df.loc[i, red_cols].values)
        
        # Update omission counts for next step
        for num in range(1, 34):
            if num in current_draw:
                current_omission[num-1] = 0
            else:
                current_omission[num-1] += 1
                
    omission_df = pd.DataFrame(omission_matrix, columns=[f'Omission_{i}' for i in range(1, 34)])
    return omission_df, current_omission # Return dataframe AND the state for the *next* unknown draw

def create_features(df, window_size=4):
    """
    Create features based on a sliding window of historical data.
    Input: Dataframe with basic features.
    Output: X (Features), y (Target - Next Draw's Red Balls)
    """
    red_cols = [f'红球{i}' for i in range(1, 7)]
    
    # Check if necessary columns exist
    required_cols = red_cols + ['和值', 'AC', '跨度', '奇数', '大号']
    if not all(col in df.columns for col in required_cols):
        logging.error("Missing required columns in input data.")
        return None, None, None

    # Calculate Omission Features
    omission_df, next_omission_state = calculate_omission(df, red_cols)
    # Concatenate omission features to main df
    df = pd.concat([df, omission_df], axis=1)
    
    X = []
    y = []
    
    # We need 'window_size' periods to form a feature vector, to predict the next period.
    # We iterate from window_size to len(df) - 1
    # Example: Window=4.
    # Use 0,1,2,3 (features) to predict 4 (target).
    
    for i in range(window_size, len(df)):
        # Target: The red balls of the current period (i)
        target_balls = df.loc[i, red_cols].values
        # Create a Multi-Hot vector for the target (1 if present, 0 if not)
        target_vector = np.zeros(33, dtype=int)
        for val in target_balls:
            if 0 < val <= 33:
                target_vector[int(val)-1] = 1
        y.append(target_vector)
        
        # Features: 
        # 1. Historical Data (Window): i-window_size to i (exclusive)
        window_data = df.iloc[i-window_size:i]
        
        feature_vector = []
        
        # A. Frequency of each number in the window (Hot/Cold)
        all_balls_in_window = window_data[red_cols].values.flatten()
        freq_counts = np.bincount(all_balls_in_window.astype(int), minlength=34)[1:] # Index 1-33
        feature_vector.extend(freq_counts)
        
        # B. Omission State entering the target draw (i)
        # This is stored in `omission_df` at index `i` (calculated from 0..i-1)
        # Since we concatenated, it's in df.iloc[i]
        current_omission_state = df.iloc[i][[f'Omission_{k}' for k in range(1, 34)]].values
        feature_vector.extend(current_omission_state)
        
        # C. Statistical Features (Average/Last of Sum, AC, Span, Odd, Big)
        # Added: Repeat Count (重号), Consecutive (二连), Tail Stats, Neighbor (邻号), Jumps
        stat_cols = [
            '和值', 'AC', '跨度', '奇数', '大号', '重号', '邻号', 
            '二连', '三连', '二跳', '三跳', '四跳', '五跳', '六跳'
        ]
        # Ensure columns exist, fill with 0 if missing
        for col in stat_cols:
            if col not in window_data.columns:
                window_data[col] = 0
                
        stats_avg = window_data[stat_cols].mean().values
        stats_last = window_data[stat_cols].iloc[-1].values
        feature_vector.extend(stats_avg)
        feature_vector.extend(stats_last)
        
        # D. Tail Analysis ( 同尾 )
        # Count frequency of each tail (0-9) in the window
        # Get all numbers in window
        all_nums = window_data[red_cols].values.flatten().astype(int)
        all_tails = [n % 10 for n in all_nums if n > 0]
        tail_counts = np.bincount(all_tails, minlength=10)
        feature_vector.extend(tail_counts)
        
        X.append(feature_vector)
        
    return np.array(X), np.array(y), next_omission_state

def train_and_predict():
    # Load Data
    df = load_data()
    if df is None: return

    logging.info("加载数据完成，开始构建特征...")
    X, y, next_omission_state = create_features(df, window_size=4)
    
    if len(X) == 0:
        logging.error("数据不足，无法构建特征。")
        return

    # Split Data (Train on old, Test on recent)
    test_size = 10
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    logging.info(f"训练模型中 (样本数: {len(X_train)})...")
    
    # Model: Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    logging.info("正在评估最近10期的预测效果...")
    probs = np.zeros((len(X_test), 33))
    
    # Predict probabilities for test set
    predictions_proba = rf.predict_proba(X_test)
    # predictions_proba is a list of [n_samples, 2] arrays
    for i in range(33):
        if hasattr(predictions_proba[i], "shape") and predictions_proba[i].shape[1] == 2:
            probs[:, i] = predictions_proba[i][:, 1]
    
    # Evaluation
    total_hits = 0
    total_drawn = 0
    
    for i in range(len(X_test)):
        actual_indices = np.where(y_test[i] == 1)[0]
        # 使用 Top 10 进行评估
        top_10_indices = probs[i].argsort()[-10:][::-1]
        hits = len(set(actual_indices) & set(top_10_indices))
        total_hits += hits
        total_drawn += 6
        
        # 为了调试，可以选择打印每期的命中情况
        # logging.info(f"回测期数 -{test_size-i}: 命中个数={hits}/6")

    logging.info(f"历史回测平均命中率 (Top 10 覆盖率): {total_hits/total_drawn:.2%}")
    
    # Predict NEXT Period
    logging.info("正在预测下一期号码...")
    
    # Construct feature for NEXT draw
    # 1. Frequency in last 4
    last_4_rows = df.iloc[-4:]
    red_cols = [f'红球{i}' for i in range(1, 7)]
    all_balls = last_4_rows[red_cols].values.flatten()
    freq_counts = np.bincount(all_balls.astype(int), minlength=34)[1:]
    
    # 2. Next Omission State (we calculated this returned from calculate_omission)
    # But wait, create_features calls calculate_omission internally on the old df.
    # We need to trust the return value `next_omission_state` which is the state AFTER the last row of df.
    
    # 3. Stats
    stat_cols = [
        '和值', 'AC', '跨度', '奇数', '大号', '重号', '邻号', 
        '二连', '三连', '二跳', '三跳', '四跳', '五跳', '六跳'
    ]
    # Check existence
    for col in stat_cols:
         if col not in last_4_rows.columns: last_4_rows[col] = 0
         
    stats_avg = last_4_rows[stat_cols].mean().values
    stats_last = last_4_rows[stat_cols].iloc[-1].values
    
    final_feature = []
    final_feature.extend(freq_counts)
    final_feature.extend(next_omission_state)
    final_feature.extend(stats_avg)
    final_feature.extend(stats_last)
    
    # 4. Tail Analysis for Last Window
    all_nums = last_4_rows[red_cols].values.flatten().astype(int)
    all_tails = [n % 10 for n in all_nums if n > 0]
    tail_counts = np.bincount(all_tails, minlength=10)
    final_feature.extend(tail_counts)
    
    final_X = np.array([final_feature])
    
    # Predict
    pred_proba = rf.predict_proba(final_X)
    final_probs = np.zeros(33)
    for i in range(33):
        if hasattr(pred_proba[i], "shape") and pred_proba[i].shape[1] == 2:
            final_probs[i] = pred_proba[i][0, 1]
    
    # Top 10 recommendations
    top_10_indices = final_probs.argsort()[-10:][::-1]
    top_10_probs = final_probs[top_10_indices]
    
    print("\n" + "="*40)
    print("🔮 下一期双色球预测 (基于RandomForest模型)")
    print("="*40)
    print(f"最近一期期号: {df.iloc[-1]['期号']}")
    print("-" * 40)
    print("推荐红球 (概率从高到低):")
    for idx, prob in zip(top_10_indices, top_10_probs):
        print(f"号码: {idx+1:02d} - 概率: {prob:.2%}")
        
    print("-" * 40)
    print(f"最佳6红组合: {sorted([x+1 for x in top_10_indices[:6]])}")
    print("="*40)

if __name__ == "__main__":
    train_and_predict()
