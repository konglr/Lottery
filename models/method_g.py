import logging
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def train_predict_hmm(df, model_config, lottery_config):
    """
    Model G: HMM (Hidden Markov Model)
    Goal: Capture "mode switching" (state transitions) in the lottery process.
    Principle: Assumes number trends (observable) are controlled by underlying "machine states" (hidden).
    Input: Normalized sequence of core statistical features (Sum, Span, AC).
    """
    logging.info("Method G: 训练 HMM 隐马尔可夫模型 (状态转移分析)...")
    
    # 1. Prepare Features (Sum, Span, AC)
    feature_cols = ['和值', '跨度', 'AC']
    # Filter columns that actually exist in the dataframe
    valid_cols = [c for c in feature_cols if c in df.columns]
    
    if len(valid_cols) < 3:
        logging.warning(f"HMM: 缺少必要特征 (需要: {feature_cols}, 实际: {valid_cols})，跳过。")
        return np.zeros(lottery_config['total_numbers'])
        
    # Use specified training length (Recent N periods)
    train_len = min(len(df), model_config.get('train_size', 1000))
    train_df = df.tail(train_len).copy().reset_index(drop=True)
    
    X = train_df[valid_cols].values
    
    # Normalize features
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError:
        return np.zeros(lottery_config['total_numbers'])
    
    # 2. Train HMM
    n_components = model_config.get('n_components', 4)
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=model_config.get('covariance_type', 'diag'),
        n_iter=100,
        random_state=model_config.get('random_state', 42),
        init_params="stmc"
    )
    
    try:
        model.fit(X_scaled)
    except Exception as e:
        logging.error(f"HMM Training Error: {e}")
        return np.zeros(lottery_config['total_numbers'])
        
    # 3. Predict Hidden States for history
    try:
        hidden_states = model.predict(X_scaled)
    except:
        return np.zeros(lottery_config['total_numbers'])
        
    # 4. Predict Next State
    last_state = hidden_states[-1]
    # Transition matrix: prob of moving from last_state to any other state
    next_state_probs = model.transmat_[last_state]
    next_state = np.argmax(next_state_probs)
    
    logging.info(f"HMM: 当前隐状态 {last_state} -> 预测下期隐状态 {next_state} (概率 {next_state_probs[next_state]:.2f})")
    
    # 5. Map State back to Numbers
    # Strategy: Aggregate numbers from all historical periods that belong to the predicted 'next_state'
    state_indices = np.where(hidden_states == next_state)[0]
    matching_draws = train_df.iloc[state_indices]
    
    if matching_draws.empty:
        return np.zeros(lottery_config['total_numbers'])
        
    # Count frequencies in these matching draws
    TOTAL_NUMBERS = lottery_config['total_numbers']
    pred_counts = np.zeros(TOTAL_NUMBERS)
    
    for _, row in matching_draws.iterrows():
        # Red Balls
        for col in lottery_config['red_cols']:
            val = row.get(col)
            if pd.notna(val):
                try:
                    v = int(val)
                    if lottery_config['separate_pool']:
                        if v in lottery_config['red_num_list']:
                            pred_counts[lottery_config['red_num_list'].index(v)] += 1
                    else:
                        if v in lottery_config['num_list']:
                            pred_counts[lottery_config['num_list'].index(v)] += 1
                except: pass
        # Blue Balls
        if lottery_config['separate_pool'] and lottery_config['blue_cols']:
            for col in lottery_config['blue_cols']:
                val = row.get(col)
                if pd.notna(val):
                    try:
                        v = int(val)
                        if v in lottery_config['blue_num_list']:
                            pred_counts[lottery_config['total_red'] + lottery_config['blue_num_list'].index(v)] += 1
                    except: pass
            
    # Normalize to probabilities
    if pred_counts.sum() > 0:
        probs = pred_counts / pred_counts.sum()
    else:
        probs = np.zeros(TOTAL_NUMBERS)
        
    return probs