import logging
import numpy as np
import pandas as pd

def train_predict_evt(df, model_config, lottery_config):
    """
    Model H: EVT (Extreme Value Theory - Mean Reversion)
    Goal: Predict "strong rebound" when statistical indicators deviate from extremes.
    Principle: 3-sigma outliers. Predict regression when indicators continuously deviate.
    Data: Recent 50 periods of sum_range, span_range, ac_trend.
    """
    logging.info("Method H: 执行 EVT 极值理论与均值回归分析...")

    TOTAL_NUMBERS = lottery_config['total_numbers']
    # Use recent 50 periods as specified
    recent_len = model_config.get('recent_periods', 50)
    if len(df) < recent_len:
        recent_df = df
    else:
        recent_df = df.tail(recent_len)

    # 1. Analyze Core Indicators (Sum, Span)
    # Calculate stats for the recent window to establish Mean and Std
    sums = recent_df['和值'].values
    spans = recent_df['跨度'].values
    
    # Current values (last draw)
    last_sum = sums[-1]
    last_span = spans[-1]
    
    # Mean and Std
    mean_sum, std_sum = np.mean(sums), np.std(sums)
    mean_span, std_span = np.mean(spans), np.std(spans)
    
    # Thresholds from config (Dynamic based on lottery type) or calculated (3-sigma)
    sum_min = model_config.get('sum_min', mean_sum - 3 * std_sum)
    sum_max = model_config.get('sum_max', mean_sum + 3 * std_sum)
    
    # 2. Determine Reversion Force
    weights = np.zeros(TOTAL_NUMBERS)
    
    # Helper to map number value to index
    def get_idx(val):
        if lottery_config['separate_pool']:
             if val in lottery_config['red_num_list']:
                 return lottery_config['red_num_list'].index(val)
             return -1
        else:
             if val in lottery_config['num_list']:
                 return lottery_config['num_list'].index(val)
             return -1

    # --- Sum Reversion Logic ---
    sum_z = (last_sum - mean_sum) / (std_sum if std_sum > 0 else 1)
    
    is_extreme_low = last_sum < sum_min
    is_extreme_high = last_sum > sum_max
    
    reversion_intensity = 0.0
    if is_extreme_low:
        reversion_intensity = 1.0 + abs(sum_z) # Strong upward push (favor large numbers)
        logging.info(f"EVT: 和值极低 ({last_sum} < {sum_min:.1f}), 触发强力反弹(偏向大号)")
    elif is_extreme_high:
        reversion_intensity = -1.0 - abs(sum_z) # Strong downward push (favor small numbers)
        logging.info(f"EVT: 和值极高 ({last_sum} > {sum_max:.1f}), 触发强力回归(偏向小号)")
    else:
        # Mild mean reversion proportional to Z-score
        reversion_intensity = -0.5 * sum_z 
    
    # Get all valid numbers (Focus on Red balls for Sum/Span)
    if lottery_config['separate_pool']:
        valid_nums = lottery_config['red_num_list']
    else:
        valid_nums = lottery_config['num_list']
        
    min_n, max_n = min(valid_nums), max(valid_nums)
    
    for num in valid_nums:
        idx = get_idx(num)
        if idx == -1: continue
        
        # Normalized position of number (0 to 1)
        norm_val = (num - min_n) / (max_n - min_n) if max_n > min_n else 0.5
        
        # Apply Sum Weight
        if reversion_intensity > 0:
            # Want bigger numbers
            weights[idx] += norm_val * abs(reversion_intensity)
        else:
            # Want smaller numbers
            weights[idx] += (1 - norm_val) * abs(reversion_intensity)
            
        # Apply Span Weight (Simplified)
        # If Span is low (clustered), want edges. If Span is high (spread), want center.
        span_z = (last_span - mean_span) / (std_span if std_span > 0 else 1)
        dist_center = abs(norm_val - 0.5)
        
        if span_z < -1.5: # Low span -> Expand -> Boost edges
            weights[idx] += dist_center * abs(span_z) * 0.5
        elif span_z > 1.5: # High span -> Contract -> Boost center
            weights[idx] += (0.5 - dist_center) * abs(span_z) * 0.5

    # Normalize to probabilities (Add base to avoid zero)
    weights += 0.1
    probs = weights / weights.sum() if weights.sum() > 0 else np.zeros(TOTAL_NUMBERS)
    return probs