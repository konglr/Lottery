import logging
import numpy as np
import pandas as pd

def train_predict_evt(df, model_config, lottery_config):
    """
    Model H: EVT (Extreme Value Theory - Mean Reversion)
    Goal: Predict "strong rebound" when statistical indicators deviate from extremes.
    Principle: 3-sigma outliers. Predict regression when indicators continuously deviate.
    """
    logging.info("Method H: 执行 EVT 极值理论与均值回归分析...")

    TOTAL_NUMBERS = lottery_config['total_numbers']
    # Use recent periods as specified
    recent_len = model_config.get('recent_periods', 150)
    if len(df) < recent_len:
        recent_df = df
    else:
        recent_df = df.tail(recent_len)

    weights = np.ones(TOTAL_NUMBERS) * 0.1 # Base weight

    # Helper to map number value to index
    def get_idx(val, is_blue=False):
        if lottery_config['separate_pool']:
             if is_blue:
                 if val in lottery_config['blue_num_list']:
                     return lottery_config['total_red'] + lottery_config['blue_num_list'].index(val)
                 return -1
             else:
                 if val in lottery_config['red_num_list']:
                     return lottery_config['red_num_list'].index(val)
                 return -1
        else:
             if val in lottery_config['num_list']:
                 return lottery_config['num_list'].index(val)
             return -1

    # Frequency analysis to provide a realistic baseline (prevents 1, 2, 3, 4 linear results)
    red_cols = lottery_config['red_cols']
    if all(c in recent_df.columns for c in red_cols):
        red_vals = recent_df[red_cols].values.flatten()
        red_vals = pd.to_numeric(red_vals, errors='coerce')
        red_vals = [int(v) for v in red_vals if not np.isnan(v)]
        for v in red_vals:
            idx = get_idx(v, is_blue=False)
            if idx != -1: weights[idx] += 0.05 # Add frequency bonus

    if lottery_config['separate_pool'] and lottery_config.get('blue_cols'):
        blue_cols = lottery_config['blue_cols']
        if all(c in recent_df.columns for c in blue_cols):
            blue_vals = recent_df[blue_cols].values.flatten()
            blue_vals = pd.to_numeric(blue_vals, errors='coerce')
            blue_vals = [int(v) for v in blue_vals if not np.isnan(v)]
            for v in blue_vals:
                idx = get_idx(v, is_blue=True)
                if idx != -1: weights[idx] += 0.05

    # 1. Analyze Core Indicators (Sum, Span) for Red Balls
    if '和值' in recent_df.columns and '跨度' in recent_df.columns:
        sums = recent_df['和值'].values
        spans = recent_df['跨度'].values
        
        last_sum = sums[-1]
        last_span = spans[-1]
        
        mean_sum, std_sum = np.mean(sums), np.std(sums)
        mean_span, std_span = np.mean(spans), np.std(spans)
        
        sum_min = model_config.get('sum_min', mean_sum - 3 * std_sum)
        sum_max = model_config.get('sum_max', mean_sum + 3 * std_sum)
        
        # Determine Reversion Force
        sum_z = (last_sum - mean_sum) / (std_sum if std_sum > 0 else 1)
        
        is_extreme_low = last_sum < sum_min
        is_extreme_high = last_sum > sum_max
        
        reversion_intensity = 0.0
        if is_extreme_low:
            reversion_intensity = 1.0 + abs(sum_z) 
            logging.info(f"EVT: 和值极低 ({last_sum} < {sum_min:.1f}), 触发强力反弹(偏向大号)")
        elif is_extreme_high:
            reversion_intensity = -1.0 - abs(sum_z)
            logging.info(f"EVT: 和值极高 ({last_sum} > {sum_max:.1f}), 触发强力回归(偏向小号)")
        else:
            reversion_intensity = -0.5 * sum_z 
        
        if lottery_config['separate_pool']:
            valid_nums = lottery_config['red_num_list']
        else:
            valid_nums = lottery_config['num_list']
            
        min_n, max_n = min(valid_nums), max(valid_nums)
        span_z = (last_span - mean_span) / (std_span if std_span > 0 else 1)
        
        for num in valid_nums:
            idx = get_idx(num, is_blue=False)
            if idx == -1: continue
            
            norm_val = (num - min_n) / (max_n - min_n) if max_n > min_n else 0.5
            
            # Apply Sum Weight (multiplier instead of flat addition)
            sum_multiplier = 1.0
            if reversion_intensity > 0:
                sum_multiplier += norm_val * abs(reversion_intensity) * 0.5
            else:
                sum_multiplier += (1 - norm_val) * abs(reversion_intensity) * 0.5
                
            # Apply Span Weight
            dist_center = abs(norm_val - 0.5)
            span_force = -span_z * 0.2
            if span_force > 0: # Boost edges
                span_multiplier = 1.0 + dist_center * span_force
            else: # Boost center
                span_multiplier = 1.0 + (0.5 - dist_center) * abs(span_force)
                
            weights[idx] = weights[idx] * sum_multiplier * span_multiplier

    # 2. Analyze Blue Balls (if separate pool)
    if lottery_config['separate_pool'] and lottery_config.get('blue_cols'):
        blue_cols = lottery_config['blue_cols']
        if all(c in recent_df.columns for c in blue_cols):
            blue_vals_for_stats = recent_df[blue_cols].values.flatten()
            blue_vals_for_stats = pd.to_numeric(blue_vals_for_stats, errors='coerce')
            blue_vals_for_stats = blue_vals_for_stats[~np.isnan(blue_vals_for_stats)]
            
            if len(blue_vals_for_stats) > 0:
                mean_blue = np.mean(blue_vals_for_stats)
                std_blue = np.std(blue_vals_for_stats)
                
                last_draw_blues = recent_df.iloc[-1][blue_cols].values
                last_draw_blues = pd.to_numeric(last_draw_blues, errors='coerce')
                last_draw_blues = last_draw_blues[~np.isnan(last_draw_blues)]
                if len(last_draw_blues) > 0:
                    last_blue = np.mean(last_draw_blues)
                else:
                    last_blue = blue_vals_for_stats[-1]
                
                blue_z = (last_blue - mean_blue) / (std_blue if std_blue > 0 else 1)
                blue_reversion = -0.5 * blue_z
                
                valid_blues = lottery_config['blue_num_list']
                if valid_blues:
                    min_b, max_b = min(valid_blues), max(valid_blues)
                    for num in valid_blues:
                        idx = get_idx(num, is_blue=True)
                        if idx == -1: continue
                        
                        norm_val = (num - min_b) / (max_b - min_b) if max_b > min_b else 0.5
                        blue_multiplier = 1.0
                        if blue_reversion > 0: 
                            blue_multiplier += norm_val * abs(blue_reversion) * 0.5
                        else:
                            blue_multiplier += (1 - norm_val) * abs(blue_reversion) * 0.5
                            
                        weights[idx] = weights[idx] * blue_multiplier
                        
    # Ensure no zeroes and normalize
    weights += 0.01
    probs = weights / weights.sum() if weights.sum() > 0 else np.zeros(TOTAL_NUMBERS)
    return probs