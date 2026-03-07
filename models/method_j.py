import logging
import numpy as np
import pandas as pd
from scipy.stats import poisson

def train_predict_poisson(df, model_config, lottery_config):
    """
    Model J: Poisson & Convergence
    Goal: Calculate "waiting time" pressure for cold numbers.
    Principle: Based on Poisson distribution. If a number hasn't appeared for M periods,
               calculate the probability of it entering the "high pressure zone".
               Pressure = 1 - exp(-lambda), where lambda = omission * theoretical_prob.
    """
    logging.info("Method J: 执行泊松分布与收敛分析 (冷态狙击)...")
    
    total_numbers = lottery_config['total_numbers']
    omission_counts = np.zeros(total_numbers)
    
    # 1. Calculate Current Omission (Scan backwards)
    # Initialize all as "not found"
    found_flags = np.zeros(total_numbers, dtype=bool)
    
    # Iterate backwards from the last row
    for idx in range(len(df)-1, -1, -1):
        if np.all(found_flags):
            break
            
        row = df.iloc[idx]
        current_draw_indices = []
        
        # Red Balls
        for c in lottery_config['red_cols']:
            val = row.get(c)
            if pd.notna(val):
                try:
                    v = int(val)
                    if lottery_config['separate_pool']:
                        if v in lottery_config['red_num_list']:
                            current_draw_indices.append(lottery_config['red_num_list'].index(v))
                    else:
                        if v in lottery_config['num_list']:
                            current_draw_indices.append(lottery_config['num_list'].index(v))
                except: pass
        
        # Blue Balls
        if lottery_config['separate_pool'] and lottery_config['blue_cols']:
            for c in lottery_config['blue_cols']:
                val = row.get(c)
                if pd.notna(val):
                    try:
                        v = int(val)
                        if v in lottery_config['blue_num_list']:
                            current_draw_indices.append(lottery_config['total_red'] + lottery_config['blue_num_list'].index(v))
                    except: pass
        
        # Update omission counts
        for i in range(total_numbers):
            if not found_flags[i]:
                if i in current_draw_indices:
                    found_flags[i] = True
                else:
                    omission_counts[i] += 1
                    
    # 2. Calculate Theoretical Probability (P) per draw
    probs_p = np.zeros(total_numbers)
    
    if lottery_config['separate_pool']:
        # Example SSQ: Red 6/33, Blue 1/16
        p_red = lottery_config['red_count'] / len(lottery_config['red_num_list'])
        p_blue = lottery_config['blue_count'] / len(lottery_config['blue_num_list'])
        
        probs_p[:lottery_config['total_red']] = p_red
        probs_p[lottery_config['total_red']:] = p_blue
    else:
        # Example KL8: 20/80
        p_val = lottery_config['red_count'] / len(lottery_config['num_list'])
        probs_p[:] = p_val
        
    # 3. Calculate Poisson Pressure Score
    # lambda = expected occurrences in 'omission' trials = omission * P
    # Pressure = Probability that it SHOULD have appeared by now = 1 - P(X=0) = 1 - exp(-lambda)
    lambdas = omission_counts * probs_p
    pressure_scores = 1 - np.exp(-lambdas)
    
    # Normalize to probabilities
    # We add a small epsilon to avoid zero probabilities for recently drawn numbers
    pressure_scores += 0.01 
    if pressure_scores.sum() > 0:
        final_probs = pressure_scores / pressure_scores.sum()
    else:
        final_probs = np.zeros(total_numbers)
        
    return final_probs