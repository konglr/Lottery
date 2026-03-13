import logging
import numpy as np
import pygad
import sys
import os
import pandas as pd

# Add project root to path to import funcs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from funcs.ball_filter import calculate_morphology_score

# Global variables to be set by the main function
# This is a workaround to pass complex data to the fitness function in pygad
LAST_N_PERIODS_NUMS = []
LOTTERY_NAME = ""

def fitness_func_wrapper(ga_instance, solution, solution_idx):
    """
    Wrapper for the fitness function to be used by PyGAD.
    It calculates the average morphology score against the last N periods.
    """
    global LAST_N_PERIODS_NUMS, LOTTERY_NAME
    
    solution_nums = sorted([int(i) for i in solution])
    
    if not LAST_N_PERIODS_NUMS:
        return 0

    total_score = 0
    for prev_nums in LAST_N_PERIODS_NUMS:
        # Use the existing morphology scoring logic
        score = calculate_morphology_score(solution_nums, prev_nums, LOTTERY_NAME)
        total_score += score
        
    # Average score across the last N periods
    average_score = total_score / len(LAST_N_PERIODS_NUMS)
    return average_score

def train_predict_ga(df, model_config, lottery_config):
    """
    Model I: GA (Genetic Algorithm)
    Goal: Evolve number combinations that best fit recent morphological trends.
    Principle: Chromosomes (combinations) are evaluated by a fitness function
               based on morphology scores against recent draws.
    """
    global LAST_N_PERIODS_NUMS, LOTTERY_NAME
    
    logging.info("Method I: 执行 GA 遗传算法优化选号...")

    # 1. Setup Fitness Function Context
    conf = model_config
    fitness_periods = conf.get('fitness_periods', 10)
    
    # Get the red ball numbers from the last N periods for fitness evaluation
    red_cols = lottery_config['red_cols']
    recent_df = df.tail(fitness_periods)
    LAST_N_PERIODS_NUMS = []
    for i in range(len(recent_df)):
        try:
            nums = sorted([int(n) for n in recent_df.iloc[i][red_cols]])
            LAST_N_PERIODS_NUMS.append(nums)
        except:
            continue
            
    if not LAST_N_PERIODS_NUMS:
        logging.warning("GA: Not enough historical data for fitness evaluation. Skipping.")
        return np.zeros(lottery_config['total_numbers'])

    LOTTERY_NAME = lottery_config.get('name', '双色球') # Pass lottery name for rules

    # 2. Configure GA
    gene_space = list(range(lottery_config['red_range'][0], lottery_config['red_range'][1] + 1))
    
    ga_instance = pygad.GA(
        num_generations=conf.get('generations', 50),
        num_parents_mating=int(conf.get('population_size', 100) / 5),
        fitness_func=fitness_func_wrapper,
        sol_per_pop=conf.get('population_size', 100),
        num_genes=lottery_config['red_count'],
        gene_space=gene_space,
        gene_type=int,
        allow_duplicate_genes=False,
        mutation_type="random",
        mutation_percent_genes=conf.get('mutation_rate', 0.1) * 100,
        suppress_warnings=True,
        stop_criteria="saturate_5" 
    )

    # 3. Run GA
    ga_instance.run()
    
    # 4. Process Results
    final_population = ga_instance.population
    all_genes = final_population.flatten()
    num_counts = np.bincount(all_genes.astype(int), minlength=lottery_config['red_range'][1] + 1)
    
    probs = np.zeros(lottery_config['total_numbers'])
    
    # Map Red balls
    if lottery_config.get('separate_pool', False):
        for i, num in enumerate(lottery_config['red_num_list']):
            # Add small random noise to red balls too just in case of frequency ties at the GA output level
            probs[i] = num_counts[num] + (np.random.random() * 0.0001)
            
        # 5. Hand-craft Blue ball distribution from recent frequency
        if lottery_config.get('blue_cols'):
            blue_conf = conf.get('blue_config', conf)
            blue_fitness_periods = blue_conf.get('fitness_periods', 30)
            blue_recent_df = df.tail(blue_fitness_periods)
            
            blue_cols = lottery_config['blue_cols']
            if all(c in blue_recent_df.columns for c in blue_cols):
                blue_vals = blue_recent_df[blue_cols].values.flatten()
                blue_vals = pd.to_numeric(blue_vals, errors='coerce')
                blue_vals = blue_vals[~np.isnan(blue_vals)]
                
                blue_counts = np.bincount(blue_vals.astype(int), minlength=lottery_config['blue_range'][1] + 1)
                for i, num in enumerate(lottery_config['blue_num_list']):
                    # Offset into blue's domain and add a tiny random noise to prevent identical ties 
                    # (identical ties result in picking the lowest sequential numbers)
                    noise = np.random.random() * 0.0001
                    probs[lottery_config['total_red'] + i] = blue_counts[num] + noise
                    
    else:
        for i, num in enumerate(lottery_config['num_list']):
            probs[i] = num_counts[num] + (np.random.random() * 0.0001)
            
    # Normalize Probabilities
    if lottery_config['separate_pool']:
        # Normalize red and blue blocks independently
        red_sum = probs[:lottery_config['total_red']].sum()
        if red_sum > 0: probs[:lottery_config['total_red']] /= red_sum
        
        blue_sum = probs[lottery_config['total_red']:].sum()
        if blue_sum > 0: probs[lottery_config['total_red']:] /= blue_sum
    else:
        if probs.sum() > 0:
            probs /= probs.sum()
    
    best_solution, best_fitness, _ = ga_instance.best_solution()
    logging.info(f"GA: Best solution found: {sorted([int(g) for g in best_solution])} with fitness {best_fitness:.2f}")

    return probs