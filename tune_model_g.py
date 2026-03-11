import os
import sys
import pandas as pd
import numpy as np
import itertools
import logging
import json
from tqdm import tqdm
from hmmlearn import hmm

# Ensure funcs and models can be imported
sys.path.append(os.getcwd())
from config import LOTTERY_CONFIG

# --- Configuration ---
LOTTERY_NAME = "双色球"
N_COMPONENTS_RANGE = range(2, 11)
COVARIANCE_TYPES = ['diag', 'full']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- Helper Functions ---

def load_data(conf):
    """Loads and prepares lottery data."""
    try:
        df = pd.read_csv(conf['data_file'])
        if 'issue' in df.columns:
            df = df.rename(columns={'issue': '期号'})
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce').fillna(0).astype(int).astype(str)
        if int(df['期号'].iloc[0]) > int(df['期号'].iloc[-1]):
            df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def calculate_bic(model, X):
    """Calculates the Bayesian Information Criterion for a fitted HMM."""
    n_samples, n_features = X.shape
    
    # Number of parameters for transitions and start probabilities
    n_params = model.n_components * (model.n_components - 1) + (model.n_components - 1)
    
    # Number of parameters for emission probabilities (means and covariances)
    n_params += model.n_components * n_features  # Means
    if model.covariance_type == 'diag':
        n_params += model.n_components * n_features
    elif model.covariance_type == 'full':
        n_params += model.n_components * n_features * (n_features + 1) / 2
        
    log_likelihood = model.score(X)
    bic = n_params * np.log(n_samples) - 2 * log_likelihood
    return bic

def run_tuning():
    """Main function to run the HMM tuning process."""
    logging.info(f"🚀 开始为 [{LOTTERY_NAME}] 的模型 G (HMM) 进行参数调优...")
    
    conf = LOTTERY_CONFIG[LOTTERY_NAME]
    df = load_data(conf)
    if df is None: return

    red_cols = [f"{conf['red_col_prefix']}{i}" for i in range(1, conf['red_count'] + 1)]
    blue_cols = [conf['blue_col_name']]

    grid_combinations = list(itertools.product(N_COMPONENTS_RANGE, COVARIANCE_TYPES))
    
    best_config = {}

    for pool_type in ['red', 'blue']:
        logging.info(f"\n--- 正在调优 {pool_type.upper()} 球参数 ---")
        
        if pool_type == 'red':
            X = df[red_cols].values
        else:
            X = df[blue_cols].values

        best_bic = np.inf
        best_params = None
        
        for n_components, cov_type in tqdm(grid_combinations, desc=f"Grid Search ({pool_type.upper()})"):
            try:
                model = hmm.GaussianHMM(n_components=n_components, covariance_type=cov_type, n_iter=100, random_state=42)
                model.fit(X)
                bic = calculate_bic(model, X)
                
                if bic < best_bic:
                    best_bic = bic
                    best_params = {'n_components': n_components, 'covariance_type': cov_type}
            except Exception as e:
                # Some combinations might fail to converge, especially with 'full' covariance on high-dim data
                # logging.warning(f"  - Failed for n={n_components}, cov={cov_type}: {e}")
                continue

        best_config[pool_type] = {'bic': best_bic, 'params': best_params}

    print("\n" + "="*50)
    print("🎉 调优完成！模型 G 建议参数 (SSQ):")
    print("="*50)
    
    red_conf = best_config['red']
    print("\n[红球] Best BIC: {:.2f}".format(red_conf['bic']))
    print("Best Parameters:")
    print(json.dumps(red_conf['params'], indent=4, ensure_ascii=False))

    blue_conf = best_config['blue']
    print("\n[蓝球] Best BIC: {:.2f}".format(blue_conf['bic']))
    print("Best Parameters:")
    print(json.dumps(blue_conf['params'], indent=4, ensure_ascii=False))
    
    print("\n请将以上参数更新到 multi_model.py 的 MODEL_CONFIG['G']['ssq_red'] 和 MODEL_CONFIG['G']['ssq_blue'] 中。")

if __name__ == "__main__":
    run_tuning()