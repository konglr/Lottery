import pandas as pd
import numpy as np
import itertools
import os
import sys

# 导入打分逻辑
project_root = os.getcwd()
sys.path.append(project_root)
from funcs.ball_filter import calculate_morphology_score, get_morphology_report

def recommend_from_backtest(model_prefix='Prob_B', top_n_balls=15, num_recommendations=5):
    # 1. 加载预测数据
    backtest_path = "data/ssq_backtest.csv"
    if not os.path.exists(backtest_path):
        print(f"Error: {backtest_path} not found.")
        return
        
    backtest_df = pd.read_csv(backtest_path)
    latest_pred = backtest_df.iloc[-1]
    target_period = latest_pred['Target_Period']
    
    # 2. 提取指定模型的所有概率
    # 获取列名 Prob_A_01, Prob_B_01 等
    # 检查有哪些 Prob 列
    prob_cols = [c for c in backtest_df.columns if c.startswith(model_prefix + '_')]
    if not prob_cols:
        # 降级尝试
        model_prefix = 'Prob_A' 
        prob_cols = [c for c in backtest_df.columns if c.startswith(model_prefix + '_')]

    probs = latest_pred[prob_cols].values
    
    # 3. 选出 Top N 个高分红球 (转为 Python int)
    top_indices = np.argsort(probs)[-top_n_balls:][::-1]
    # 列名通常是 Prob_X_01，索引 0 对应号码 1
    top_balls = sorted([int(c.split('_')[-1]) for c in np.array(prob_cols)[top_indices]])
    
    print(f">>> Target Period: {target_period}")
    print(f">>> Model: {model_prefix}")
    print(f">>> Top {top_n_balls} Candidates: {top_balls}")
    
    # 4. 获取上一期号码
    history_path = "data/双色球_lottery_data.csv"
    history_df = pd.read_csv(history_path)
    last_row = history_df.iloc[0]
    last_period_nums = sorted([
        int(last_row['红球1']), int(last_row['红球2']), int(last_row['红球3']),
        int(last_row['红球4']), int(last_row['红球5']), int(last_row['红球6'])
    ])
    print(f">>> Last Period ({last_row['issue']}): {last_period_nums}")
    print("-" * 30)

    # 5. 生成所有组合并打分
    all_combinations = list(itertools.combinations(top_balls, 6))
    scored_combinations = []
    
    for combo in all_combinations:
        combo_list = [int(x) for x in combo]
        score = calculate_morphology_score(combo_list, last_period_nums)
        scored_combinations.append((combo_list, score))
    
    # 6. 排序并输出结果
    scored_combinations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top {num_recommendations} Morphological Gems:")
    for i in range(min(num_recommendations, len(scored_combinations))):
        combo, score = scored_combinations[i]
        report = get_morphology_report(combo, last_period_nums)
        print(f"\n[Rank {i+1}] Score: {score}/100")
        print(report)

if __name__ == "__main__":
    recommend_from_backtest(model_prefix='Prob_B', top_n_balls=15, num_recommendations=5)
