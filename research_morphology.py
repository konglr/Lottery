import pandas as pd
import numpy as np
from collections import Counter
import os
import sys

# 确保可以导入 funcs 中的逻辑
sys.path.append(os.getcwd())
from funcs.ball_filter import calculate_morphology_score
from config import LOTTERY_CONFIG

def research_lottery_morphology(lottery_name="双色球", limit=1000):
    print(f"=== {lottery_name} Morphological Backtest Research (Recent {limit} Periods) ===")
    
    config = LOTTERY_CONFIG.get(lottery_name)
    if not config:
        print("Lottery config not found")
        return

    # 1. Load Data
    df = pd.read_csv(config["data_file"])
    if 'issue' in df.columns:
        df = df.rename(columns={'issue': 'issue_num'})
    elif 'period' in df.columns:
        df = df.rename(columns={'period': 'issue_num'})
    
    # Ensure correct sorting
    df['issue_num'] = pd.to_numeric(df['issue_num'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('issue_num', ascending=True)
    
    # Red ball column names
    red_cols = [f"{config['red_col_prefix']}{i}" for i in range(1, config['red_count'] + 1)]
    
    stats = {
        'scores': [], 'repeats': [], 'consecutive_groups': [],
        'odd_ratios': [], 'same_tail_groups': [], 'spans': [], 'sums': []
    }

    # Analyze period by period
    data_to_analyze = df.tail(limit + 1)
    if len(data_to_analyze) < 2:
        print("Not enough data to analyze")
        return

    for i in range(1, len(data_to_analyze)):
        curr_row = data_to_analyze.iloc[i]
        prev_row = data_to_analyze.iloc[i-1]
        
        try:
            curr_nums = sorted([int(curr_row[c]) for c in red_cols])
            prev_nums = sorted([int(prev_row[c]) for c in red_cols])
        except (ValueError, TypeError):
            continue
        
        # Calculate current system score (Passing lottery_name)
        score = calculate_morphology_score(curr_nums, prev_nums, lottery_name)
        stats['scores'].append(score)
        
        stats['repeats'].append(len(set(curr_nums) & set(prev_nums)))
        cons_groups = 0
        for j in range(len(curr_nums) - 1):
            if curr_nums[j+1] - curr_nums[j] == 1: cons_groups += 1
        stats['consecutive_groups'].append(cons_groups)
        odds = len([n for n in curr_nums if n % 2 != 0])
        stats['odd_ratios'].append(f"{odds}:{config['red_count']-odds}")
        tails = [n % 10 for n in curr_nums]
        stats['same_tail_groups'].append(len([v for v in Counter(tails).values() if v >= 2]))
        stats['spans'].append(curr_nums[-1] - curr_nums[0])
        stats['sums'].append(sum(curr_nums))

    total = len(stats['scores'])
    if total == 0: return

    rules = config.get("morphology_rules", {})

    print(f"\n[1] Score Distribution (Logic V3.0):")
    score_counts = Counter(stats['scores'])
    for s in sorted(score_counts.keys(), reverse=True):
        print(f"  - {s} pts: {score_counts[s]} rows ({(score_counts[s]/total):.1%})")

    print(f"\n[2] Repeats (Ideal {rules.get('ideal_repeats')}):")
    for r, count in sorted(Counter(stats['repeats']).items()):
        print(f"  - {r} repeats: {count} rows ({(count/total):.1%})")

    print(f"\n[3] Consecutive Groups (Ideal {rules.get('ideal_consecutive')}):")
    for c, count in sorted(Counter(stats['consecutive_groups']).items()):
        print(f"  - {c} groups: {count} rows ({(count/total):.1%})")

    print(f"\n[4] Odd-Even Ratio (Ideal Odds {rules.get('ideal_odd_counts')}):")
    for o, count in sorted(Counter(stats['odd_ratios']).items()):
        print(f"  - Ratio {o}: {count} rows ({(count/total):.1%})")

    print(f"\n[5] Same Tail Groups:")
    for tg, count in sorted(Counter(stats['same_tail_groups']).items()):
        print(f"  - {tg} groups: {count} rows ({(count/total):.1%})")

    print(f"\n[6] Span (Ideal {rules.get('span_range')}):")
    low_b, high_b = rules.get('span_range', (0, 100))
    mid = len([s for s in stats['spans'] if low_b <= s <= high_b])
    print(f"  - In Range: {mid} rows ({(mid/total):.1%})")

    print(f"\n[7] Sum (Ideal {rules.get('sum_range')}):")
    low_b, high_b = rules.get('sum_range', (0, 9999))
    mid = len([s for s in stats['sums'] if low_b <= s <= high_b])
    print(f"  - In Range: {mid} rows ({(mid/total):.1%})")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "双色球"
    research_lottery_morphology(target, 1000)
