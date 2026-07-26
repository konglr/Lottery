"""
KL8 2026196 期预测 - 修复版
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
import json

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)

def freq_window(start_idx, window):
    if start_idx + window > n_periods:
        window = n_periods - start_idx
    if window <= 0:
        return np.zeros(80, dtype=int)
    block = data[start_idx+1:start_idx+1+window]
    flat = block.flatten()
    return np.bincount(flat, minlength=81)[1:81]

# 预测目标期: 2026196 (假设 i=0 是 2026196,data[i+1] = 2026195)
# 但实际上 data[0] 就是 2026195,所以 i=0 = 2026196

def v8_score(i):
    freq_100 = freq_window(i, 100).astype(float)
    scores = freq_100.copy()
    # 邻号 (近 2 期 ±2)
    neighbors = set()
    for j in range(1, 3):
        last = set(data[i+j].tolist())
        for n in last:
            for d in [-2, -1, 1, 2]:
                if 1 <= n+d <= 80: neighbors.add(n+d)
    for n in neighbors:
        scores[n-1] += 3.0
    # 重号 (近 2 期)
    for j in range(1, 3):
        nums = set(data[i+j].tolist())
        for n in nums:
            scores[n-1] += 5.0
    return scores

# 朴素频次 W=100
freq_100 = freq_window(0, 100)
top_idx_naive = np.argsort(-freq_100)[:40] + 1
top30_naive = sorted(top_idx_naive[:30].tolist())
top40_naive = sorted(top_idx_naive.tolist())

# V8 评分
scores = v8_score(0)
top_idx_v8 = np.argsort(-scores)[:40] + 1
top30_v8 = sorted(top_idx_v8[:30].tolist())
top40_v8 = sorted(top_idx_v8.tolist())

# 形态
def morphology(nums):
    nums = sorted(nums)
    sum_v = sum(nums)
    span = max(nums) - min(nums)
    z1 = sum(1 for n in nums if n <= 27)
    z2 = sum(1 for n in nums if 28 <= n <= 53)
    z3 = sum(1 for n in nums if n >= 54)
    odd = sum(1 for n in nums if n % 2 == 1)
    big = sum(1 for n in nums if n > 40)
    consec = sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
    return {
        'sum': sum_v, 'span': span,
        'zone': f"{z1}:{z2}:{z3}",
        'odd_even': f"{odd}:{len(nums)-odd}",
        'big_small': f"{big}:{len(nums)-big}",
        'consec_pairs': consec,
    }

print('=' * 70)
print('KL8 2026196 期 V8 优化预测 (预测时间 2026-07-25 12:11 GMT+8)')
print('=' * 70)
print()
print(f'上期 2026195 开奖: {sorted(data[0].tolist())}')
print()
print('【策略 A: 朴素频次 W=100】(历史最优 Top40 策略)')
print(f'Top20: {top40_naive[:20]}')
print(f'Top30: {top30_naive}')
print(f'Top40: {top40_naive}')
print()
print('【策略 B: V8 综合 (频次 + 邻号 + 重号)】')
print(f'Top30: {top30_v8}')
print(f'Top40: {top40_v8}')
print()

# 形态对比
print('=' * 70)
print('形态对比')
print('=' * 70)
print(f'{"策略":<30} {"和值":>6} {"跨度":>5} {"三区":>8} {"奇偶":>7} {"大小":>7} {"连号对":>7}')

m_top20 = morphology(top40_naive[:20])
m_top30_naive = morphology(top30_naive)
m_top40_naive = morphology(top40_naive)
m_top30_v8 = morphology(top30_v8)
m_top40_v8 = morphology(top40_v8)

print(f'{"朴素 W=100 Top20":<30} {m_top20["sum"]:>6} {m_top20["span"]:>5} {m_top20["zone"]:>8} {m_top20["odd_even"]:>7} {m_top20["big_small"]:>7} {m_top20["consec_pairs"]:>7}')
print(f'{"朴素 W=100 Top30":<30} {m_top30_naive["sum"]:>6} {m_top30_naive["span"]:>5} {m_top30_naive["zone"]:>8} {m_top30_naive["odd_even"]:>7} {m_top30_naive["big_small"]:>7} {m_top30_naive["consec_pairs"]:>7}')
print(f'{"朴素 W=100 Top40":<30} {m_top40_naive["sum"]:>6} {m_top40_naive["span"]:>5} {m_top40_naive["zone"]:>8} {m_top40_naive["odd_even"]:>7} {m_top40_naive["big_small"]:>7} {m_top40_naive["consec_pairs"]:>7}')
print(f'{"V8 综合 Top30":<30} {m_top30_v8["sum"]:>6} {m_top30_v8["span"]:>5} {m_top30_v8["zone"]:>8} {m_top30_v8["odd_even"]:>7} {m_top30_v8["big_small"]:>7} {m_top30_v8["consec_pairs"]:>7}')
print(f'{"V8 综合 Top40":<30} {m_top40_v8["sum"]:>6} {m_top40_v8["span"]:>5} {m_top40_v8["zone"]:>8} {m_top40_v8["odd_even"]:>7} {m_top40_v8["big_small"]:>7} {m_top40_v8["consec_pairs"]:>7}')

# 共识度
print()
print('=' * 70)
print('Top40 共识度')
print('=' * 70)
set_a = set(top40_naive)
set_b = set(top40_v8)
print(f'A ∩ B: {sorted(set_a & set_b)} ({len(set_a & set_b)}/40)')
print(f'A 独有: {sorted(set_a - set_b)} ({len(set_a - set_b)} 个)')
print(f'B 独有: {sorted(set_b - set_a)} ({len(set_b - set_a)} 个)')

# 保存结果
out = {
    'target_period': '2026196',
    'predict_time': '2026-07-25 12:11 GMT+8',
    'last_period': '2026195',
    'last_nums': sorted(data[0].tolist()),
    'strategy_A_naive_W100': {
        'Top20': top40_naive[:20],
        'Top30': top30_naive,
        'Top40': top40_naive,
    },
    'strategy_B_v8': {
        'Top30': top30_v8,
        'Top40': top40_v8,
    },
    'morphology': {
        'A_top20': m_top20,
        'A_top30': m_top30_naive,
        'A_top40': m_top40_naive,
        'B_top30': m_top30_v8,
        'B_top40': m_top40_v8,
    },
    'consensus_top40': {
        'both': sorted(set_a & set_b),
        'only_A': sorted(set_a - set_b),
        'only_B': sorted(set_b - set_a),
    }
}
with open('/Users/clarkkong/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery/data/backtest/2026196_predictions_v8.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print()
print('✅ 预测已保存到 data/backtest/2026196_predictions_v8.json')