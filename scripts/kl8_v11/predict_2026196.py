"""
KL8 V11 2026196 期预测 + 完整选号方案
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

def feat_freq(i, window):
    if i + window > n_periods: window = n_periods - i
    if window <= 0: return np.zeros(80, dtype=float)
    block = data[i+1:i+1+window]
    flat = block.flatten()
    return np.bincount(flat, minlength=81)[1:81].astype(float)

def feat_repeat(i, span):
    counts = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if i + j >= n_periods: break
        nums = data[i+j]
        counts[nums-1] += 1
    return counts

def feat_neighbor(i, span, distance=2):
    counts = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if i + j >= n_periods: break
        last = data[i+j]
        for n in last:
            for d in range(-distance, distance+1):
                if d == 0: continue
                if 1 <= n+d <= 80: counts[n+d-1] += 1
    return counts

def score_v10(i):
    return (feat_freq(i, 10) * -3.0 +
            feat_repeat(i, 3) * -3.0 +
            feat_neighbor(i, 5) * -3.0)

def score_v3(i):
    return feat_freq(i, 50)

# 2026195 实际
actual_2026195 = set(data[0].tolist())
print(f'2026195 实际开奖: {sorted(actual_2026195)}')

# 2026195 形态
n_repeat = len(actual_2026195 & set(data[1].tolist()))
sorted_curr = sorted(actual_2026195)
n_consec = sum(1 for j in range(len(sorted_curr)-1) if sorted_curr[j+1] - sorted_curr[j] == 1)
print(f'2026195 形态: 重号={n_repeat}, 连号对={n_consec}')

# 2026196 预测
# 用 i=0 = 2026196, data[1:] 为历史 (但 i=0 用 data[1:] 即 2026195)
# data[0] = 2026195, 所以 "i = 0" 表示 2026196
# 但 score_v10(0) 用的是 data[1:1+window] = data[1:...] 即 2026195 + 历史
# 所以 V10(0) = V11 的 2026196 预测

# 上期 (2026195) 的形态决定 2026196 用 V10 还是 V3
th_repeat = 7
th_consec = 7
if n_repeat >= th_repeat or n_consec >= th_consec:
    strategy = 'V3 (追热号)'
    scores = score_v3(0)
else:
    strategy = 'V10 (反热号)'
    scores = score_v10(0)
print(f'\n2026196 选 {strategy}')

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
    return f'和值={sum_v}, 跨度={span}, 三区={z1}:{z2}:{z3}, 奇偶={odd}:{len(nums)-odd}, 大小={big}:{len(nums)-big}, 连号对={consec}'

# 不同 TopN
print()
print('=' * 70)
print('V11 2026196 期预测')
print('=' * 70)

for top_n in [9, 15, 20, 25, 30]:
    top = sorted((np.argsort(-scores)[:top_n] + 1).tolist())
    print(f'\nV11 Top{top_n}: {top}')
    print(f'   形态: {morphology(top)}')

# 同时给 V10 和 V3 单独的 Top20 做对比
print()
print('=' * 70)
print('V10 / V3 单独 Top20 对比')
print('=' * 70)

scores_v10 = score_v10(0)
top20_v10 = sorted((np.argsort(-scores_v10)[:20] + 1).tolist())
print(f'\nV10 Top20: {top20_v10}')
print(f'   形态: {morphology(top20_v10)}')

scores_v3 = score_v3(0)
top20_v3 = sorted((np.argsort(-scores_v3)[:20] + 1).tolist())
print(f'\nV3 Top20: {top20_v3}')
print(f'   形态: {morphology(top20_v3)}')

# V11 Top20
top20_v11 = sorted((np.argsort(-scores)[:20] + 1).tolist())
print(f'\nV11 Top20 (本次选择): {top20_v11}')
print(f'   形态: {morphology(top20_v11)}')

# V10 vs V3 共识
set_v10 = set(top20_v10)
set_v3 = set(top20_v3)
set_v11 = set(top20_v11)
print(f'\n共识分析 (Top20):')
print(f'  V10 ∩ V3: {sorted(set_v10 & set_v3)} ({len(set_v10 & set_v3)}/20)')
print(f'  V10 ∩ V11: {sorted(set_v10 & set_v11)} ({len(set_v10 & set_v11)}/20)')
print(f'  V3  ∩ V11: {sorted(set_v3 & set_v11)} ({len(set_v3 & set_v11)}/20)')

# 保存
out = {
    'target_period': '2026196',
    'predict_time': '2026-07-25 13:00 GMT+8',
    'last_period': '2026195',
    'last_nums': sorted(actual_2026195),
    'last_meta': {'n_repeat': n_repeat, 'n_consec': n_consec},
    'v11_strategy': strategy,
    'v11_thresholds': {'th_repeat': th_repeat, 'th_consec': th_consec},
    'v11_top20': top20_v11,
    'v11_top25': sorted((np.argsort(-scores)[:25] + 1).tolist()),
    'v11_top30': sorted((np.argsort(-scores)[:30] + 1).tolist()),
    'v10_top20': top20_v10,
    'v3_top20': top20_v3,
    'consensus': {
        'v10_v3': sorted(set_v10 & set_v3),
        'v10_v11': sorted(set_v10 & set_v11),
        'v3_v11': sorted(set_v3 & set_v11),
    },
    'morphology': {
        'v11_top20': morphology(top20_v11),
        'v10_top20': morphology(top20_v10),
        'v3_top20': morphology(top20_v3),
    }
}
with open('/Users/clarkkong/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery/data/backtest/2026196_predictions_v11.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('预测已保存到 data/backtest/2026196_predictions_v11.json')