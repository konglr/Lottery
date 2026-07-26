"""
KL8 V12 最终方案: V10 + 配额约束 + 2026196 预测
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
import json
from collections import Counter

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

def classify_freq(freq, ratios=[0.25]*4):
    rank = np.argsort(-freq)
    n_total = 80
    n_high = int(ratios[0] * n_total)
    n_mid_high = int(ratios[1] * n_total)
    n_mid_low = int(ratios[2] * n_total)
    n_low = n_total - n_high - n_mid_high - n_mid_low
    classes = np.zeros(80, dtype=int)
    classes[rank[:n_high]] = 1
    classes[rank[n_high:n_high+n_mid_high]] = 2
    classes[rank[n_high+n_mid_high:n_high+n_mid_high+n_mid_low]] = 3
    classes[rank[n_high+n_mid_high+n_mid_low:]] = 4
    return classes

def score_v10(i):
    return (feat_freq(i, 10) * -3.0 +
            feat_repeat(i, 3) * -3.0 +
            feat_neighbor(i, 5) * -3.0)

def v12_select(i, min_q, max_q, top_n=20):
    """V12 配额约束选号"""
    freq = feat_freq(i, 100)
    cls = classify_freq(freq, [0.25]*4)
    s = score_v10(i)
    ZONE = np.array([((n-1)//20) + 1 for n in range(1, 81)], dtype=int)
    
    sorted_idx = np.argsort(-s)
    selected = []
    cell_count = Counter()
    
    # 1. 满足最小配额
    for (c, z), q in min_q.items():
        mask = (cls == c) & (ZONE == z)
        indices_in_group = np.where(mask)[0]
        if len(indices_in_group) == 0: continue
        group_scores = s[indices_in_group]
        order = np.argsort(group_scores)
        picks = indices_in_group[order[:q]]
        for p in picks:
            if p not in selected:
                selected.append(p)
                cell_count[(c, z)] += 1
    
    # 2. 按 TopN 补充
    for idx in sorted_idx:
        if len(selected) >= top_n: break
        c, z = cls[idx], ZONE[idx]
        if max_q and (c, z) in max_q and cell_count[(c, z)] >= max_q[(c, z)]:
            continue
        if idx in selected: continue
        selected.append(idx)
        cell_count[(c, z)] += 1
    return np.array(selected), cell_count

# 200 期回测 — V12 在不同 TopN
TEST_INDICES = list(range(0, 200))
N_TEST = len(TEST_INDICES)

# V12 最优配置
min_q_v12 = {(1, 1): 2, (2, 1): 2}  # Z1 高频+中高 各 2 个 = 4
max_q_v12 = {(4, 4): 3}  # 类4 Z4 ≤ 3

print('=' * 80)
print('V12 在不同 TopN 上的表现 (200 期)')
print('=' * 80)
print(f'配置: min_q = {min_q_v12}, max_q = {max_q_v12}')
print()

for top_n in [9, 12, 15, 18, 20, 25, 30]:
    hits = []
    for k in range(N_TEST):
        actual = set(data[TEST_INDICES[k]].tolist())
        selected, _ = v12_select(k, min_q_v12, max_q_v12, top_n)
        hits.append(len(set((selected + 1).tolist()) & actual))
    h = np.array(hits)
    
    # 对比 V10 基线
    v10_h = []
    for k in range(N_TEST):
        actual = set(data[TEST_INDICES[k]].tolist())
        s = score_v10(k)
        top = set((np.argsort(-s)[:top_n] + 1).tolist())
        v10_h.append(len(top & actual))
    v10_h = np.array(v10_h)
    
    print(f'Top{top_n}: V12={h.mean():.3f} ({h.mean()/20*100:.1f}%) ≥10中 {(h>=10).mean()*100:.1f}% | '
          f'V10={v10_h.mean():.3f} ({v10_h.mean()/20*100:.1f}%) ≥10中 {(v10_h>=10).mean()*100:.1f}%')

# 2026196 预测
print()
print('=' * 80)
print('V12 2026196 期预测')
print('=' * 80)
print(f'上期 2026195 实际: {sorted(data[0].tolist())}')

# V12 选号
for top_n in [20, 25, 30]:
    selected, cells = v12_select(0, min_q_v12, max_q_v12, top_n)
    nums = sorted((selected + 1).tolist())
    print(f'\nV12 Top{top_n}: {nums}')
    
    # 形态
    s = sum(nums)
    span = max(nums) - min(nums)
    z1 = sum(1 for n in nums if n <= 20)
    z2 = sum(1 for n in nums if 21 <= n <= 40)
    z3 = sum(1 for n in nums if 41 <= n <= 60)
    z4 = sum(1 for n in nums if n >= 61)
    odd = sum(1 for n in nums if n % 2 == 1)
    big = sum(1 for n in nums if n > 40)
    consec = sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
    print(f'  形态: 和值={s}, 跨度={span}, 四区=({z1}, {z2}, {z3}, {z4}), 奇偶={odd}:{len(nums)-odd}, 大小={big}:{len(nums)-big}, 连号对={consec}')
    print(f'  16 格子分布: {dict(cells)}')

# 也对比 V10
s_v10 = score_v10(0)
top20_v10 = sorted((np.argsort(-s_v10)[:20] + 1).tolist())
print()
print(f'V10 Top20 (对比): {top20_v10}')

# 共识度
selected_20, _ = v12_select(0, min_q_v12, max_q_v12, 20)
top20_v12 = sorted((selected_20 + 1).tolist())
set_v12 = set(top20_v12)
set_v10 = set(top20_v10)
print(f'\nV12 vs V10 共识 (Top20): {sorted(set_v12 & set_v10)} ({len(set_v12 & set_v10)}/20)')

# 保存
out = {
    'target_period': '2026196',
    'predict_time': '2026-07-25 13:07 GMT+8',
    'last_period': '2026195',
    'last_nums': sorted(data[0].tolist()),
    'v12_config': {
        'min_quota': {str(k): v for k, v in min_q_v12.items()},
        'max_quota': {str(k): v for k, v in max_q_v12.items()},
        'description': 'V10 反向特征 + Z1 高频+中高 各 2 个下限 + 类4 Z4 上限 3',
    },
    'v12_top20': top20_v12,
    'v12_top25': sorted((v12_select(0, min_q_v12, max_q_v12, 25)[0] + 1).tolist()),
    'v12_top30': sorted((v12_select(0, min_q_v12, max_q_v12, 30)[0] + 1).tolist()),
    'v10_top20': top20_v10,
    'consensus_top20': sorted(set_v12 & set_v10),
    'backtest_200periods': {
        'v12_top20_avg': 5.345,
        'v12_top20_ge10_pct': 3.5,
        'v10_top20_avg': 5.425,
        'v10_top20_ge10_pct': 1.0,
    }
}
with open('/Users/clarkkong/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery/data/backtest/2026196_predictions_v12.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('预测已保存到 data/backtest/2026196_predictions_v12.json')
EOF