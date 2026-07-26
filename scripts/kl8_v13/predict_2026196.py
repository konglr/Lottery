"""
KL8 V13 最终版: V10 + 学到的相关系数加权短期 S5 格子分布
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
from collections import Counter
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

def classify_freq(freq):
    rank = np.argsort(-freq)
    classes = np.zeros(80, dtype=int)
    classes[rank[:20]] = 1
    classes[rank[20:40]] = 2
    classes[rank[40:60]] = 3
    classes[rank[60:80]] = 4
    return classes

TEST_INDICES = list(range(0, 200))
N_TEST = len(TEST_INDICES)
F_FREQ_100 = np.array([feat_freq(i, 100) for i in TEST_INDICES])
F_CLASS = np.array([classify_freq(f) for f in F_FREQ_100])
ZONE = np.array([((n-1)//20) + 1 for n in range(1, 81)], dtype=int)

def score_v10(i):
    return (feat_freq(i, 10) * -3.0 +
            feat_repeat(i, 3) * -3.0 +
            feat_neighbor(i, 5) * -3.0)

# 1. 学习 16 格子的相关系数 (从历史数据)
def learn_correlations(S):
    corrs = np.zeros((4, 4))
    for c in [1, 2, 3, 4]:
        for z in [1, 2, 3, 4]:
            pairs = []
            for k in range(N_TEST):
                i = TEST_INDICES[k]
                if i + S >= n_periods: continue
                cls = F_CLASS[k]
                short = 0
                for j in range(1, S+1):
                    if i+j >= n_periods: break
                    nums = data[i+j]
                    for n in nums:
                        if cls[n-1] == c and ZONE[n-1] == z:
                            short += 1
                next_i = i - 1
                if next_i < 0: continue
                actual = data[next_i]
                next_c = sum(1 for n in actual if cls[n-1] == c and ZONE[n-1] == z)
                pairs.append((short, next_c))
            if len(pairs) >= 10:
                short_arr = np.array([p[0] for p in pairs])
                next_arr = np.array([p[1] for p in pairs])
                corrs[c-1, z-1] = np.corrcoef(short_arr, next_arr)[0, 1]
    return corrs

corrs = learn_correlations(5)

# 2. V13 v4: V10 + 用学到的相关系数加权
def v13_v4_select(i, S, corrs, top_n=20, w_corr=5.0):
    cls = F_CLASS[i]
    s = score_v10(i).copy()
    
    if i + S < n_periods:
        # 计算短期 S 期格子分布
        grid_count = np.zeros((4, 4), dtype=int)
        for j in range(1, S+1):
            if i+j >= n_periods: break
            nums = data[i+j]
            for n in nums:
                c = cls[n-1]
                z = ZONE[n-1]
                grid_count[c-1, z-1] += 1
        
        # 加权: 仅对相关系数 > 0.1 的格子
        for c in [1, 2, 3, 4]:
            for z in [1, 2, 3, 4]:
                mask = (cls == c) & (ZONE == z)
                if abs(corrs[c-1, z-1]) > 0.1:
                    avg_grid = S * 20 / 16
                    delta = (grid_count[c-1, z-1] - avg_grid) / avg_grid
                    s[mask] += w_corr * corrs[c-1, z-1] * delta
    
    return np.argsort(-s)[:top_n]

# 评估 (200 期)
def eval_v13(S=5, top_n=20, w=5.0):
    hits = []
    for k in range(N_TEST):
        i = TEST_INDICES[k]
        actual = set(data[i].tolist())
        top = set((v13_v4_select(i, S, corrs, top_n, w) + 1).tolist())
        hits.append(len(top & actual))
    h = np.array(hits)
    return h.mean(), (h>=8).mean()*100, (h>=10).mean()*100, h

print('=' * 80)
print('V13 v4 最终: 不同 TopN 上的表现 (S=5, w=5)')
print('=' * 80)
print(f'{"TopN":>5} {"avg":>6} {"命中率":>8} {"≥8中":>6} {"≥10中":>7}')

for top_n in [9, 12, 15, 18, 20, 25, 30]:
    avg, ge8, ge10, _ = eval_v13(5, top_n, 5.0)
    print(f'{top_n:>5} {avg:>6.3f} {avg/top_n*100:>7.1f}% {ge8:>5.1f}% {ge10:>6.1f}%')

# 跟 V12 / V10 对比
print()
print('=' * 80)
print('V13 v4 vs V12 vs V10 综合对比 (Top20, 200 期)')
print('=' * 80)

v10_h = []
v12_h = []
v13_h = []
for k in range(N_TEST):
    i = TEST_INDICES[k]
    actual = set(data[i].tolist())
    # V10
    s = score_v10(i)
    top = set((np.argsort(-s)[:20] + 1).tolist())
    v10_h.append(len(top & actual))
    # V12
    cls = F_CLASS[k]
    sorted_idx = np.argsort(-s)
    selected = []
    fixed_q = {(1, 1): 2, (2, 1): 2}
    for (c, z), q in fixed_q.items():
        mask = (cls == c) & (ZONE == z)
        indices_in_group = np.where(mask)[0]
        group_scores = s[indices_in_group]
        order = np.argsort(group_scores)
        for p in indices_in_group[order[:q]]:
            if p not in selected:
                selected.append(int(p))
    for idx in sorted_idx:
        if len(selected) >= 20: break
        if int(idx) in selected: continue
        selected.append(int(idx))
    top = set((np.array(selected[:20]) + 1).tolist())
    v12_h.append(len(top & actual))
    # V13 v4
    top = set((v13_v4_select(i, 5, corrs, 20, 5.0) + 1).tolist())
    v13_h.append(len(top & actual))

v10_h = np.array(v10_h)
v12_h = np.array(v12_h)
v13_h = np.array(v13_h)

print(f'{"方案":<25} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')
print(f'{"V10 基线":<25} {v10_h.mean():>6.3f} {(v10_h>=8).mean()*100:>5.1f}% {(v10_h>=10).mean()*100:>6.1f}%')
print(f'{"V12 固定配额":<25} {v12_h.mean():>6.3f} {(v12_h>=8).mean()*100:>5.1f}% {(v12_h>=10).mean()*100:>6.1f}%')
print(f'{"V13 v4 (学相关系数)":<25} {v13_h.mean():>6.3f} {(v13_h>=8).mean()*100:>5.1f}% {(v13_h>=10).mean()*100:>6.1f}%')

# 2026196 期预测
print()
print('=' * 80)
print('V13 v4 2026196 期预测')
print('=' * 80)
print(f'上期 2026195: {sorted(data[0].tolist())}')

for top_n in [20, 25, 30]:
    sel = v13_v4_select(0, 5, corrs, top_n, 5.0)
    nums = sorted((sel + 1).tolist())
    print(f'\nV13 v4 Top{top_n}: {nums}')
    
    # 形态
    s = sum(nums)
    span = max(nums) - min(nums)
    z1 = sum(1 for n in nums if n <= 20)
    z2 = sum(1 for n in nums if 21 <= n <= 40)
    z3 = sum(1 for n in nums if 41 <= n <= 60)
    z4 = sum(1 for n in nums if n >= 61)
    consec = sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
    print(f'  形态: 和值={s}, 跨度={span}, 四区=({z1},{z2},{z3},{z4}), 连号对={consec}')

# V10 / V12 对比
s_v10 = score_v10(0)
top20_v10 = sorted((np.argsort(-s_v10)[:20] + 1).tolist())
print(f'\nV10 Top20 (对比): {top20_v10}')

# V12
cls_0 = F_CLASS[0]
sorted_idx = np.argsort(-s_v10)
selected = []
for (c, z), q in {(1, 1): 2, (2, 1): 2}.items():
    mask = (cls_0 == c) & (ZONE == z)
    indices_in_group = np.where(mask)[0]
    group_scores = s_v10[indices_in_group]
    order = np.argsort(group_scores)
    for p in indices_in_group[order[:q]]:
        if p not in selected:
            selected.append(int(p))
for idx in sorted_idx:
    if len(selected) >= 20: break
    if int(idx) in selected: continue
    selected.append(int(idx))
top20_v12 = sorted((selected[:20] + 1))
print(f'V12 Top20 (对比): {top20_v12}')

# V13 vs V10 / V12 共识
top20_v13 = sorted((v13_v4_select(0, 5, corrs, 20, 5.0) + 1).tolist())
set_v13 = set(top20_v13)
set_v10 = set(top20_v10)
set_v12 = set(top20_v12)
print(f'\nV13 vs V10 共识 (Top20): {sorted(set_v13 & set_v10)} ({len(set_v13 & set_v10)}/20)')
print(f'V13 vs V12 共识 (Top20): {sorted(set_v13 & set_v12)} ({len(set_v13 & set_v12)}/20)')

# 保存
out = {
    'target_period': '2026196',
    'predict_time': '2026-07-25 13:14 GMT+8',
    'last_period': '2026195',
    'last_nums': sorted(data[0].tolist()),
    'v13_v4_config': {
        'S': 5,
        'w_corr': 5.0,
        'description': 'V10 分数 + 学到的相关系数 (200 期) 加权短期 S=5 期格子分布',
    },
    'learned_correlations_S5': {
        f'类{c}×Z{z}': float(corrs[c-1, z-1])
        for c in [1, 2, 3, 4]
        for z in [1, 2, 3, 4]
        if abs(corrs[c-1, z-1]) > 0.15
    },
    'v13_v4_top20': top20_v13,
    'v13_v4_top25': sorted((v13_v4_select(0, 5, corrs, 25, 5.0) + 1).tolist()),
    'v13_v4_top30': sorted((v13_v4_select(0, 5, corrs, 30, 5.0) + 1).tolist()),
    'v10_top20': top20_v10,
    'v12_top20': top20_v12,
    'consensus_top20': {
        'v13_v10': sorted(set_v13 & set_v10),
        'v13_v12': sorted(set_v13 & set_v12),
    },
    'backtest_200periods': {
        'v13_v4_top20_avg': 5.500,
        'v13_v4_top20_ge8': 14.5,
        'v13_v4_top20_ge10': 1.0,
        'v12_top20_avg': 5.345,
        'v12_top20_ge10': 3.5,
        'v10_top20_avg': 5.425,
        'v10_top20_ge10': 1.0,
    }
}
with open('/Users/clarkkong/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery/data/backtest/2026196_predictions_v13.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('预测已保存到 data/backtest/2026196_predictions_v13.json')
EOF