"""
KL8 V13 v3: 用历史数据学到的格子相关性作为权重
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
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

def score_v10(k):
    i = TEST_INDICES[k]
    return (feat_freq(i, 10) * -3.0 +
            feat_repeat(i, 3) * -3.0 +
            feat_neighbor(i, 5) * -3.0)

# ============================================================
# 核心问题: 短期配额 vs V10 顺序冲突
# ============================================================
# 思路: V10 按分数排序, 但配额调整某些格子的优先级
# 解决方案: V10 分数 + 短期加权 (而不是替换)

def v13_v3_select(k, S, top_n=20, w_short=1.0):
    """
    V13 v3: V10 分数 + 短期配额加权
    短期 S 期格子出号多的 → 格子内所有号分数加权 +w_short
    """
    cls = F_CLASS[k]
    s = score_v10(k)
    i = TEST_INDICES[k]
    
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
        
        # 归一化: 每格子出号数 - 平均 (S * 20/16)
        avg = S * 20 / 16
        # 加权: 在格子 (c, z) 内的所有号, score += w_short * (grid_count - avg)
        for c in [1, 2, 3, 4]:
            for z in [1, 2, 3, 4]:
                mask = (cls == c) & (ZONE == z)
                s[mask] += w_short * (grid_count[c-1, z-1] - avg)
    
    sorted_idx = np.argsort(-s)
    return sorted_idx[:top_n]

# 评估
def eval_v13_v3(S, top_n=20, w_short=1.0):
    hits = []
    for k in range(N_TEST):
        actual = set(data[TEST_INDICES[k]].tolist())
        top = set((v13_v3_select(k, S, top_n, w_short) + 1).tolist())
        hits.append(len(top & actual))
    h = np.array(hits)
    return h.mean(), (h>=8).mean()*100, (h>=10).mean()*100, h

# 基线
v10_h = []
for k in range(N_TEST):
    actual = set(data[TEST_INDICES[k]].tolist())
    s = score_v10(k)
    top = set((np.argsort(-s)[:20] + 1).tolist())
    v10_h.append(len(top & actual))
v10_h = np.array(v10_h)

# V12 固定配额
def v12_select(k, top_n=20):
    cls = F_CLASS[k]
    s = score_v10(k)
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
        if len(selected) >= top_n: break
        if int(idx) in selected: continue
        selected.append(int(idx))
    return np.array(selected[:top_n])

v12_h = []
for k in range(N_TEST):
    actual = set(data[TEST_INDICES[k]].tolist())
    sel = v12_select(k)
    v12_h.append(len(set((sel + 1).tolist()) & actual))
v12_h = np.array(v12_h)

# 测试
print('=' * 80)
print('V13 v3: V10 分数 + 短期配额加权 (短期多的格子分数加权)')
print('=' * 80)
print(f'{"方案":<40} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')
print(f'{"V10 基线":<40} {v10_h.mean():>6.3f} {(v10_h>=8).mean()*100:>5.1f}% {(v10_h>=10).mean()*100:>6.1f}%')
print(f'{"V12 固定配额":<40} {v12_h.mean():>6.3f} {(v12_h>=8).mean()*100:>5.1f}% {(v12_h>=10).mean()*100:>6.1f}%')

print()
print('--- V13 v3 不同 S 和 w_short ---')
for S in [3, 5, 7, 10]:
    for w in [0.5, 1.0, 2.0, 3.0, 5.0]:
        avg, ge8, ge10, _ = eval_v13_v3(S, 20, w)
        label = f'V13v3 S={S} w={w}'
        print(f'{label:<40} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

# 也试试: 用相关系数作为权重 (而不是 grid_count)
print()
print('=' * 80)
print('V13 v4: 用历史数据学到的相关系数作为权重')
print('=' * 80)

# 先学历史相关系数
def learn_correlations(S):
    """学 16 格子的相关系数"""
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

corrs_s5 = learn_correlations(5)
print(f'S=5 学习的 16 格子相关系数:')
for c in [1, 2, 3, 4]:
    for z in [1, 2, 3, 4]:
        if abs(corrs_s5[c-1, z-1]) > 0.15:
            print(f'  类{c}×Z{z}: {corrs_s5[c-1, z-1]:+.3f} (强信号)')

# V13 v4: 用相关系数 * 格子出号数 作为加权
def v13_v4_select(k, S, corrs, top_n=20, w_corr=3.0):
    cls = F_CLASS[k]
    s = score_v10(k)
    i = TEST_INDICES[k]
    
    if i + S < n_periods:
        grid_count = np.zeros((4, 4), dtype=int)
        for j in range(1, S+1):
            if i+j >= n_periods: break
            nums = data[i+j]
            for n in nums:
                c = cls[n-1]
                z = ZONE[n-1]
                grid_count[c-1, z-1] += 1
        
        # 归一化格子出号数
        for c in [1, 2, 3, 4]:
            for z in [1, 2, 3, 4]:
                mask = (cls == c) & (ZONE == z)
                # 格子内分数加权 = 相关系数 * (grid_count / avg) * w_corr
                # 仅对相关系数 > 0.1 的格子做加权
                if abs(corrs[c-1, z-1]) > 0.1:
                    avg_grid = S * 20 / 16
                    delta = (grid_count[c-1, z-1] - avg_grid) / avg_grid
                    s[mask] += w_corr * corrs[c-1, z-1] * delta
    
    sorted_idx = np.argsort(-s)
    return sorted_idx[:top_n]

# 测试
print()
for S in [3, 5, 7]:
    corrs = learn_correlations(S)
    for w in [1.0, 3.0, 5.0, 10.0]:
        hits = []
        for k in range(N_TEST):
            actual = set(data[TEST_INDICES[k]].tolist())
            top = set((v13_v4_select(k, S, corrs, 20, w) + 1).tolist())
            hits.append(len(top & actual))
        h = np.array(hits)
        print(f'V13v4 S={S} w={w}: avg={h.mean():.3f} ≥8中 {(h>=8).mean()*100:.1f}% ≥10中 {(h>=10).mean()*100:.1f}%')
EOF