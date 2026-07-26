"""
KL8 V13 修正版: 配额上限保护
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

TEST_INDICES = list(range(0, 200))
N_TEST = len(TEST_INDICES)

F_FREQ_100 = np.array([feat_freq(i, 100) for i in TEST_INDICES])
F_CLASS = np.zeros((N_TEST, 80), dtype=int)
ZONE = np.array([((n-1)//20) + 1 for n in range(1, 81)], dtype=int)

for k in range(N_TEST):
    F_CLASS[k] = classify_freq(F_FREQ_100[k], [0.25]*4)

def score_v10(k):
    i = TEST_INDICES[k]
    return (feat_freq(i, 10) * -3.0 +
            feat_repeat(i, 3) * -3.0 +
            feat_neighbor(i, 5) * -3.0)

# ============================================================
# V13 修正版: 配额归一化
# ============================================================
def v13_select_v2(k, S, top_n=20, fixed_q=None, beta_short=0.5):
    """
    V13 v2: 配额归一化到 top_n
    fixed_q: 固定配额 {(c, z): count}, None = 不加固定
    beta_short: 短期配额的权重 (0 = 无短期, 1 = 完全短期)
    """
    cls = F_CLASS[k]
    s = score_v10(k)
    sorted_idx = np.argsort(-s)
    i = TEST_INDICES[k]
    
    # 1. 计算最终配额
    quota = {}
    if fixed_q:
        for (c, z), q in fixed_q.items():
            quota[(c, z)] = q * 1.0
    
    if beta_short > 0 and i + S < n_periods:
        # 计算短期 S 期格子分布
        grid_count = np.zeros((4, 4), dtype=int)
        for j in range(1, S+1):
            if i+j >= n_periods: break
            nums = data[i+j]
            for n in nums:
                c = cls[n-1]
                z = ZONE[n-1]
                grid_count[c-1, z-1] += 1
        # 归一化: S 期每期 20 个, 总共 S*20
        # 平均每格子 (S * 20) / 16
        avg_per_grid = S * 20 / 16  # ≈ 3.75 for S=3
        for c in [1, 2, 3, 4]:
            for z in [1, 2, 3, 4]:
                # 短期贡献 = beta_short * (grid_count - avg)
                short_contrib = beta_short * (grid_count[c-1, z-1] - avg_per_grid)
                key = (c, z)
                if key in quota:
                    quota[key] += short_contrib
                else:
                    quota[key] = short_contrib
    
    # 2. 归一化 quota 到 top_n
    total = sum(max(0, q) for q in quota.values())
    if total == 0:
        # 全是 0, 用均匀
        quota = {(c, z): 1 for c in [1,2,3,4] for z in [1,2,3,4]}
        total = 16
    
    scale = top_n / total
    quota_int = {}
    for k2, v in quota.items():
        if v <= 0:
            quota_int[k2] = 0
        else:
            quota_int[k2] = max(0, int(round(v * scale)))
    
    # 3. 调整到精确 top_n
    diff = top_n - sum(quota_int.values())
    if diff != 0:
        # 按比例调整
        if diff > 0:
            sorted_keys = sorted(quota_int.keys(), key=lambda x: -quota[x])
        else:
            sorted_keys = sorted(quota_int.keys(), key=lambda x: quota_int[x])
        idx = 0
        attempts = 0
        while diff != 0 and attempts < 200:
            k2 = sorted_keys[idx % len(sorted_keys)]
            if diff > 0:
                quota_int[k2] += 1
                diff -= 1
            elif diff < 0 and quota_int[k2] > 0:
                quota_int[k2] -= 1
                diff += 1
            idx += 1
            attempts += 1
    
    # 4. 选号
    selected = []
    for (c, z), q in quota_int.items():
        if q <= 0: continue
        mask = (cls == c) & (ZONE == z)
        indices_in_group = np.where(mask)[0]
        if len(indices_in_group) == 0: continue
        group_scores = s[indices_in_group]
        order = np.argsort(group_scores)
        for p in indices_in_group[order[:q]]:
            if p not in selected:
                selected.append(int(p))
                if len(selected) >= top_n: break
        if len(selected) >= top_n: break
    
    # 5. 补足
    for idx in sorted_idx:
        if len(selected) >= top_n: break
        if int(idx) in selected: continue
        selected.append(int(idx))
    
    return np.array(selected[:top_n])

# 评估
def eval_v13_v2(S, top_n=20, fixed_q=None, beta_short=0.5):
    hits = []
    for k in range(N_TEST):
        actual = set(data[TEST_INDICES[k]].tolist())
        sel = v13_select_v2(k, S, top_n, fixed_q, beta_short)
        hits.append(len(set((sel + 1).tolist()) & actual))
    h = np.array(hits)
    return h.mean(), (h>=8).mean()*100, (h>=10).mean()*100, h

# V10 基线
v10_h = []
for k in range(N_TEST):
    actual = set(data[TEST_INDICES[k]].tolist())
    s = score_v10(k)
    top = set((np.argsort(-s)[:20] + 1).tolist())
    v10_h.append(len(top & actual))
v10_h = np.array(v10_h)

# V12 固定配额
def v12_fixed(k, top_n=20):
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
    sel = v12_fixed(k)
    v12_h.append(len(set((sel + 1).tolist()) & actual))
v12_h = np.array(v12_h)

# 测试
print('=' * 80)
print('V13 修正版: 短期 S 期动态配额 (200 期 Top20)')
print('=' * 80)
print(f'{"方案":<40} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')
print(f'{"V10 基线":<40} {v10_h.mean():>6.3f} {(v10_h>=8).mean()*100:>5.1f}% {(v10_h>=10).mean()*100:>6.1f}%')
print(f'{"V12 固定配额":<40} {v12_h.mean():>6.3f} {(v12_h>=8).mean()*100:>5.1f}% {(v12_h>=10).mean()*100:>6.1f}%')
print()

print('--- V13 仅短期配额 (无固定, beta_short=1) ---')
for S in [3, 5, 7, 10]:
    avg, ge8, ge10, _ = eval_v13_v2(S, 20, None, beta_short=1.0)
    print(f'{"V13 S="+str(S)+" beta=1.0":<40} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

print()
print('--- V13 混合 (固定配额 + 短期微调) ---')
for beta in [0.1, 0.3, 0.5, 0.7, 1.0]:
    avg, ge8, ge10, _ = eval_v13_v2(3, 20, {(1, 1): 2, (2, 1): 2}, beta_short=beta)
    print(f'{"V13 混合 S=3 beta="+str(beta):<40} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

print()
for beta in [0.1, 0.3, 0.5, 0.7, 1.0]:
    avg, ge8, ge10, _ = eval_v13_v2(5, 20, {(1, 1): 2, (2, 1): 2}, beta_short=beta)
    print(f'{"V13 混合 S=5 beta="+str(beta):<40} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

# 找最优组合
print()
print('=' * 80)
print('V13 找最优组合 (网格搜索 S × beta)')
print('=' * 80)

best = []
for S in [3, 4, 5, 7, 10]:
    for beta in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        avg, ge8, ge10, _ = eval_v13_v2(S, 20, {(1, 1): 2, (2, 1): 2}, beta_short=beta)
        # 综合得分 (考虑 avg 和 ≥10中)
        score = avg * 0.7 + ge10 * 0.3
        best.append((score, avg, ge8, ge10, S, beta))

best.sort(key=lambda x: -x[0])
print(f'{"方案":<40} {"avg":>6} {"≥8中":>6} {"≥10中":>7} {"综合得分":>8}')
for c in best[:10]:
    label = f'V13 S={c[4]} beta={c[5]}'
    print(f'{label:<40} {c[1]:>6.3f} {c[2]:>5.1f}% {c[3]:>6.1f}% {c[0]:>8.3f}')
EOF