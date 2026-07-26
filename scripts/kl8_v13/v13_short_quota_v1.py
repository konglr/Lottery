"""
KL8 V13 短期配额方案 - 修正版
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
# V13: 短期 S 期格子分布 → 动态配额
# ============================================================
def v13_select(k, S, top_n=20, alpha=1.0):
    """
    k: 测试期索引
    S: 短期窗口 (3 或 5)
    alpha: 短期配额强度 (0 = 无配额, 1 = 完全用短期)
    """
    i = TEST_INDICES[k]
    cls = F_CLASS[k]
    s = score_v10(k)
    
    # 1. 计算短期 S 期的格子分布
    if i + S >= n_periods:
        return None
    
    # 用预测期的 4 分类 (基于 freq_100)
    # 统计短期 S 期每个格子的实际出号数
    grid_count = np.zeros((4, 4), dtype=int)
    for j in range(1, S+1):
        if i+j >= n_periods: break
        nums = data[i+j]
        for n in nums:
            c = cls[n-1]
            z = ZONE[n-1]
            grid_count[c-1, z-1] += 1
    
    # 2. 归一化到 top_n, 加权 (alpha)
    # 期望: 每格子预期 S * 5/16 ≈ S * 0.31
    # 如果 alpha=1, 配额 = S * grid_count / 5 (即每期平均 5 个, 总共 S*5*16 个格子分布)
    total_in_short = grid_count.sum()  # ≈ S * 20
    if total_in_short == 0:
        return None
    
    quota = {}
    for c in [1, 2, 3, 4]:
        for z in [1, 2, 3, 4]:
            q = grid_count[c-1, z-1]
            quota[(c, z)] = q
    
    # 归一化: 每个配额按比例缩放到 top_n
    scale = top_n / total_in_short
    quota_scaled = {}
    for k2, v in quota.items():
        quota_scaled[k2] = max(0, int(round(v * scale * alpha)))
    
    # 调整到 top_n
    diff = top_n - sum(quota_scaled.values())
    if diff != 0:
        sorted_keys = sorted(quota_scaled.keys(), key=lambda x: -quota_scaled[x])
        idx = 0
        while diff != 0 and idx < 100:
            k2 = sorted_keys[idx % len(sorted_keys)]
            if diff > 0:
                quota_scaled[k2] += 1
                diff -= 1
            elif diff < 0 and quota_scaled[k2] > 0:
                quota_scaled[k2] -= 1
                diff += 1
            idx += 1
    
    # 3. 按配额选号
    sorted_idx = np.argsort(-s)
    selected = []
    cell_count = Counter()
    
    # 满足配额
    for (c, z), q in quota_scaled.items():
        if q <= 0: continue
        mask = (cls == c) & (ZONE == z)
        indices_in_group = np.where(mask)[0]
        if len(indices_in_group) == 0: continue
        group_scores = s[indices_in_group]
        order = np.argsort(group_scores)
        for p in indices_in_group[order[:q]]:
            if p not in selected:
                selected.append(p)
                cell_count[(c, z)] += 1
    
    # 补足到 top_n
    for idx in sorted_idx:
        if len(selected) >= top_n: break
        if idx in selected: continue
        selected.append(idx)
    
    return np.array(selected)

# 评估
def eval_v13(S, top_n=20, alpha=1.0):
    hits = []
    for k in range(N_TEST):
        actual = set(data[TEST_INDICES[k]].tolist())
        selected = v13_select(k, S, top_n, alpha)
        if selected is None: continue
        hits.append(len(set((selected + 1).tolist()) & actual))
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
                selected.append(p)
    for idx in sorted_idx:
        if len(selected) >= top_n: break
        if idx in selected: continue
        selected.append(idx)
    return np.array(selected)

v12_h = []
for k in range(N_TEST):
    actual = set(data[TEST_INDICES[k]].tolist())
    sel = v12_select(k)
    v12_h.append(len(set((sel + 1).tolist()) & actual))
v12_h = np.array(v12_h)

# ============================================================
# 测试 V13 不同 S 和 alpha
# ============================================================
print('=' * 80)
print('V13: 短期 S 期动态配额对比 (200 期 Top20)')
print('=' * 80)
print(f'{"方案":<35} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')
print(f'{"V10 基线 (无配额)":<35} {v10_h.mean():>6.3f} {(v10_h>=8).mean()*100:>5.1f}% {(v10_h>=10).mean()*100:>6.1f}%')
print(f'{"V12 固定配额":<35} {v12_h.mean():>6.3f} {(v12_h>=8).mean()*100:>5.1f}% {(v12_h>=10).mean()*100:>6.1f}%')
print()
print('--- V13 短期配额 (alpha=1.0) ---')
for S in [3, 4, 5, 7, 10]:
    avg, ge8, ge10, _ = eval_v13(S, 20, alpha=1.0)
    print(f'{"V13 S="+str(S)+" alpha=1.0 Top20":<35} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

print()
print('--- V13 短期配额 (不同 alpha 强度) ---')
for S in [3, 5]:
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        avg, ge8, ge10, _ = eval_v13(S, 20, alpha=alpha)
        print(f'{"V13 S="+str(S)+" alpha="+str(alpha)+" Top20":<35} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

# 混合: 固定配额 + 短期配额
print()
print('=' * 80)
print('V13 混合: 固定配额 + 短期配额')
print('=' * 80)

def v13_mixed(k, S, fixed_q, alpha_short, top_n=20):
    cls = F_CLASS[k]
    s = score_v10(k)
    sorted_idx = np.argsort(-s)
    selected = []
    cell_count = Counter()
    
    # 1. 先满足固定配额
    for (c, z), q in fixed_q.items():
        mask = (cls == c) & (ZONE == z)
        indices_in_group = np.where(mask)[0]
        group_scores = s[indices_in_group]
        order = np.argsort(group_scores)
        for p in indices_in_group[order[:q]]:
            if p not in selected:
                selected.append(p)
                cell_count[(c, z)] += 1
    
    # 2. 补充短期配额
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
        # 短期配额 = alpha_short * grid_count
        for c in [1, 2, 3, 4]:
            for z in [1, 2, 3, 4]:
                short_q = int(round(grid_count[c-1, z-1] * alpha_short))
                if short_q <= 0: continue
                needed = max(0, short_q - cell_count[(c, z)])
                if needed <= 0: continue
                mask = (cls == c) & (ZONE == z)
                indices_in_group = np.where(mask)[0]
                group_scores = s[indices_in_group]
                order = np.argsort(group_scores)
                added = 0
                for p in indices_in_group[order]:
                    if added >= needed: break
                    if p not in selected:
                        selected.append(p)
                        cell_count[(c, z)] += 1
                        added += 1
    
    # 3. 补足到 top_n
    for idx in sorted_idx:
        if len(selected) >= top_n: break
        if idx in selected: continue
        selected.append(idx)
    
    return np.array(selected)

def eval_mixed(S, alpha_short):
    hits = []
    for k in range(N_TEST):
        actual = set(data[TEST_INDICES[k]].tolist())
        sel = v13_mixed(k, S, {(1, 1): 2, (2, 1): 2}, alpha_short)
        hits.append(len(set((sel + 1).tolist()) & actual))
    h = np.array(hits)
    return h.mean(), (h>=8).mean()*100, (h>=10).mean()*100, h

print(f'{"方案":<45} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')
for S in [3, 5]:
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        avg, ge8, ge10, _ = eval_mixed(S, alpha)
        print(f'{"V13 混合 S="+str(S)+" alpha="+str(alpha)+" Top20":<45} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')
EOF