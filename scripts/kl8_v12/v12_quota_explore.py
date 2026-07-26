"""
KL8 V12 配额选号方案 — 修正版
混合方案: 配额作为基础, 再用 TopN 调整
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np

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

# 预计算
print('预计算...')
F_FREQ_100 = np.array([feat_freq(i, 100) for i in TEST_INDICES])
F_CLASS = np.zeros((N_TEST, 80), dtype=int)
ZONE = np.array([((n-1)//20) + 1 for n in range(1, 81)], dtype=int)

for k in range(N_TEST):
    F_CLASS[k] = classify_freq(F_FREQ_100[k], [0.25]*4)

def score_v10(i):
    return (feat_freq(i, 10) * -3.0 +
            feat_repeat(i, 3) * -3.0 +
            feat_neighbor(i, 5) * -3.0)

def score_v3(i):
    return feat_freq(i, 50)

# 配额选号 (修正版)
def quota_select_v10(i, quota, top_n=20):
    """V10 反向: 选每个配额格子中分数最低(最冷)的号"""
    cls = F_CLASS[i]
    scores = score_v10(i)
    selected = []
    for c in [1, 2, 3, 4]:
        for z in [1, 2, 3, 4]:
            mask = (cls == c) & (ZONE == z)
            indices_in_group = np.where(mask)[0]
            if len(indices_in_group) == 0: continue
            q = quota.get((c, z), 0)
            if q <= 0: continue
            # 在这个组里按 scores 升序 (反向 = 选分数最低)
            group_scores = scores[indices_in_group]
            order = np.argsort(group_scores)
            selected.extend(indices_in_group[order[:q]].tolist())
    return np.array(selected)

def quota_select_v3(i, quota, top_n=20):
    """V3 正向: 选每个配额格子中分数最高(最热)的号"""
    cls = F_CLASS[i]
    scores = score_v3(i)
    selected = []
    for c in [1, 2, 3, 4]:
        for z in [1, 2, 3, 4]:
            mask = (cls == c) & (ZONE == z)
            indices_in_group = np.where(mask)[0]
            if len(indices_in_group) == 0: continue
            q = quota.get((c, z), 0)
            if q <= 0: continue
            # 在这个组里按 scores 降序 (正向)
            group_scores = scores[indices_in_group]
            order = np.argsort(-group_scores)
            selected.extend(indices_in_group[order[:q]].tolist())
    return np.array(selected)

# 评估
def eval_method(select_fn, top_n=20):
    hits = []
    for k in range(N_TEST):
        actual = set(data[TEST_INDICES[k]].tolist())
        selected = select_fn(k)
        selected_set = set((selected + 1).tolist())
        hits.append(len(selected_set & actual))
    h = np.array(hits)
    return h.mean(), (h>=8).mean()*100, (h>=10).mean()*100, h

# 基线
print()
print('=' * 80)
print('V12 配额方案对比 (200 期)')
print('=' * 80)
print(f'{"方案":<40} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')

# V10 基线
def v10_baseline(k):
    s = score_v10(k)
    return np.argsort(-s)[:20]
avg, ge8, ge10, _ = eval_method(v10_baseline)
print(f'{"V10 基线 (无配额)":<40} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

# V3 基线
def v3_baseline(k):
    s = score_v3(k)
    return np.argsort(-s)[:20]
avg, ge8, ge10, _ = eval_method(v3_baseline)
print(f'{"V3 基线 (无配额)":<40} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

# 配额方案 (基于 V10)
print()
print('--- V10 配额方案 ---')

# 配额 A: 均匀 (4 类各 5 个, 每区均匀分)
# 4 类 × 4 区 = 16 格, 每格约 1.25 个
quota_A = {}
for c in [1, 2, 3, 4]:
    for z in [1, 2, 3, 4]:
        quota_A[(c, z)] = 1  # 共 16
# 还差 4 个, 加到中间类
quota_A[(2, 2)] = 2
quota_A[(3, 2)] = 2
quota_A[(2, 3)] = 2
quota_A[(3, 3)] = 2
# 16+4 = 20 ✓

# 配额 B: 低频区 (类 4) 多选, 高频区 (类 1) 少选
quota_B = {
    (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0,
    (2, 1): 1, (2, 2): 1, (2, 3): 1, (2, 4): 1,
    (3, 1): 1, (3, 2): 1, (3, 3): 1, (3, 4): 1,
    (4, 1): 2, (4, 2): 3, (4, 3): 3, (4, 4): 3,
}
# 0+4+4+(2+3+3+3) = 4+4+11 = 19, 加 1
quota_B[(4, 1)] = 3
# 0+4+4+(3+3+3+3) = 4+4+12 = 20 ✓

# 配额 C: Z4 区 (61-80) 多选
quota_C = {}
for c in [1, 2, 3, 4]:
    for z in [1, 2, 3, 4]:
        quota_C[(c, z)] = 1
# Z1-Z3 各 4 个 = 12, Z4 要 8 个 → 4 格各 +1
quota_C[(1, 4)] += 1
quota_C[(2, 4)] += 1
quota_C[(3, 4)] += 1
quota_C[(4, 4)] += 1
# 16+4 = 20 ✓

# 配额 D: 极端 — 完全只在类 4 (低频区) 选
quota_D = {}
for c in [1, 2, 3]:
    for z in [1, 2, 3, 4]:
        quota_D[(c, z)] = 0
for z in [1, 2, 3, 4]:
    quota_D[(4, z)] = 5  # 每区 5 个, 共 20

# 配额 E: 类 4 + Z4 交集强化 (双重低频)
quota_E = {
    (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 1,
    (2, 1): 0, (2, 2): 0, (2, 3): 0, (2, 4): 1,
    (3, 1): 1, (3, 2): 1, (3, 3): 1, (3, 4): 2,
    (4, 1): 3, (4, 2): 3, (4, 3): 3, (4, 4): 4,
}
# 0+0+0+1+0+0+0+1+(1+1+1+2)+(3+3+3+4) = 2+5+13 = 20 ✓

for label, quota in [
    ('配额 A: 均匀 (V10)', quota_A),
    ('配额 B: 低频区主导 (V10)', quota_B),
    ('配额 C: Z4 区主导 (V10)', quota_C),
    ('配额 D: 极端 — 仅类 4 (V10)', quota_D),
    ('配额 E: 类4 + Z4 强化 (V10)', quota_E),
]:
    avg, ge8, ge10, _ = eval_method(lambda k, q=quota: quota_select_v10(k, q))
    print(f'{label:<40} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

# 配额方案 (基于 V3)
print()
print('--- V3 配额方案 ---')

# 配额 F: 高频区 (类 1) 多选 (V3 正向, 选高频号)
quota_F = {
    (1, 1): 2, (1, 2): 2, (1, 3): 2, (1, 4): 2,  # 8
    (2, 1): 1, (2, 2): 1, (2, 3): 1, (2, 4): 1,  # 4
    (3, 1): 1, (3, 2): 1, (3, 3): 1, (3, 4): 1,  # 4
    (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 1,  # 4
}

# 配额 G: 极端 — 仅类 1 (高频区) 选 (V3)
quota_G = {}
for c in [2, 3, 4]:
    for z in [1, 2, 3, 4]:
        quota_G[(c, z)] = 0
for z in [1, 2, 3, 4]:
    quota_G[(1, z)] = 5

# 配额 H: 高频 + Z1 强化
quota_H = {
    (1, 1): 3, (1, 2): 2, (1, 3): 2, (1, 4): 1,
    (2, 1): 1, (2, 2): 1, (2, 3): 1, (2, 4): 1,
    (3, 1): 1, (3, 2): 1, (3, 3): 1, (3, 4): 1,
    (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 1,
}
# 8+4+4+4 = 20 ✓

for label, quota in [
    ('配额 F: 高频主导 (V3)', quota_F),
    ('配额 G: 极端 — 仅类 1 (V3)', quota_G),
    ('配额 H: 高频 + Z1 强化 (V3)', quota_H),
]:
    avg, ge8, ge10, _ = eval_method(lambda k, q=quota: quota_select_v3(k, q))
    print(f'{label:<40} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

EOF