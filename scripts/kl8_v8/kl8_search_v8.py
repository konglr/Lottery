"""
KL8 搜索最优组合 - 尝试突破 50% 命中率
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
data = df[red_cols].astype(int).values  # (n, 20)
n_periods = len(data)

def freq_window(start_idx, window, target_n=80):
    if start_idx + window > n_periods:
        window = n_periods - start_idx
    if window <= 0:
        return np.zeros(target_n, dtype=int)
    block = data[start_idx+1:start_idx+1+window]
    flat = block.flatten()
    freq = np.bincount(flat, minlength=target_n+1)[1:target_n+1]
    return freq

# 测试更多策略
test_indices = list(range(200, 400))

# ============= 策略 A: 朴素频次多窗口排名 =============
# 给每个号算多个窗口的得分,加权综合
print('=' * 70)
print('策略 A: 多窗口排名加权 (排名越靠前分数越高)')
print('=' * 70)
windows = [30, 75, 150, 300]
hits_list_25 = []
hits_list_30 = []
for i in test_indices:
    actual = set(data[i].tolist())
    ranks = np.zeros((80, len(windows)))
    for k, W in enumerate(windows):
        freq = freq_window(i, W)
        # 排名: 频次越高排名越高 (rank 0 = 最高)
        r = np.argsort(np.argsort(-freq))
        ranks[:, k] = r
    # 排名得分 (排名 0 = 80 分,排名 79 = 1 分)
    rank_scores = (80 - ranks).mean(axis=1)
    for top_n, lst in [(25, hits_list_25), (30, hits_list_30)]:
        top_idx = np.argsort(-rank_scores)[:top_n] + 1
        top_set = set(top_idx.tolist())
        lst.append(len(top_set & actual))

for top_n, lst in [(25, hits_list_25), (30, hits_list_30)]:
    h = np.array(lst)
    print(f'  Top{top_n}: 平均 {h.mean():.2f}/20  ≥8中 {(h>=8).mean()*100:.1f}%  ≥10中 {(h>=10).mean()*100:.1f}%')

# ============= 策略 B: 频次 + 重号 + 邻号 + 漏号 综合(greedy weight) =============
print()
print('=' * 70)
print('策略 B: 综合特征(5 类特征叠加)')
print('=' * 70)
# 不同权重组合
configs_b = [
    # (W_freq, w_repeat, w_neighbor, w_miss_bonus, w_anti_miss)
    (100, 5.0, 2.0, 2.0, -2.0),
    (100, 8.0, 3.0, 2.0, -3.0),
    (100, 6.0, 2.5, 1.5, -1.5),
    (75, 5.0, 2.0, 2.0, -2.0),
    (150, 5.0, 2.0, 2.0, -2.0),
    (100, 4.0, 1.5, 1.0, -1.0),
]
for cfg in configs_b:
    W, wr, wn, wb, wa = cfg
    for top_n in [20, 25, 30]:
        hits = []
        for i in test_indices:
            actual = set(data[i].tolist())
            freq = freq_window(i, W).astype(float)
            scores = freq.copy()
            # 重号 (近 2 期)
            for j in range(1, 3):
                nums = set(data[i+j].tolist())
                for n in nums:
                    scores[n-1] += wr
            # 邻号
            last = set(data[i+1].tolist())
            neighbors = set()
            for n in last:
                for d in [-2, -1, 1, 2]:
                    if 1 <= n+d <= 80: neighbors.add(n+d)
            for n in (neighbors - last):
                scores[n-1] += wn
            # 漏号
            miss_count = np.zeros(80)
            for j in range(min(30, i+1)):
                in_period = np.isin(np.arange(1, 81), data[i+1+j])
                miss_count = np.where(in_period, 0, miss_count + 1)
            bonus = np.where((miss_count >= 5) & (miss_count <= 20), wb, 0)
            anti = np.where(miss_count >= 25, wa, 0)
            scores += bonus + anti
            top_idx = np.argsort(-scores)[:top_n] + 1
            top_set = set(top_idx.tolist())
            hits.append(len(top_set & actual))
        h = np.array(hits)
        print(f'  cfg={cfg} Top{top_n}: 平均 {h.mean():.2f}/20  ≥8中 {(h>=8).mean()*100:.1f}%  ≥10中 {(h>=10).mean()*100:.1f}%')
    print()

# ============= 策略 C: 跨多期重号权重衰减 =============
print('=' * 70)
print('策略 C: 跨期重号衰减')
print('=' * 70)
for span in [3, 5, 7, 10]:
    for top_n in [25, 30]:
        hits = []
        for i in test_indices:
            actual = set(data[i].tolist())
            freq = freq_window(i, 100).astype(float)
            scores = freq.copy()
            # 重号: 第 j 期前出现 + weight / j
            for j in range(1, span+1):
                nums = set(data[i+j].tolist())
                w = 6.0 / j
                for n in nums:
                    scores[n-1] += w
            top_idx = np.argsort(-scores)[:top_n] + 1
            top_set = set(top_idx.tolist())
            hits.append(len(top_set & actual))
        h = np.array(hits)
        print(f'  span={span} Top{top_n}: 平均 {h.mean():.2f}/20  ≥8中 {(h>=8).mean()*100:.1f}%  ≥10中 {(h>=10).mean()*100:.1f}%')

# ============= 策略 D: 邻号 + 跨期邻号 =============
print()
print('=' * 70)
print('策略 D: 多期邻号加权')
print('=' * 70)
for span in [2, 3, 5]:
    for top_n in [25, 30]:
        hits = []
        for i in test_indices:
            actual = set(data[i].tolist())
            freq = freq_window(i, 100).astype(float)
            scores = freq.copy()
            # 跨 span 期的所有 ±2 邻号
            neighbors = set()
            for j in range(1, span+1):
                last = set(data[i+j].tolist())
                for n in last:
                    for d in [-2, -1, 1, 2]:
                        if 1 <= n+d <= 80: neighbors.add(n+d)
                # 当期号不加(避免重复加)
            for n in neighbors:
                if scores[n-1] == freq[n-1]:  # 没被重号加权过
                    scores[n-1] += 3.0
            top_idx = np.argsort(-scores)[:top_n] + 1
            top_set = set(top_idx.tolist())
            hits.append(len(top_set & actual))
        h = np.array(hits)
        print(f'  多期邻号 span={span} Top{top_n}: 平均 {h.mean():.2f}/20  ≥8中 {(h>=8).mean()*100:.1f}%  ≥10中 {(h>=10).mean()*100:.1f}%')

# ============= 策略 E: 排除极冷号 =============
print()
print('=' * 70)
print('策略 E: 排除极冷号(连续 ≥30 期未出)')
print('=' * 70)
for top_n in [20, 25, 30]:
    hits = []
    for i in test_indices:
        actual = set(data[i].tolist())
        freq = freq_window(i, 100).astype(float)
        # 计算漏号
        miss_count = np.zeros(80)
        for j in range(min(50, i+1)):
            in_period = np.isin(np.arange(1, 81), data[i+1+j])
            miss_count = np.where(in_period, 0, miss_count + 1)
        # 极冷号大幅减分
        scores = freq.copy()
        scores = np.where(miss_count >= 30, scores * 0.3, scores)
        # 重号加权
        for j in range(1, 3):
            nums = set(data[i+j].tolist())
            for n in nums:
                scores[n-1] += 5.0
        top_idx = np.argsort(-scores)[:top_n] + 1
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    h = np.array(hits)
    print(f'  排除极冷 Top{top_n}: 平均 {h.mean():.2f}/20  ≥8中 {(h>=8).mean()*100:.1f}%  ≥10中 {(h>=10).mean()*100:.1f}%')

# ============= 策略 F: 极热号+邻号 重号 =============
print()
print('=' * 70)
print('策略 F: 极端加权 — 频次 Top15 强制入 + 邻号')
print('=' * 70)
for top_n in [20, 25, 30]:
    hits = []
    for i in test_indices:
        actual = set(data[i].tolist())
        freq = freq_window(i, 100).astype(float)
        # 强制 Top15 入
        top15 = set(np.argsort(-freq)[:15].tolist())
        scores = freq.copy()
        # 剩余 5 个从邻号+重号选
        # 但更简单: 加权让 top15 排前
        scores[list(top15)] *= 2.0
        # 重号加权
        for j in range(1, 3):
            nums = set(data[i+j].tolist())
            for n in nums:
                scores[n-1] += 5.0
        top_idx = np.argsort(-scores)[:top_n] + 1
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    h = np.array(hits)
    print(f'  强制 Top15 Top{top_n}: 平均 {h.mean():.2f}/20  ≥8中 {(h>=8).mean()*100:.1f}%  ≥10中 {(h>=10).mean()*100:.1f}%')