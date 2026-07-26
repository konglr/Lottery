"""
KL8 探索 TopN = 25~40 的命中率,看用户"50% 命中率"的可行版本
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

def freq_window(start_idx, window):
    if start_idx + window > n_periods:
        window = n_periods - start_idx
    if window <= 0:
        return np.zeros(80, dtype=int)
    block = data[start_idx+1:start_idx+1+window]
    flat = block.flatten()
    return np.bincount(flat, minlength=81)[1:81]

# 拓展 TopN 到 40,看哪个 N 能稳定达到 50% 平均命中率
print('=' * 70)
print('不同 TopN 的平均命中率(200 期回测)')
print('=' * 70)
print(f'{"TopN":>5} {"W":>5} {"平均":>8} {"命中率":>8} {"≥随机期望":>10}')

for W in [50, 75, 100, 150]:
    print(f'\n--- W={W} ---')
    for top_n in [25, 30, 32, 35, 38, 40, 45, 50]:
        hits = []
        for i in range(0, 200):
            actual = set(data[i].tolist())
            freq = freq_window(i, W)
            top_idx = np.argsort(-freq)[:top_n] + 1
            top_set = set(top_idx.tolist())
            hits.append(len(top_set & actual))
        h = np.array(hits)
        rate = h.mean() / 20 * 100
        rand = top_n * 20 / 80
        print(f'{top_n:>5} {W:>5} {h.mean():>8.2f} {rate:>7.1f}% {(h.mean() - rand):>+10.2f}')

# 现在找"命中率 50%+"的真正含义:
# 如果 TopN=40,平均命中 10/20 = 50% 命中率
# W=100, TopN=40 时实际表现
print()
print('=' * 70)
print('TopN=40 时不同窗口的命中率')
print('=' * 70)
for W in [50, 75, 100, 150, 200]:
    hits = []
    for i in range(0, 200):
        actual = set(data[i].tolist())
        freq = freq_window(i, W)
        top_idx = np.argsort(-freq)[:40] + 1
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    h = np.array(hits)
    ge10 = (h >= 10).mean() * 100
    ge12 = (h >= 12).mean() * 100
    ge15 = (h >= 15).mean() * 100
    print(f'W={W} Top40: 平均 {h.mean():.2f}/20 (50%基准)  ≥10中 {ge10:.1f}%  ≥12中 {ge12:.1f}%  ≥15中 {ge15:.1f}%')

# TopN=50 也试一下
print()
print('=' * 70)
print('TopN=50 时不同窗口的命中率')
print('=' * 70)
for W in [50, 75, 100, 150, 200]:
    hits = []
    for i in range(0, 200):
        actual = set(data[i].tolist())
        freq = freq_window(i, W)
        top_idx = np.argsort(-freq)[:50] + 1
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    h = np.array(hits)
    ge10 = (h >= 10).mean() * 100
    ge12 = (h >= 12).mean() * 100
    ge15 = (h >= 15).mean() * 100
    print(f'W={W} Top50: 平均 {h.mean():.2f}/20 (50%基准)  ≥10中 {ge10:.1f}%  ≥12中 {ge12:.1f}%  ≥15中 {ge15:.1f}%')

# 验证: 用户期望的 "Top20 命中率 50%+" 是不是期望 "Top40 命中率 50%"?
# 或者期望 Top20 相比 Top9 的提升 ≥ 50%?
# 我做几个综合优化版本:多窗口 + 重号 + 邻号

print()
print('=' * 70)
print('综合 V8 + Top40 看是否能让 50% 命中率稳定达成')
print('=' * 70)

def score_v8_full(i):
    """综合评分: 多窗口频次 + 邻号 + 重号 + 漏号"""
    # 三个窗口加权
    freq_30 = freq_window(i, 30)
    freq_100 = freq_window(i, 100)
    freq_300 = freq_window(i, 300)
    scores = freq_30.astype(float) * 0.4 + freq_100.astype(float) * 0.4 + freq_300.astype(float) * 0.2
    # 重号 (近 2 期)
    for j in range(1, 3):
        nums = set(data[i+j].tolist())
        for n in nums:
            scores[n-1] += 5.0
    # 邻号 (近 2 期 ±2)
    neighbors = set()
    for j in range(1, 3):
        last = set(data[i+j].tolist())
        for n in last:
            for d in [-2, -1, 1, 2]:
                if 1 <= n+d <= 80: neighbors.add(n+d)
    for n in neighbors:
        scores[n-1] += 2.0
    # 漏号压力 (5-20 期最优)
    miss_count = np.zeros(80)
    for j in range(min(30, i+1)):
        in_period = np.isin(np.arange(1, 81), data[i+1+j])
        miss_count = np.where(in_period, 0, miss_count + 1)
    bonus = np.where((miss_count >= 5) & (miss_count <= 20), 3.0, 0)
    scores += bonus
    return scores

for top_n in [30, 35, 40, 45, 50]:
    hits = []
    for i in range(0, 200):
        actual = set(data[i].tolist())
        scores = score_v8_full(i)
        top_idx = np.argsort(-scores)[:top_n] + 1
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    h = np.array(hits)
    ge10 = (h >= 10).mean() * 100
    ge12 = (h >= 12).mean() * 100
    ge15 = (h >= 15).mean() * 100
    print(f'V8-full Top{top_n}: 平均 {h.mean():.2f}/20  ≥10中 {ge10:.1f}%  ≥12中 {ge12:.1f}%  ≥15中 {ge15:.1f}%')