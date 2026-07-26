"""
KL8 快速回测脚本
向量化,避免逐号循环
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

# 转换为 numpy 数组 (n_periods, 20)
data = df[red_cols].astype(int).values  # (n, 20)
n_periods = len(data)
print(f'数据: {n_periods} 期')

# 频次矩阵: freq[n_periods, 80] = 前 n 期每个号出现次数
# 但我们滑动窗口,需要快速计算
# 用 cumulative count trick
# 简化为: 用最近 K 期的 numpy 频次

def freq_window(start_idx, window, target_n=80):
    """返回 (80,) array,start_idx 之前 window 期内每个号出现次数"""
    if start_idx + window > n_periods:
        window = n_periods - start_idx
    if window <= 0:
        return np.zeros(target_n, dtype=int)
    block = data[start_idx+1:start_idx+1+window]  # (window, 20)
    flat = block.flatten()
    freq = np.bincount(flat, minlength=target_n+1)[1:target_n+1]
    return freq

# 测试范围: 索引 200-400 (足够历史,且预测期足够新)
test_indices = list(range(200, 400))
print(f'测试 {len(test_indices)} 期,期号 {df["期号"].iloc[test_indices[0]]} -> {df["期号"].iloc[test_indices[-1]]}')

# ============= 朴素频次 =============
print()
print('=' * 70)
print('朴素频次 (单窗口)')
print('=' * 70)
print(f'{"W":>5} {"TopN":>5} {"Avg":>6} {"≥8中":>6} {"≥10中":>7} {"≥11中":>7}')
results = {}
for W in [30, 50, 75, 100, 150, 200]:
    for top_n in [9, 15, 20, 25, 30]:
        hits = []
        for i in test_indices:
            actual = set(data[i].tolist())
            freq = freq_window(i, W)
            top_idx = np.argsort(-freq)[:top_n] + 1  # 转回 1-80
            top_set = set(top_idx.tolist())
            hits.append(len(top_set & actual))
        hits = np.array(hits)
        avg = hits.mean()
        ge8 = (hits >= 8).mean() * 100
        ge10 = (hits >= 10).mean() * 100
        ge11 = (hits >= 11).mean() * 100
        results[(W, top_n)] = (avg, ge8, ge10, ge11)
        print(f'{W:>5} {top_n:>5} {avg:>6.2f} {ge8:>5.1f}% {ge10:>6.2f}% {ge11:>6.2f}%')

# ============= 多窗口加权频次 =============
print()
print('=' * 70)
print('多窗口加权频次')
print('=' * 70)
print(f'{"配置":>30} {"TopN":>5} {"Avg":>6} {"≥8中":>6} {"≥10中":>7}')
configs = [
    ('[30, 100] 均权', [(30, 1.0), (100, 1.0)]),
    ('[10, 30, 100] 均权', [(10, 1.0), (30, 1.0), (100, 1.0)]),
    ('[30, 100, 200] 均权', [(30, 1.0), (100, 1.0), (200, 1.0)]),
    ('[50, 150, 300]', [(50, 1.0), (150, 1.0), (300, 1.0)]),
    ('[30]*2 + [100]', [(30, 2.0), (100, 1.0)]),
    ('[20, 50, 100, 200]', [(20, 1.0), (50, 1.0), (100, 1.0), (200, 1.0)]),
]
for name, windows in configs:
    for top_n in [20, 25, 30]:
        hits = []
        for i in test_indices:
            actual = set(data[i].tolist())
            scores = np.zeros(80)
            for W, weight in windows:
                freq = freq_window(i, W)
                scores += freq * weight
            top_idx = np.argsort(-scores)[:top_n] + 1
            top_set = set(top_idx.tolist())
            hits.append(len(top_set & actual))
        hits = np.array(hits)
        avg = hits.mean()
        ge8 = (hits >= 8).mean() * 100
        ge10 = (hits >= 10).mean() * 100
        print(f'{name:>30} {top_n:>5} {avg:>6.2f} {ge8:>5.1f}% {ge10:>6.2f}%')

# ============= 重号加成 =============
print()
print('=' * 70)
print('重号加成 (近 1-3 期重号)')
print('=' * 70)
for span in [1, 2, 3]:
    for top_n in [20, 25, 30]:
        hits = []
        for i in test_indices:
            actual = set(data[i].tolist())
            freq = freq_window(i, 100)
            scores = freq.astype(float).copy()
            # 加权近 span 期出现的号
            for j in range(1, span+1):
                nums = set(data[i+j].tolist())
                for n in nums:
                    scores[n-1] += 5.0
            top_idx = np.argsort(-scores)[:top_n] + 1
            top_set = set(top_idx.tolist())
            hits.append(len(top_set & actual))
        hits = np.array(hits)
        avg = hits.mean()
        ge8 = (hits >= 8).mean() * 100
        ge10 = (hits >= 10).mean() * 100
        print(f'span={span} Top{top_n}: 平均 {avg:.2f}/20  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%')

# ============= 邻号加成 =============
print()
print('=' * 70)
print('邻号加成 (上期 ±2 邻号)')
print('=' * 70)
for top_n in [20, 25, 30]:
    hits = []
    for i in test_indices:
        actual = set(data[i].tolist())
        freq = freq_window(i, 100)
        scores = freq.astype(float).copy()
        last = set(data[i+1].tolist())
        # 上期号加权
        for n in last:
            scores[n-1] += 3.0
        # 邻号加权
        neighbors = set()
        for n in last:
            for d in [-2, -1, 1, 2]:
                if 1 <= n+d <= 80:
                    neighbors.add(n+d)
        for n in neighbors - last:
            scores[n-1] += 1.5
        top_idx = np.argsort(-scores)[:top_n] + 1
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    hits = np.array(hits)
    avg = hits.mean()
    ge8 = (hits >= 8).mean() * 100
    ge10 = (hits >= 10).mean() * 100
    print(f'上期+邻号 Top{top_n}: 平均 {avg:.2f}/20  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%')

# ============= 漏号压力 =============
print()
print('=' * 70)
print('漏号压力 (适中期号优先)')
print('=' * 70)
for top_n in [20, 25, 30]:
    hits = []
    for i in test_indices:
        actual = set(data[i].tolist())
        freq = freq_window(i, 100)
        scores = freq.astype(float).copy()
        # 漏号: 对每个号,统计近 50 期连续未出期数
        miss_count = np.zeros(80)
        for j in range(min(50, i+1)):
            if j == i+1: break
            in_period = np.isin(np.arange(1, 81), data[i+1+j])
            miss_count = np.where(in_period, 0, miss_count + 1)
        # 适中期号 (3-15 期未出) 加分
        bonus = np.where((miss_count >= 3) & (miss_count <= 15), 5.0, 0)
        bonus += np.where(miss_count >= 25, -3.0, 0)
        scores += bonus
        top_idx = np.argsort(-scores)[:top_n] + 1
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    hits = np.array(hits)
    avg = hits.mean()
    ge8 = (hits >= 8).mean() * 100
    ge10 = (hits >= 10).mean() * 100
    print(f'漏号压力 Top{top_n}: 平均 {avg:.2f}/20  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%')

# ============= 综合 V8 策略 =============
print()
print('=' * 70)
print('综合 V8: 频次(100) + 重号(3 期) + 邻号 + 漏号 + 跨期(5)')
print('=' * 70)
for top_n in [20, 25, 30]:
    hits = []
    for i in test_indices:
        actual = set(data[i].tolist())
        # 频次
        freq100 = freq_window(i, 100)
        scores = freq100.astype(float).copy() * 1.0
        # 重号 (近 3 期)
        for j in range(1, 4):
            nums = set(data[i+j].tolist())
            for n in nums:
                scores[n-1] += 3.0
        # 邻号 (上期)
        last = set(data[i+1].tolist())
        neighbors = set()
        for n in last:
            for d in [-2, -1, 1, 2]:
                if 1 <= n+d <= 80: neighbors.add(n+d)
        for n in (neighbors - last):
            scores[n-1] += 2.0
        # 漏号
        miss_count = np.zeros(80)
        for j in range(min(30, i+1)):
            in_period = np.isin(np.arange(1, 81), data[i+1+j])
            miss_count = np.where(in_period, 0, miss_count + 1)
        bonus = np.where((miss_count >= 5) & (miss_count <= 20), 2.0, 0)
        scores += bonus
        top_idx = np.argsort(-scores)[:top_n] + 1
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    hits = np.array(hits)
    avg = hits.mean()
    ge8 = (hits >= 8).mean() * 100
    ge10 = (hits >= 10).mean() * 100
    ge11 = (hits >= 11).mean() * 100
    print(f'V8 Top{top_n}: 平均 {avg:.2f}/20  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%  ≥11中 {ge11:.1f}%')

# 存结果
import json
out = {
    'test_period_count': len(test_indices),
    'test_period_range': f'{df["期号"].iloc[test_indices[0]]} -> {df["期号"].iloc[test_indices[-1]]}',
}
with open('/tmp/kl8_backtest_v8_results.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print()
print('Done.')