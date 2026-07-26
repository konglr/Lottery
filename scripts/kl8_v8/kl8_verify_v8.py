"""
KL8 策略 V8 最终验证
最佳候选: 多期邻号 + 频次 Top30
+ 验证稳定性
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

def freq_window(start_idx, window):
    if start_idx + window > n_periods:
        window = n_periods - start_idx
    if window <= 0:
        return np.zeros(80, dtype=int)
    block = data[start_idx+1:start_idx+1+window]
    flat = block.flatten()
    return np.bincount(flat, minlength=81)[1:81]

# 最佳策略: 多期邻号 + 频次 TopN
def score_v8(i, freq_W=100, neighbor_span=2, repeat_span=2, neighbor_w=3.0, repeat_w=5.0):
    freq = freq_window(i, freq_W).astype(float)
    scores = freq.copy()
    # 邻号 (近 span 期 ±2)
    neighbors = set()
    for j in range(1, neighbor_span+1):
        last = set(data[i+j].tolist())
        for n in last:
            for d in [-2, -1, 1, 2]:
                if 1 <= n+d <= 80: neighbors.add(n+d)
    for n in (neighbors):
        scores[n-1] += neighbor_w
    # 重号 (近 span 期)
    for j in range(1, repeat_span+1):
        nums = set(data[i+j].tolist())
        for n in nums:
            scores[n-1] += repeat_w
    return scores

# 在不同时间窗口上验证稳定性
print('=' * 70)
print('最佳策略 V8 稳定性验证')
print('=' * 70)
print('配置: 频次(100) + 邻号(2 期) + 重号(2 期)')
print()

windows_to_test = [
    ('最近 50 期 (索引 0-49)', list(range(0, 50))),
    ('最近 100 期 (索引 0-99)', list(range(0, 100))),
    ('最近 200 期 (索引 0-199)', list(range(0, 200))),
    ('中间 100 期 (索引 100-199)', list(range(100, 200))),
    ('早期 100 期 (索引 200-299)', list(range(200, 300))),
]
for name, indices in windows_to_test:
    print(f'\n{name}:')
    for top_n in [9, 15, 20, 25, 30]:
        hits = []
        for i in indices:
            actual = set(data[i].tolist())
            scores = score_v8(i)
            top_idx = np.argsort(-scores)[:top_n] + 1
            top_set = set(top_idx.tolist())
            hits.append(len(top_set & actual))
        h = np.array(hits)
        ge5 = (h >= 5).mean() * 100
        ge8 = (h >= 8).mean() * 100
        ge10 = (h >= 10).mean() * 100
        print(f'  Top{top_n}: 平均 {h.mean():.2f}/20 ({h.mean()*5:.1f}%)  ≥5中 {ge5:.1f}%  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%')

# 现在做对比: 朴素频次 vs V8 在 Top30 上
print()
print('=' * 70)
print('朴素 vs V8 在 Top30 的对比 (最近 200 期)')
print('=' * 70)
print(f'{"策略":>30} {"平均":>8} {"≥10中":>8} {"≥12中":>8}')
indices = list(range(0, 200))

# 朴素 W=100
hits = []
for i in indices:
    actual = set(data[i].tolist())
    freq = freq_window(i, 100)
    top_idx = np.argsort(-freq)[:30] + 1
    top_set = set(top_idx.tolist())
    hits.append(len(top_set & actual))
h = np.array(hits)
print(f'{"朴素 W=100 Top30":>30} {h.mean():>8.2f} {(h>=10).mean()*100:>7.1f}% {(h>=12).mean()*100:>7.1f}%')

# V8
hits = []
for i in indices:
    actual = set(data[i].tolist())
    scores = score_v8(i)
    top_idx = np.argsort(-scores)[:30] + 1
    top_set = set(top_idx.tolist())
    hits.append(len(top_set & actual))
h = np.array(hits)
print(f'{"V8 Top30":>30} {h.mean():>8.2f} {(h>=10).mean()*100:>7.1f}% {(h>=12).mean()*100:>7.1f}%')

# 在 2026195 这一期上的对比
print()
print('=' * 70)
print('在 2026195 期 (索引 0) 上的实测')
print('=' * 70)
actual = set(data[0].tolist())
print(f'2026195 实际: {sorted(actual)}')

scores_v8 = score_v8(0)
top30 = sorted(np.argsort(-scores_v8)[:30].tolist())
top30_v8 = [t+1 for t in top30]
print(f'\nV8 Top30: {top30_v8}')
print(f'V8 Top30 命中: {sorted(set(top30_v8) & actual)} = {len(set(top30_v8) & actual)}/20 = {len(set(top30_v8) & actual)*5:.1f}%')

# 朴素 W=100 Top30
freq = freq_window(0, 100)
top30_naive = sorted((np.argsort(-freq)[:30] + 1).tolist())
print(f'\n朴素 W=100 Top30: {top30_naive}')
print(f'朴素命中: {sorted(set(top30_naive) & actual)} = {len(set(top30_naive) & actual)}/20 = {len(set(top30_naive) & actual)*5:.1f}%')

# 100 期频次 Top25
top25 = sorted((np.argsort(-freq)[:25] + 1).tolist())
print(f'\n朴素 W=100 Top25: {top25}')
print(f'命中: {sorted(set(top25) & actual)} = {len(set(top25) & actual)}/20 = {len(set(top25) & actual)*100/20:.1f}%')

# 保存结果
out = {
    'best_strategy': 'V8: freq(100) + neighbor(span=2) + repeat(span=2)',
    'test_period_count': 200,
    'top30_avg_hits': float(np.mean(hits)),
    'top30_ge10_prob': float((np.array(hits)>=10).mean()),
}
with open('/tmp/kl8_v8_verify.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nDone.')