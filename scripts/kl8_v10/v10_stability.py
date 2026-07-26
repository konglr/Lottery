"""
KL8 V10 长期稳定性验证 + 多窗口回测
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
                if 1 <= n+d <= 80:
                    counts[n+d-1] += 1
    return counts

def score_v10(i):
    return (feat_freq(i, 10) * -3 +
            feat_repeat(i, 3) * -3 +
            feat_neighbor(i, 5) * -3)

def score_v3(i):
    return feat_freq(i, 50)  # 朴素频次 W=50

# 不同时间窗口的稳定性测试
print('=' * 70)
print('V10 vs V3 长期稳定性 (不同时间窗口)')
print('=' * 70)
windows_to_test = [
    ('最近 50 期', range(0, 50)),
    ('最近 100 期', range(0, 100)),
    ('最近 200 期', range(0, 200)),
    ('最近 400 期', range(0, 400)),
    ('中间 100 期 (100-199)', range(100, 200)),
    ('早期 100 期 (300-399)', range(300, 400)),
]

for name, indices in windows_to_test:
    print(f'\n{name}:')
    for top_n in [9, 15, 20, 25, 30]:
        v3_hits = []
        v10_hits = []
        for i in indices:
            actual = set(data[i].tolist())
            # V3
            s = score_v3(i)
            top_v3 = set((np.argsort(-s)[:top_n] + 1).tolist())
            v3_hits.append(len(top_v3 & actual))
            # V10
            s = score_v10(i)
            top_v10 = set((np.argsort(-s)[:top_n] + 1).tolist())
            v10_hits.append(len(top_v10 & actual))
        v3_avg = np.mean(v3_hits)
        v10_avg = np.mean(v10_hits)
        v3_ge8 = sum(1 for h in v3_hits if h >= 8) / len(v3_hits) * 100
        v10_ge8 = sum(1 for h in v10_hits if h >= 8) / len(v10_hits) * 100
        delta = (v10_avg - v3_avg) / v3_avg * 100 if v3_avg > 0 else 0
        winner = 'V10' if v10_avg > v3_avg else 'V3'
        print(f'  Top{top_n}: V3={v3_avg:.3f}/20 ({v3_avg*5:.1f}%, ≥8中 {v3_ge8:.1f}%) | '
              f'V10={v10_avg:.3f}/20 ({v10_avg*5:.1f}%, ≥8中 {v10_ge8:.1f}%) | Δ={delta:+.1f}% → {winner}')

# 关键洞察: V10 的稳定性
print()
print('=' * 70)
print('V10 稳定性分析')
print('=' * 70)
print()
print('观察: V10 在最近 50/100/200/400 期上的 Top20 表现')
print('期望: 如果策略稳定,各窗口 avg 应该相近')
print()

# 全 400 期回测的胜率
v10_wins = 0
v3_wins = 0
total = 0
for i in range(0, 400):
    actual = set(data[i].tolist())
    s_v3 = score_v3(i)
    s_v10 = score_v10(i)
    for top_n in [9, 15, 20, 25, 30]:
        top_v3 = set((np.argsort(-s_v3)[:top_n] + 1).tolist())
        top_v10 = set((np.argsort(-s_v10)[:top_n] + 1).tolist())
        h_v3 = len(top_v3 & actual)
        h_v10 = len(top_v10 & actual)
        if h_v10 > h_v3: v10_wins += 1
        elif h_v3 > h_v10: v3_wins += 1
        total += 1

print(f'400 期 × 5 TopN = {total} 次对比')
print(f'V10 胜: {v10_wins} ({v10_wins/total*100:.1f}%)')
print(f'V3 胜: {v3_wins} ({v3_wins/total*100:.1f}%)')
print(f'平局: {total - v10_wins - v3_wins}')

# 保存
out = {
    'method': 'v10_stability_test',
    'test_periods': 400,
    'v10_wins': v10_wins,
    'v3_wins': v3_wins,
    'ties': total - v10_wins - v3_wins,
    'v10_win_rate': v10_wins / total,
}
with open('/tmp/kl8_v10_stability.json', 'w') as f:
    json.dump(out, f, indent=2)
print()
print('稳定性数据保存到 /tmp/kl8_v10_stability.json')