"""
KL8 V10 最终验证: 在 2026195 实测 + 2026196 预测
对比 V3 / V9 / V10
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

# 2026195 = 索引 0, 2026196 = 索引 -1 (预测)
# 用全部历史数据 (索引 0-2016) 来预测 2026196
def score_v10(i, w_freq=10, w_repeat=3, w_neighbor=5, wf=-3, wr=-3, wn=-3):
    """V10 反向特征评分"""
    s = (feat_freq(i, w_freq) * wf +
         feat_repeat(i, w_repeat) * wr +
         feat_neighbor(i, w_neighbor) * wn)
    return s

def score_v3(i, w_freq=50):
    """V3 等价: 朴素频次"""
    return feat_freq(i, w_freq)

# 2026195 实测
print('=' * 70)
print('2026195 期 V3 vs V10 实测对比')
print('=' * 70)
actual_2026195 = set(data[0].tolist())
print(f'2026195 实际开奖: {sorted(actual_2026195)}')
print()

for top_n in [9, 15, 20, 25, 30]:
    # V3 (朴素频次 W=50)
    s_v3 = score_v3(0, w_freq=50)
    top_v3 = sorted((np.argsort(-s_v3)[:top_n] + 1).tolist())
    hits_v3 = len(set(top_v3) & actual_2026195)
    # V10 (反向)
    s_v10 = score_v10(0)
    top_v10 = sorted((np.argsort(-s_v10)[:top_n] + 1).tolist())
    hits_v10 = len(set(top_v10) & actual_2026195)
    print(f'Top{top_n}: V3 (freq50) = {hits_v3}/20 ({hits_v3*5}%) | V10 (反向) = {hits_v10}/20 ({hits_v10*5}%)')

# 2026196 预测
print()
print('=' * 70)
print('2026196 期 V10 预测 (Top20 / Top30)')
print('=' * 70)

# data[0] = 2026195, data[-1] 不存在
# 我们把 2026196 当作"未来期",索引用 -1 (不存在)
# 实际处理: 把索引 0 当作 2026196, data[1] = 2026195

# 但实际数据 data[0] = 2026195,所以:
# 想预测 2026196 = 把"当前期"当作不存在的 -1
# 需要在 i = -1 时使用 data[0:...] 作为历史

# 简单做法: 模拟 2026196 = 把 data 当作[i+1] 开始, i = -1
def feat_freq_neg1(window):
    """i = -1 的频次 = data[0:window]"""
    if window > n_periods: window = n_periods
    block = data[:window]
    flat = block.flatten()
    return np.bincount(flat, minlength=81)[1:81].astype(float)

def feat_repeat_neg1(span):
    counts = np.zeros(80, dtype=float)
    for j in range(span):
        nums = data[j]
        counts[nums-1] += 1
    return counts

def feat_neighbor_neg1(span, distance=2):
    counts = np.zeros(80, dtype=float)
    for j in range(span):
        last = data[j]
        for n in last:
            for d in range(-distance, distance+1):
                if d == 0: continue
                if 1 <= n+d <= 80:
                    counts[n+d-1] += 1
    return counts

# 2026196 预测 (V10 反向)
s_v10_2026196 = (feat_freq_neg1(10) * -3 +
                  feat_repeat_neg1(3) * -3 +
                  feat_neighbor_neg1(5) * -3)

top30_v10 = sorted((np.argsort(-s_v10_2026196)[:30] + 1).tolist())
top20_v10 = top30_v10[:20]

print(f'V10 Top20: {top20_v10}')
print(f'V10 Top30: {top30_v10}')
print()

# 形态
def morphology(nums):
    nums = sorted(nums)
    sum_v = sum(nums)
    span = max(nums) - min(nums)
    z1 = sum(1 for n in nums if n <= 27)
    z2 = sum(1 for n in nums if 28 <= n <= 53)
    z3 = sum(1 for n in nums if n >= 54)
    odd = sum(1 for n in nums if n % 2 == 1)
    big = sum(1 for n in nums if n > 40)
    consec = sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
    return f'和值={sum_v}, 跨度={span}, 三区={z1}:{z2}:{z3}, 奇偶={odd}:{len(nums)-odd}, 大小={big}:{len(nums)-big}, 连号对={consec}'

print(f'V10 Top20 形态: {morphology(top20_v10)}')
print(f'V10 Top30 形态: {morphology(top30_v10)}')

# 也对比 朴素 freq(100) Top20 (基线 V3)
freq100_neg1 = feat_freq_neg1(100)
top20_naive = sorted((np.argsort(-freq100_neg1)[:20] + 1).tolist())
print()
print(f'朴素 freq(100) Top20: {top20_naive}')
print(f'朴素 freq(100) Top20 形态: {morphology(top20_naive)}')

# 共识度
set_a = set(top20_v10)
set_b = set(top20_naive)
print()
print(f'V10 vs 朴素 Top20 共识: {sorted(set_a & set_b)} ({len(set_a & set_b)}/20)')

# 保存
out = {
    'target_period': '2026196',
    'predict_time': '2026-07-25 12:24 GMT+8',
    'last_period': '2026195',
    'last_nums': sorted(data[0].tolist()),
    'strategy_v10': {
        'config': 'freq(10, w=-3) + repeat(3, w=-3) + neighbor(5, w=-3)',
        'description': 'V10 反向特征组合 — 历史回测 avg=5.425/20 (27.13%)',
        'Top20': top20_v10,
        'Top30': top30_v10,
    },
    'strategy_baseline': {
        'config': 'freq(100, w=1.0)',
        'description': '朴素频次 W=100 (基线)',
        'Top20': top20_naive,
    },
    'consensus_top20': {
        'both': sorted(set_a & set_b),
        'only_v10': sorted(set_a - set_b),
        'only_naive': sorted(set_b - set_a),
    },
    'morphology_v10_top20': morphology(top20_v10),
    'morphology_v10_top30': morphology(top30_v10),
    'backtest_200periods': {
        'method': 'V10 freq(10,-3) + repeat(3,-3) + neighbor(5,-3)',
        'top20_avg': 5.425,
        'top20_avg_pct': 27.13,
        'top20_ge8': 12.0,
        'top20_ge10': 1.0,
        'top25_avg': 6.655,
        'top25_ge8': 31.5,
        'top25_ge10': 6.0,
        'top30_avg': 7.840,
        'top30_ge8': 57.5,
        'top30_ge10': 15.5,
    }
}
with open('/Users/clarkkong/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery/data/backtest/2026196_predictions_v10.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('✅ 2026196 预测已保存到 data/backtest/2026196_predictions_v10.json')