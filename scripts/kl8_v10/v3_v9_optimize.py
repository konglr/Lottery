"""
KL8 V3-V9 特征工程优化
目标: 在 V3-V9 特征框架内, 找最优特征组合 + 权重 + 窗口, 让 Top20 命中率最大化
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
from collections import Counter
import json

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')

red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values  # (n, 20)
n_periods = len(data)
print(f'数据: {n_periods} 期')

# ============================================================
# 6 大特征函数 (向量化, 一次算所有号)
# ============================================================

def feat_freq(i, window):
    """F1 频次 (近 window 期每个号出现次数)"""
    if i + window > n_periods:
        window = n_periods - i
    if window <= 0:
        return np.zeros(80, dtype=float)
    block = data[i+1:i+1+window]
    flat = block.flatten()
    return np.bincount(flat, minlength=81)[1:81].astype(float)

def feat_repeat(i, span):
    """F2 重号 (近 span 期号加权, 重号次数)"""
    counts = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if i + j >= n_periods: break
        nums = data[i+j]
        counts[nums-1] += 1
    return counts

def feat_neighbor(i, span, distance=2):
    """F3 邻号 (近 span 期所有 ±distance 邻号加权)"""
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

def feat_miss(i, max_window):
    """F4 漏号 (近 max_window 期连续未出次数)"""
    miss_count = np.zeros(80, dtype=float)
    for j in range(min(max_window, i+1)):
        nums = data[i+1+j]
        in_period = np.zeros(80, dtype=bool)
        in_period[nums-1] = True
        miss_count = np.where(in_period, 0, miss_count + 1)
    return miss_count

def feat_span(i, span):
    """F5 跨期回溯 (近 span 期出现总次数, 但每期最多 +1)"""
    counts = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if i + j >= n_periods: break
        nums = data[i+j]
        counts[nums-1] += 1
    return counts

def feat_recent_freq(i, window):
    """F6 近期频次 (近 window 期频次, 短窗口更敏感)"""
    return feat_freq(i, window)

# ============================================================
# 评分函数: 加权多个特征
# ============================================================

def score_combination(i, config):
    """
    config: list of (feature_name, span/window, weight)
    例: [('freq', 100, 1.0), ('repeat', 2, 0.5), ('neighbor', 2, 0.3)]
    """
    scores = np.zeros(80, dtype=float)
    for feat_name, span, weight in config:
        if feat_name == 'freq':
            s = feat_freq(i, span)
        elif feat_name == 'repeat':
            s = feat_repeat(i, span)
        elif feat_name == 'neighbor':
            s = feat_neighbor(i, span)
        elif feat_name == 'miss':
            s = feat_miss(i, span)
        elif feat_name == 'span':
            s = feat_span(i, span)
        elif feat_name == 'recent_freq':
            s = feat_recent_freq(i, span)
        else:
            raise ValueError(f'未知特征: {feat_name}')
        scores += s * weight
    return scores

# ============================================================
# 回测函数
# ============================================================

def backtest_config(config, top_n, indices):
    """评估一个 (config, top_n) 在 indices 上的平均命中"""
    hits = []
    for i in indices:
        actual = set(data[i].tolist())
        scores = score_combination(i, config)
        top_idx = np.argsort(-scores)[:top_n] + 1
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    return np.array(hits)

# ============================================================
# 测试范围: 最近 200 期
# ============================================================
test_indices = list(range(0, 200))

# ============================================================
# 基线: V3 (F1 频次 + F4 重号邻号 + F5 漏号 + F6 同尾 等权)
# ============================================================
# 简化版 V3 = freq(30) + repeat_neighbor + miss + tail
# 这里我用 freq(50) 作为基准

print('=' * 70)
print('基线测试: 朴素频次 单特征')
print('=' * 70)
baseline_results = {}
for window in [30, 50, 100, 150]:
    for top_n in [9, 15, 20, 25, 30]:
        hits = backtest_config([('freq', window, 1.0)], top_n, test_indices)
        h = np.array(hits)
        avg = h.mean()
        baseline_results[(window, top_n)] = (avg, (h>=10).mean()*100, (h>=8).mean()*100)
        print(f'  freq(W={window}) Top{top_n}: 平均 {avg:.2f}/20  ≥8中 {(h>=8).mean()*100:.1f}%  ≥10中 {(h>=10).mean()*100:.1f}%')

# ============================================================
# 双特征组合: 频次 + 重号
# ============================================================
print()
print('=' * 70)
print('双特征: 频次(W=100) + 重号(span)')
print('=' * 70)
for span in [1, 2, 3, 5]:
    for w_repeat in [2.0, 5.0, 8.0, 12.0]:
        for top_n in [9, 15, 20]:
            hits = backtest_config([('freq', 100, 1.0), ('repeat', span, w_repeat)], top_n, test_indices)
            h = np.array(hits)
            ge8 = (h >= 8).mean() * 100
            ge10 = (h >= 10).mean() * 100
            avg = h.mean()
            if avg >= 5.0:  # 只打印 Top20 平均 ≥5
                print(f'  freq(100)+repeat(span={span},w={w_repeat}) Top{top_n}: 平均 {avg:.2f}  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%')

# ============================================================
# 三特征: 频次 + 重号 + 邻号
# ============================================================
print()
print('=' * 70)
print('三特征: 频次(100) + 重号(2) + 邻号(span)')
print('=' * 70)
for nspan in [1, 2, 3]:
    for w_n in [1.0, 2.0, 3.0, 5.0]:
        for top_n in [15, 20]:
            hits = backtest_config([
                ('freq', 100, 1.0),
                ('repeat', 2, 5.0),
                ('neighbor', nspan, w_n),
            ], top_n, test_indices)
            h = np.array(hits)
            avg = h.mean()
            ge8 = (h >= 8).mean() * 100
            ge10 = (h >= 10).mean() * 100
            if avg >= 5.5:
                print(f'  freq+repeat(2,5)+neighbor({nspan},{w_n}) Top{top_n}: 平均 {avg:.2f}  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%')

# ============================================================
# 漏号特征: 频次 + 重号 + 漏号
# ============================================================
print()
print('=' * 70)
print('三特征: 频次 + 重号 + 漏号')
print('=' * 70)
for miss_window in [30, 50, 100]:
    for w_miss in [-1.0, -2.0, 1.0, 2.0]:  # 漏号反向或正向
        for top_n in [20]:
            # 漏号 = miss / avg_miss, 适中期号加分
            # 这里我们用 miss_count 直接, 用 w 控制方向
            hits = backtest_config([
                ('freq', 100, 1.0),
                ('repeat', 2, 5.0),
                ('miss', miss_window, w_miss),
            ], top_n, test_indices)
            h = np.array(hits)
            avg = h.mean()
            ge8 = (h >= 8).mean() * 100
            ge10 = (h >= 10).mean() * 100
            print(f'  freq+repeat(2,5)+miss(W={miss_window},w={w_miss}) Top{top_n}: 平均 {avg:.2f}  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%')

# 保存初步结果
out = {
    'test_periods': 200,
    'baseline_naive_freq': {f'W{w}_Top{n}': {'avg': float(v[0]), 'ge10_pct': float(v[1])} 
                            for (w, n), v in baseline_results.items()}
}
with open('/tmp/kl8_v3v9_optimize_step1.json', 'w') as f:
    json.dump(out, f, indent=2)
print()
print('Step 1 完成, 结果保存到 /tmp/kl8_v3v9_optimize_step1.json')